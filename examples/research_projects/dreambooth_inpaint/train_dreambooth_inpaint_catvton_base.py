import argparse
import itertools
import math
import os
# 设置huggingface镜像
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from catvton_base_infer import run_inference
import json
from unet_adapter import adapt_unet_with_catvton_attn

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)


def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def collate_fn(examples):
    # 收集所有图像数据
    real_images = torch.stack([example["real_images"] for example in examples])
    real_masks = torch.stack([example["real_masks"] for example in examples])
    # 判断是否存在对应的Key
    if "cloth_warp_images" in examples[0]:
        cloth_warp_images = torch.stack([example["cloth_warp_images"] for example in examples])
        cloth_warp_masks = torch.stack([example["cloth_warp_masks"] for example in examples])
    else:
        cloth_warp_images = None
        cloth_warp_masks = None
    if "condition_images" in examples[0]:
        condition_images = torch.stack([example["condition_images"] for example in examples])
    else:
        condition_images = None
    if "openpose_images" in examples[0]:
        openpose_images = torch.stack([example["openpose_images"] for example in examples])
    else:
        openpose_images = None
    if "canny_images" in examples[0]:
        canny_images = torch.stack([example["canny_images"] for example in examples])
    else:
        canny_images = None
    # 确保数据格式正确
    real_images = real_images.to(memory_format=torch.contiguous_format).float()
    real_masks = real_masks.to(memory_format=torch.contiguous_format).float()
    if cloth_warp_images is not None:
        cloth_warp_images = cloth_warp_images.to(memory_format=torch.contiguous_format).float()
        cloth_warp_masks = cloth_warp_masks.to(memory_format=torch.contiguous_format).float()
    if condition_images is not None:
        condition_images = condition_images.to(memory_format=torch.contiguous_format).float()
    if openpose_images is not None:
        openpose_images = openpose_images.to(memory_format=torch.contiguous_format).float()
    if canny_images is not None:
        canny_images = canny_images.to(memory_format=torch.contiguous_format).float()

    batch = {
        "real_images": real_images,
        "real_masks": real_masks,
        "cloth_warp_images": cloth_warp_images,
        "cloth_warp_masks": cloth_warp_masks,
        "condition_images": condition_images,
        "openpose_images": openpose_images,
        "canny_images": canny_images,
    }


    return batch
# generate random masks
def random_mask(im_shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--catvton_attn_path",
        type=str,
        default=None,
        help="Path to catvton unet attention weights",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从指定检查点恢复训练，可以是检查点路径或'latest'",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help="运行验证的步数间隔。如果设置，将每隔指定步数生成测试图片。",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="用于验证的提示词。",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help="用于验证的输入图片路径。",
    )
    parser.add_argument(
        "--validation_mask",
        type=str,
        default=None,
        help="用于验证的mask图片路径。",
    )
    parser.add_argument(
        "--validation_condition_image",
        type=str,
        default=None,
        help="用于验证的条件图片路径。",
    )
    parser.add_argument(
        "--validation_canny_image",
        type=str,
        default=None,
        help="用于验证的canny图片路径。",
    )

    parser.add_argument(
        "--valid_cloth_warp_image",
        type=str,
        default=None,
        help="用于验证的变形服装图片路径。",
    )
    parser.add_argument(
        "--valid_cloth_warp_mask",
        type=str,
        default=None,
        help="用于验证的变形服装掩码路径。",
    )
    parser.add_argument(
        "--random_transform",
        action="store_true",
        help="是否使用随机变换（随机裁剪等）。如果不设置，则只进行居中裁剪。",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="每训练多少步保存一次模型",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="最多保存多少个模型检查点，超过后会删除最早的检查点",
    )
    parser.add_argument(
        "--trainable_modules",
        type=str,
        default="attention",
        help="要训练的模块，多个模块用分号分隔",
    )
    parser.add_argument(
        "--use_openpose_conditioning",
        action="store_true",
        help="是否使用openpose条件控制",
    )
    parser.add_argument(
        "--use_warp_cloth",
        action="store_true",
        help="是否使用变形服装",
    )
    parser.add_argument(
        "--dream_lambda",
        type=float,
        default=10.0,
        help="DREAM loss的权重系数λ",
    )
    parser.add_argument(
        "--condition_image_drop_out",
        type=float,
        default=0.0,
        help="将condition image变为全黑的概率",
    )
    parser.add_argument(
        "--cloth_warp_drop_out",
        type=float,
        default=0.0,
        help="不使用cloth warp的概率",
    )

    parser.add_argument(
        "--use_canny_conditioning",
        action="store_true",
        help="是否使用canny conditioning",
    )
    parser.add_argument(
        "--canny_conditioning_type",
        type=int,
        default=1,
        help="canny conditioning的类型",
    )

    parser.add_argument(
        "--latent_append_num",
        type=int,
        default=1,
        help="使用的latent数量：1=CatVTON, 2=CatVTON+Openpose, 3=CatVTON+Canny+Openpose",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")

    return args


class DreamBoothDataset(Dataset):
    """
    数据集类用于准备训练数据:
    - real_images: RGB真实图像
    - real_masks: 1通道掩码图像
    - cloth_warp_images: RGB变形服装图像
    - cloth_warp_masks: 1通道变形服装掩码
    - condition_images: RGB条件图像
    - openpose_images: RGB姿态图像
    - canny_images: RGB边缘图像
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        size=512,
        center_crop=False,
        random_transform=False,
        canny_conditioning_type=1,
        use_canny_conditioning=False,
        use_openpose_conditioning=False,
        use_warp_cloth=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.random_transform = random_transform
        self.tokenizer = tokenizer
        self.use_warp_cloth = use_warp_cloth
        self.use_openpose_conditioning = use_openpose_conditioning
        self.use_canny_conditioning = use_canny_conditioning
        self.canny_conditioning_type = canny_conditioning_type

        # 获取所有图片路径
        self.real_images_path = sorted(Path(os.path.join(instance_data_root, "real_images")).iterdir())
        self.real_masks_path = sorted(Path(os.path.join(instance_data_root, "real_masks")).iterdir())
        if use_warp_cloth:
            self.cloth_warp_images_path = sorted(Path(os.path.join(instance_data_root, "cloth_warp_images")).iterdir())
            self.cloth_warp_masks_path = sorted(Path(os.path.join(instance_data_root, "cloth_warp_masks")).iterdir())
        else:
            self.cloth_warp_masks_path = None
            self.cloth_warp_images_path = None
        if use_openpose_conditioning:
            self.openpose_images_path = sorted(Path(os.path.join(instance_data_root, "openpose_images")).iterdir())
        else:
            self.openpose_images_path = None

        self.condition_images_path = sorted(Path(os.path.join(instance_data_root, "condition_images")).iterdir())

        
        canny_dir = "canny_images_2" if canny_conditioning_type == 2 else "canny_images"
        if use_canny_conditioning:
            self.canny_images_path = sorted(Path(os.path.join(instance_data_root, canny_dir)).iterdir())
        else:
            self.canny_images_path = None

        check_list = [
            self.real_images_path,
            self.real_masks_path,
            self.condition_images_path,
        ]

        if use_warp_cloth:
            check_list.append(self.cloth_warp_masks_path)
            check_list.append(self.cloth_warp_images_path)
        if use_openpose_conditioning:
            check_list.append(self.openpose_images_path)
        if use_canny_conditioning:
            check_list.append(self.canny_images_path)
            

        
        # 确保所有文件夹中的图片数量相同
        num_images = len(self.real_images_path)
        assert all(len(x) == num_images for x in check_list), "所有文件夹中的图片数量必须相同"
        
        self.num_images = num_images
        self.instance_prompt = instance_prompt

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        example = {}
        
        # 加载所有图像
        real_image = Image.open(self.real_images_path[index])
        real_mask = Image.open(self.real_masks_path[index])
        if self.use_warp_cloth:
            cloth_warp_image = Image.open(self.cloth_warp_images_path[index])
            cloth_warp_mask = Image.open(self.cloth_warp_masks_path[index])
        else:
            cloth_warp_image = None
            cloth_warp_mask = None

        condition_image = Image.open(self.condition_images_path[index])

        if self.use_openpose_conditioning:
            openpose_image = Image.open(self.openpose_images_path[index])
        else:
            openpose_image = None
        if self.use_canny_conditioning:
            canny_image = Image.open(self.canny_images_path[index])
        else:
            canny_image = None
        # 转换图像模式
        if not real_mask.mode == "L":
            real_mask = real_mask.convert("L")
        if cloth_warp_image is not None and not cloth_warp_image.mode == "RGB":
            cloth_warp_image = cloth_warp_image.convert("RGB")
        if cloth_warp_mask is not None and not cloth_warp_mask.mode == "L":
            cloth_warp_mask = cloth_warp_mask.convert("L")
        if condition_image is not None and not condition_image.mode == "RGB":
            condition_image = condition_image.convert("RGB")
        if openpose_image is not None and not openpose_image.mode == "RGB":
            openpose_image = openpose_image.convert("RGB")
        if canny_image is not None and not canny_image.mode == "RGB":
            canny_image = canny_image.convert("RGB")
        
        # 转换为tensor并规范化
        example["real_images"] = transforms.ToTensor()(real_image)
        
        example["real_masks"] = transforms.ToTensor()(real_mask)
        
        if cloth_warp_image is not None:
            example["cloth_warp_images"] = transforms.ToTensor()(cloth_warp_image)
        
        if cloth_warp_mask is not None:
            example["cloth_warp_masks"] = transforms.ToTensor()(cloth_warp_mask)
        
        if condition_image is not None:
            example["condition_images"] = transforms.ToTensor()(condition_image)

        if openpose_image is not None:
            example["openpose_images"] = transforms.ToTensor()(openpose_image)

        if canny_image is not None:
            example["canny_images"] = transforms.ToTensor()(canny_image)

        return example


class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_parameter_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_model(args, global_step, unet, accelerator, is_final=False):
    """只保存可训练参数的检查点"""
    if not accelerator.is_main_process:
        return

    print(f"\n保存模型检查点 step-{global_step}...")
    save_path = os.path.join(args.output_dir, f"step-{global_step}")
    os.makedirs(save_path, exist_ok=True)

    # 1. 只保存requires_grad=True的参数
    unwrapped_unet = accelerator.unwrap_model(unet)
    trainable_params = {
        name: param.data.cpu()
        for name, param in unwrapped_unet.named_parameters()
        if param.requires_grad
    }
    
    # 使用safetensors保存
    from safetensors.torch import save_file
    save_file(
        trainable_params,
        os.path.join(save_path, "trainable_params.safetensors")
    )

    # 2. 保存训练配置
    training_config = {
        "step": global_step,
        "trainable_modules": args.trainable_modules,  # 记录哪些模块被训练
        "total_trainable_params": sum(p.numel() for p in trainable_params.values()),
        "dream_lambda": args.dream_lambda,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "resume_from_checkpoint_path": args.resume_from_checkpoint,
        "use_warp_cloth": args.use_warp_cloth,
        "condition_image_drop_out": args.condition_image_drop_out,
        "cloth_warp_drop_out": args.cloth_warp_drop_out,
        "canny_conditioning_type": args.canny_conditioning_type,
        "latent_append_num": args.latent_append_num,
    }
    import json
    with open(os.path.join(save_path, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)

    print(f"可训练参数已保存到: {save_path}")

    # 如果设置了保存数量限制，删除旧的检查点
    if args.save_total_limit is not None:
        checkpoints = [d for d in os.listdir(args.output_dir) 
                      if d.startswith("step-") and os.path.isdir(os.path.join(args.output_dir, d))]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        
        # 保留最新的 save_total_limit 个检查点
        if len(checkpoints) > args.save_total_limit:
            checkpoints_to_remove = checkpoints[:-args.save_total_limit]
            for checkpoint in checkpoints_to_remove:
                checkpoint_path = os.path.join(args.output_dir, checkpoint)
                print(f"删除旧检查点: {checkpoint_path}")
                import shutil
                shutil.rmtree(checkpoint_path)


def load_trainable_params(unet, checkpoint_path):
    """加载可训练参数"""
    from safetensors.torch import load_file
    
    # 加载参数
    # 检查只要文件夹下面有safetensors文件，就加载
    if os.path.isdir(checkpoint_path):
        for file in os.listdir(checkpoint_path):
            if file.endswith(".safetensors"):
                trainable_params = load_file(os.path.join(checkpoint_path, file))
                break
    else:
        raise ValueError(f"未指定safetensors文件路径: {checkpoint_path}")
    
    # 加载配置
    training_config = {}
    if os.path.exists(os.path.join(checkpoint_path, "training_config.json")):
        with open(os.path.join(checkpoint_path, "training_config.json"), "r") as f:
            training_config = json.load(f)
    
    # 将参数加载到模型中
    current_state_dict = unet.state_dict()
    for name, param in trainable_params.items():
        if name in current_state_dict:
            current_state_dict[name].copy_(param)
    
    print(f"已加载检查点: {checkpoint_path}")
    return training_config


def freeze_model_layers(unet, trainable_modules=None):
    """冻结UNet中除了指定模块之外的所有层"""
    # 首先冻结所有参数
    for param in unet.parameters():
        param.requires_grad = False
    
    if not trainable_modules:
        return
    
    # 记录解冻的参数数量
    unfrozen_params = 0
    
    # 根据指定模块解冻参数
    for module_name in trainable_modules:
        if module_name == "attention":
            # 只训练attention层
            for name, module in unet.named_modules():
                if "attn" in name:
                    for param in module.parameters():
                        param.requires_grad = True
                        unfrozen_params += param.numel()
        elif module_name == "transformer":
            # 训练transformer blocks
            for block in [unet.down_blocks, unet.mid_block, unet.up_blocks]:
                if hasattr(block, "attentions"):
                    for attn in block.attentions:
                        for param in attn.parameters():
                            param.requires_grad = True
                            unfrozen_params += param.numel()
                elif isinstance(block, list):
                    for b in block:
                        if hasattr(b, "attentions"):
                            for attn in b.attentions:
                                for param in attn.parameters():
                                    param.requires_grad = True
                                    unfrozen_params += param.numel()
    
    # 确保至少有一些参数是可训练的
    if unfrozen_params == 0:
        raise ValueError(f"没有可训练的参数！请检查 trainable_modules 设置: {trainable_modules}")
    
    print(f"已解冻参数数量: {unfrozen_params:,}")


def main():
    args = parse_args()

    # 给output_dir添加时间戳
    import time
    args.output_dir = os.path.join(args.output_dir, time.strftime("%m%d_%H:%M"))
    
    logging_dir = Path(args.output_dir, args.logging_dir)

    project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                args.pretrained_model_name_or_path, torch_dtype=torch_dtype, safety_checker=None
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size, num_workers=1
            )

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)
            transform_to_pil = transforms.ToPILImage()
            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                bsz = len(example["prompt"])
                fake_images = torch.rand((3, args.resolution, args.resolution))
                transform_to_pil = transforms.ToPILImage()
                fake_pil_images = transform_to_pil(fake_images)

                fake_mask = random_mask((args.resolution, args.resolution), ratio=1, mask_full_image=True)

                images = pipeline(prompt=example["prompt"], mask_image=fake_mask, image=fake_pil_images).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")


    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    # ! 使用stabilityai/sd-vae-ft-mse作为vae
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    attn_modules = adapt_unet_with_catvton_attn(
        unet = unet,
        catvton_attn_path=args.catvton_attn_path,
        trainable_modules=args.trainable_modules
    )

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    # 将除了attn_modules中的模块之外的参数设置为不更新
    attn_params = set()
    for module in attn_modules:
        attn_params.update(id(p) for p in module.parameters())
    
    for param in unet.parameters():
        if id(param) not in attn_params:
            param.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        random_transform=args.random_transform,
        use_warp_cloth=args.use_warp_cloth,
        use_openpose_conditioning=args.use_openpose_conditioning,
        use_canny_conditioning=args.use_canny_conditioning,
        canny_conditioning_type=args.canny_conditioning_type,
    )



    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    accelerator.register_for_checkpointing(lr_scheduler)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # 打印参数统计信息
    print("开始统计模型参数...")
    try:
        unet_total, unet_trainable = get_parameter_stats(unet)
        text_encoder_total, text_encoder_trainable = get_parameter_stats(text_encoder)
        vae_total, vae_trainable = get_parameter_stats(vae)
        
        total_params = unet_total + text_encoder_total + vae_total
        total_trainable_params = unet_trainable + (text_encoder_trainable if args.train_text_encoder else 0)
        
        print("模型参数统计:")
        print(f"  UNet 参数: {unet_total:,} (可训练: {unet_trainable:,})")
        print(f"  Text Encoder 参数: {text_encoder_total:,} (可训练: {text_encoder_trainable if args.train_text_encoder else 0:,})")
        print(f"  VAE 参数: {vae_total:,} (可训练: {vae_trainable:,})")
        print(f"  总参数: {total_params:,}")
        print(f"  总可训练参数: {total_trainable_params:,}")
        print(f"  可训练参数占比: {(total_trainable_params/total_params)*100:.2f}%")
    except Exception as e:
        print(f"统计参数时出错: {str(e)}")

    print("***** Running training *****")
    print(f"  数据集大小 = {len(train_dataset)}")
    print(f"  每个epoch的batch数 = {len(train_dataloader)}")
    print(f"  训练epoch数 = {args.num_train_epochs}")
    print(f"  每个设备的batch大小 = {args.train_batch_size}")
    print(f"  总训练batch大小 (包含并行、分布式和梯度累积) = {total_batch_size}")
    print(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    print(f"  总优化步数 = {args.max_train_steps}")
    
    # 检查模型状态
    print("检查模型状态:")
    print(f"  UNet device: {next(unet.parameters()).device}")
    print(f"  Text Encoder device: {next(text_encoder.parameters()).device}")
    print(f"  VAE device: {next(vae.parameters()).device}")
    print(f"  是否使用混合精度: {args.mixed_precision}")
    print(f"  权重数据类型: {weight_dtype}")
    
    # 检查优化器状态
    print("检查优化器状态:")
    print(f"  优化器类型: {optimizer.__class__.__name__}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  优化器参数组数: {len(optimizer.param_groups)}")
    for i, group in enumerate(optimizer.param_groups):
        print(f"  参数组 {i} 学习率: {group['lr']}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        print("正在加载检查点...")
        if args.resume_from_checkpoint == "latest":
            # 获取最新的检查点
            checkpoints = [d for d in os.listdir(args.output_dir) 
                         if d.startswith("step-") and os.path.isdir(os.path.join(args.output_dir, d))]
            if not checkpoints:
                raise ValueError(f"在 {args.output_dir} 中没有找到检查点")
            
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            args.resume_from_checkpoint = os.path.join(args.output_dir, checkpoints[-1])

        
        # 加载检查点
        training_config = load_trainable_params(unet, args.resume_from_checkpoint)
        
        # 更新起始步数
        global_step = training_config["step"]
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch
        
        print(f"从步数 {global_step} 继续训练")
    else:
        global_step = 0
        first_epoch = 0
        resume_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # 在开始训练循环之前添加验证pipeline的初始化
    if args.validation_steps is not None and accelerator.is_main_process:
        if None in [args.validation_prompt, args.validation_image, args.validation_mask]:
            print("验证参数不完整，将跳过验证步骤。需要同时提供 validation_prompt, validation_image 和 validation_mask。")
            args.validation_steps = None
        else:
            validation_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=weight_dtype,
                safety_checker=None,
            )
            validation_pipeline.scheduler = DDPMScheduler.from_config(validation_pipeline.scheduler.config)
            validation_pipeline.to(accelerator.device)
            
            # 加载验证用的图片和mask
            from PIL import Image
            init_image = Image.open(args.validation_image).convert("RGB")
            mask_image = Image.open(args.validation_mask).convert("L")
            init_image = init_image.resize((args.resolution, args.resolution))
            mask_image = mask_image.resize((args.resolution, args.resolution))

    # 在开始训练前进行测试生成
    if accelerator.is_main_process:
        print("正在进行预训练测试...")

        # 加载验证图片
        validation_image = Image.open(args.validation_image).convert("RGB")
        validation_mask = Image.open(args.validation_mask).convert("L")
        validation_condition_image = Image.open(args.validation_condition_image).convert("RGB")
        validation_canny_image = Image.open(args.validation_canny_image).convert("RGB")
        # 加载cloth_warp相关的图片
        validation_cloth_warp_image = None
        validation_cloth_warp_mask = None
        if args.valid_cloth_warp_image is not None and args.valid_cloth_warp_mask is not None:
            validation_cloth_warp_image = Image.open(args.valid_cloth_warp_image).convert("RGB")
            validation_cloth_warp_mask = Image.open(args.valid_cloth_warp_mask).convert("L")

        # 运行推理
        result = run_inference(
            unet=accelerator.unwrap_model(unet),
            vae=vae,
            noise_scheduler=noise_scheduler,
            device=accelerator.device,
            weight_dtype=weight_dtype,
            image=validation_image,
            mask=validation_mask,
            condition_image=validation_condition_image,
            cloth_warp_image=validation_cloth_warp_image,
            cloth_warp_mask=validation_cloth_warp_mask,
            canny_image=validation_canny_image,
        )[0]
        
        # 保存推理结果
        result.save(os.path.join(args.output_dir, "pretrain_test.png"))
        print(f"预训练测试图片已保存到: {os.path.join(args.output_dir, 'pretrain_test.png')}")

        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 等待所有进程
    accelerator.wait_for_everyone()

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # 基础输入处理
                real_images_batch = batch["real_images"]
                real_masks_batch = batch["real_masks"]
                
                # 将real_images转换到潜空间
                real_image_latents = vae.encode(real_images_batch.to(dtype=weight_dtype)).latent_dist.sample()
                real_image_latents = real_image_latents * vae.config.scaling_factor

                # 处理condition_images，添加dropout
                condition_images = batch["condition_images"].to(dtype=weight_dtype)
                
                # 生成batch大小的随机数，决定哪些样本的condition image要变成全黑
                condition_dropout_mask = torch.rand(condition_images.shape[0], 1, 1, 1, device=condition_images.device) < args.condition_image_drop_out
                condition_dropout_mask = condition_dropout_mask.expand(-1, *condition_images.shape[1:])
                
                # 将选中的样本的condition image变成全黑 (-1是归一化后的黑色值)
                condition_images = torch.where(condition_dropout_mask, torch.full_like(condition_images, -1.0), condition_images)
                
                # 编码condition images
                condition_image_latents = vae.encode(condition_images).latent_dist.sample()
                condition_image_latents = condition_image_latents * vae.config.scaling_factor

                # 准备masked_image
                masked_real_images = real_images_batch * (real_masks_batch < 0.5)
                
                # 处理cloth_warp
                if args.use_warp_cloth:
                    cloth_warp_images_batch = batch["cloth_warp_images"]
                    cloth_warp_masks_batch = batch["cloth_warp_masks"]
                    
                    # 生成batch大小的随机数，决定哪些样本不使用cloth warp
                    cloth_warp_dropout_mask = torch.rand(cloth_warp_images_batch.shape[0], 1, 1, 1, device=cloth_warp_images_batch.device) < args.cloth_warp_drop_out
                    cloth_warp_dropout_mask = cloth_warp_dropout_mask.expand(-1, *cloth_warp_images_batch.shape[1:])
                    
                    # 对于要dropout的样本，将cloth_warp_masks设为0，这样就不会应用cloth warp
                    cloth_warp_masks_batch = torch.where(cloth_warp_dropout_mask, torch.zeros_like(cloth_warp_masks_batch), cloth_warp_masks_batch)
                    
                    # 应用cloth warp
                    cloth_warp_images_batch = cloth_warp_images_batch * (cloth_warp_masks_batch >= 0.5)
                    masked_part = masked_real_images + cloth_warp_images_batch
                else:
                    masked_part = masked_real_images

                # 编码masked_part
                masked_part_latents = vae.encode(masked_part.to(dtype=weight_dtype)).latent_dist.sample()
                masked_part_latents = masked_part_latents * vae.config.scaling_factor

                # 根据latent_append_num决定拼接方式
                latents_to_concat = [real_image_latents, condition_image_latents]
                masks_to_concat = [mask_latent := torch.nn.functional.interpolate(real_masks_batch, size=real_image_latents.shape[-2:], mode="nearest"),
                                  torch.zeros_like(mask_latent)]
                masked_latents_to_concat = [masked_part_latents, condition_image_latents]

                # 如果需要openpose (latent_append_num >= 2)
                if args.use_openpose_conditioning:
                    openpose_images = batch["openpose_images"].to(dtype=weight_dtype)
                    openpose_latents = vae.encode(openpose_images).latent_dist.sample()
                    openpose_latents = openpose_latents * vae.config.scaling_factor
                    
                    latents_to_concat.append(openpose_latents)
                    masks_to_concat.append(torch.zeros_like(mask_latent))
                    masked_latents_to_concat.append(openpose_latents)

                # 如果需要canny (latent_append_num >= 3)
                if args.use_canny_conditioning:
                    canny_images = batch["canny_images"].to(dtype=weight_dtype)
                    canny_latents = vae.encode(canny_images).latent_dist.sample()
                    canny_latents = canny_latents * vae.config.scaling_factor
                    
                    latents_to_concat.append(canny_latents)
                    masks_to_concat.append(torch.zeros_like(mask_latent))
                    masked_latents_to_concat.append(canny_latents)

                # 拼接所有latents
                latent_model_input_p1 = torch.cat(latents_to_concat, dim=-2)
                mask_latent_concat = torch.cat(masks_to_concat, dim=-2)
                masked_latent_concat = torch.cat(masked_latents_to_concat, dim=-2)

                # 添加噪声
                noise = torch.randn_like(latent_model_input_p1)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latent_model_input_p1.shape[0],), device=latent_model_input_p1.device)
                noisy_latents = noise_scheduler.add_noise(latent_model_input_p1, noise, timesteps)

                # 连接所有输入
                latent_model_input = torch.cat([
                    noisy_latents,
                    mask_latent_concat,
                    masked_latent_concat,
                ], dim=1)

                # 预测噪声残差
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states=None).sample

                # 计算DREAM损失
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                    # 确保mask_latent的维度与noise_pred匹配
                    dream_weights = 1.0 + (args.dream_lambda - 1.0) * mask_latent
                    # 根据实际的latent_append_num扩展dream_weights
                    num_appends = 1  # 基础CatVTON
                    if args.use_openpose_conditioning:
                        num_appends += 1
                    if args.use_canny_conditioning:
                        num_appends += 1
                        
                    #! 注意这里先尝试一下全0，看看效果
                    # 扩展dream_weights
                    dream_weights = torch.cat([dream_weights] + [torch.zeros_like(dream_weights)] * num_appends, dim=-2)
                    
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = (loss * dream_weights).mean()
                else:
                    # v_prediction的情况类似
                    target = noise_scheduler.get_velocity(latent_model_input_p1, noise, timesteps)
                    dream_weights = 1.0 + (args.dream_lambda - 1.0) * mask_latent
                    num_appends = 1 + args.use_openpose_conditioning + args.use_canny_conditioning
                    dream_weights = torch.cat([dream_weights] + [torch.zeros_like(dream_weights)] * num_appends, dim=-2)
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = (loss * dream_weights).mean()

                # 在反向传播前检查损失是否有梯度
                if not loss.requires_grad:
                    raise ValueError("损失没有梯度！请检查模型参数是否正确解冻。")

                accelerator.backward(loss)
                
                # 添加梯度检查
                if accelerator.sync_gradients:
                    trainable_params = [p for p in unet.parameters() if p.requires_grad]
                    if not trainable_params:
                        raise ValueError("没有可训练的参数！")
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # 添加保存检查点的逻辑
                if args.save_steps is not None and global_step % args.save_steps == 0:
                    save_model(args, global_step, unet, accelerator)

                # 添加验证步骤
                if args.validation_steps is not None and global_step % args.validation_steps == 0 and accelerator.is_main_process:
                    print("正在生成验证图片...")
                    output_path = os.path.join(args.output_dir, f"validation_step_{global_step}.png")
                    
                    # 加载验证图片
                    validation_image = Image.open(args.validation_image).convert("RGB")
                    validation_mask = Image.open(args.validation_mask).convert("L")
                    validation_condition_image = Image.open(args.validation_condition_image).convert("RGB")
                    
                    # 加载cloth_warp相关的图片
                    validation_cloth_warp_image = None
                    validation_cloth_warp_mask = None
                    if args.valid_cloth_warp_image is not None and args.valid_cloth_warp_mask is not None:
                        validation_cloth_warp_image = Image.open(args.valid_cloth_warp_image).convert("RGB")
                        validation_cloth_warp_mask = Image.open(args.valid_cloth_warp_mask).convert("L")

                    # 生成验证图片
                    with torch.autocast(accelerator.device.type):
                        result = run_inference(
                            unet=accelerator.unwrap_model(unet),
                            vae=vae,
                            noise_scheduler=noise_scheduler,
                            device=accelerator.device,
                            weight_dtype=weight_dtype,
                            image=validation_image,
                            mask=validation_mask,
                            condition_image=validation_condition_image,
                            cloth_warp_image=validation_cloth_warp_image,
                            cloth_warp_mask=validation_cloth_warp_mask,
                        )[0]

                    # 保存验证图片
                    result.save(output_path)
                    print(f"验证图片已保存到: {output_path}")
                    
                    # 确保模型回到训练模式
                    unet.train()
                    if args.train_text_encoder:
                        text_encoder.train()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # 在训练结束时保存最终模型
    if accelerator.is_main_process:
        save_model(args, global_step, unet, accelerator, is_final=True)


if __name__ == "__main__":
    main()

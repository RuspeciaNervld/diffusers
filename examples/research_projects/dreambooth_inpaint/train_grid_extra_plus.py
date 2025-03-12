import argparse
import itertools
import math
import os

import torchvision.transforms.functional
# 设置huggingface镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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
from train_grid_extra_plus_infer import run_inference_2
import json
#! 如果不想用image_encoder，则用before
from unet_adapter import adapt_unet_with_catvton_attn
import lpips
from torchvision.transforms.functional import to_tensor
import torchvision
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.color import rgb2lab
from image_encoder import CLIPPeftModel

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from train_grid_extra_plus_infer import prepare_image, prepare_mask_image, inverse_transform, adaptive_crop_with_margin


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)

def canny_edge_detector(images):
    # 转换为灰度图（假设输入为RGB）
    gray_images = 0.299 * images[:,0] + 0.587 * images[:,1] + 0.114 * images[:,2]
    gray_images = gray_images.unsqueeze(1)  # [B,1,H,W]

    # 高斯滤波（保持与原始代码一致）
    kernel = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.bfloat16, device=images.device)
    kernel = kernel.view(1,1,3,3) / 16.0
    blurred = F.conv2d(gray_images, kernel, padding=1)  # 单通道无需groups参数
    
    # Sobel梯度计算（单通道输入）
    sobel_x = F.conv2d(blurred, torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.bfloat16, device=images.device).view(1,1,3,3))
    sobel_y = F.conv2d(blurred, torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.bfloat16, device=images.device).view(1,1,3,3))
    grad_mag = torch.sqrt(sobel_x**2 + sobel_y**2)
    
    # 后续步骤不变
    edge_map = (grad_mag > 0.3).float()
    return edge_map

def collate_fn(examples):
    # 收集所有图像数据
    real_images = torch.stack([example["real_images"] for example in examples])
    real_masks = torch.stack([example["real_masks"] for example in examples])
    condition_images = torch.stack([example["condition_images"] for example in examples])
    # 判断是否存在对应的Key
    if "cloth_warp_images" in examples[0]:
        cloth_warp_images = torch.stack([example["cloth_warp_images"] for example in examples])
        cloth_warp_masks = torch.stack([example["cloth_warp_masks"] for example in examples])
    else:
        cloth_warp_images = None
        cloth_warp_masks = None
    
    if "extra_cond1_images" in examples[0]:
        extra_cond1_images = torch.stack([example["extra_cond1_images"] for example in examples])
    else:
        extra_cond1_images = None
    if "extra_cond2_images" in examples[0]:
        extra_cond2_images = torch.stack([example["extra_cond2_images"] for example in examples])
    else:
        extra_cond2_images = None
    if "extra_cond3_images" in examples[0]:
        extra_cond3_images = torch.stack([example["extra_cond3_images"] for example in examples])
    else:
        extra_cond3_images = None
    # 确保数据格式正确
    real_images = real_images.to(memory_format=torch.contiguous_format).float()
    real_masks = real_masks.to(memory_format=torch.contiguous_format).float()
    condition_images = condition_images.to(memory_format=torch.contiguous_format).float()
    cloth_warp_images = cloth_warp_images.to(memory_format=torch.contiguous_format).float()
    cloth_warp_masks = cloth_warp_masks.to(memory_format=torch.contiguous_format).float()
    if extra_cond1_images is not None:
        extra_cond1_images = extra_cond1_images.to(memory_format=torch.contiguous_format).float()
    if extra_cond2_images is not None:
        extra_cond2_images = extra_cond2_images.to(memory_format=torch.contiguous_format).float()
    if extra_cond3_images is not None:
        extra_cond3_images = extra_cond3_images.to(memory_format=torch.contiguous_format).float()

    batch = {
        "real_images": real_images,
        "real_masks": real_masks,
        "condition_images": condition_images,
        "cloth_warp_images": cloth_warp_images,
        "cloth_warp_masks": cloth_warp_masks,
        "extra_cond1_images": extra_cond1_images,
        "extra_cond2_images": extra_cond2_images,
        "extra_cond3_images": extra_cond3_images,
    }


    return batch


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
        "--validation_root_dir",
        type=str,
        default=None,
        help="验证集根目录，包含real_images, condition_images, densepose_images, canny_images, real_masks",
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
    # 添加颜色损失参数

    parser.add_argument(
        "--other_loss_type",
        type=str,
        default="lpips",
        choices=["lpips", "mse", "ssim", "canny","vgg"],
        help="颜色损失类型：lpips或mse"
    )
    parser.add_argument(
        "--other_loss_weight",
        type=float,
        default=0.0,
        help="额外损失的权重系数"
    )
    parser.add_argument(
        "--extra_cond1",
        type=str,
        default=None,
        help="额外条件1"
    )
    parser.add_argument(
        "--extra_cond2",
        type=str,
        default=None,
        help="额外条件2"
    )
    parser.add_argument(
        "--extra_cond3",
        type=str,
        default=None,
        help="额外条件3"
    )
    parser.add_argument(
        "--use_warp_as_condition",
        action="store_true",
        help="是否使用warp作为条件"
    )
    parser.add_argument(
        "--use_origin_condition",
        action="store_true",
        help="是否使用原始condition"
    )
    parser.add_argument(
        "--extra_cond1_drop_out",
        type=float,
        default=0.0,
        help="将extra_cond1 image变为全黑的概率"
    )
    parser.add_argument(
        "--pretrained_clip_model_path",
        type=str,
        default=None,
        help="预训练的CLIP模型路径"
    )
    parser.add_argument(
        "--train_image_encoder",
        action="store_true",
        help="是否训练image encoder"
    )
    parser.add_argument(
        "--image_encoder_lora_r",
        type=int,
        default=8,
        help="image encoder的lora r"
    )
    parser.add_argument(
        "--image_encoder_lora_alpha",
        type=int,
        default=32,
        help="image encoder的lora alpha"
    )
    parser.add_argument(
        "--image_encoder_lora_dropout",
        type=float,
        default=0.0,
        help="image encoder的lora dropout"
    )
    parser.add_argument(
        "--pretrained_image_encoder_path",
        type=str,
        default=None,
        help="预训练的image encoder路径"
    )
    parser.add_argument(
        "--reverse_right",
        action="store_true",
        help="是否把右边反向",
    )
    parser.add_argument(
        "--predict_together",
        action="store_true",
        help="是否一同预测",
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
        extra_cond1=None,
        extra_cond2=None,
        extra_cond3=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.random_transform = random_transform
        self.tokenizer = tokenizer

        # 获取所有图片路径
        self.real_images_path = sorted(Path(os.path.join(instance_data_root, "real_images")).iterdir())
        self.real_masks_path = sorted(Path(os.path.join(instance_data_root, "real_masks")).iterdir())
        self.condition_images_path = sorted(Path(os.path.join(instance_data_root, "condition_images")).iterdir())
        self.cloth_warp_images_path = sorted(Path(os.path.join(instance_data_root, "cloth_warp_images")).iterdir())
        self.cloth_warp_masks_path = sorted(Path(os.path.join(instance_data_root, "cloth_warp_masks")).iterdir())

        if extra_cond1 is not None:
            self.extra_cond1_images_path = sorted(Path(os.path.join(instance_data_root, extra_cond1)).iterdir())
        else:
            self.extra_cond1_images_path = None

        if extra_cond2 is not None:
            self.extra_cond2_images_path = sorted(Path(os.path.join(instance_data_root, extra_cond2)).iterdir())
        else:
            self.extra_cond2_images_path = None
        
        if extra_cond3 is not None:
            self.extra_cond3_images_path = sorted(Path(os.path.join(instance_data_root, extra_cond3)).iterdir())
        else:
            self.extra_cond3_images_path = None
            
        # 输出每个文件夹中的图片数量
        print(f"real_images_path: {len(self.real_images_path)}")
        print(f"real_masks_path: {len(self.real_masks_path)}")
        print(f"condition_images_path: {len(self.condition_images_path)}")
        print(f"cloth_warp_images_path: {len(self.cloth_warp_images_path)}")
        print(f"cloth_warp_masks_path: {len(self.cloth_warp_masks_path)}")
        if self.extra_cond1_images_path is not None:
            print(f"extra_cond1_images_path: {len(self.extra_cond1_images_path)}")
        if self.extra_cond2_images_path is not None:
            print(f"extra_cond2_images_path: {len(self.extra_cond2_images_path)}")
        if self.extra_cond3_images_path is not None:
            print(f"extra_cond3_images_path: {len(self.extra_cond3_images_path)}")

        self.num_images = len(self.real_images_path)


    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        example = {}
        
        # 加载所有图像
        real_image = Image.open(self.real_images_path[index])
        real_mask = Image.open(self.real_masks_path[index])
        condition_image = Image.open(self.condition_images_path[index])
        cloth_warp_image = Image.open(self.cloth_warp_images_path[index])
        cloth_warp_mask = Image.open(self.cloth_warp_masks_path[index])

        if self.extra_cond1_images_path is not None:
            extra_cond1_image = Image.open(self.extra_cond1_images_path[index])
        else:
            extra_cond1_image = None
        if self.extra_cond2_images_path is not None:
            extra_cond2_image = Image.open(self.extra_cond2_images_path[index])
        else:
            extra_cond2_image = None
        if self.extra_cond3_images_path is not None:
            extra_cond3_image = Image.open(self.extra_cond3_images_path[index])
        else:
            extra_cond3_image = None

        # 转换图像模式
        if not real_mask.mode == "L":
            real_mask = real_mask.convert("L")
        if cloth_warp_image is not None and not cloth_warp_image.mode == "RGB":
            cloth_warp_image = cloth_warp_image.convert("RGB")
        if cloth_warp_mask is not None and not cloth_warp_mask.mode == "L":
            cloth_warp_mask = cloth_warp_mask.convert("L")
        if condition_image is not None and not condition_image.mode == "RGB":
            condition_image = condition_image.convert("RGB")
        if extra_cond1_image is not None and not extra_cond1_image.mode == "RGB":
            extra_cond1_image = extra_cond1_image.convert("RGB")
        if extra_cond2_image is not None and not extra_cond2_image.mode == "RGB":
            extra_cond2_image = extra_cond2_image.convert("RGB")
        if extra_cond3_image is not None and not extra_cond3_image.mode == "RGB":
            extra_cond3_image = extra_cond3_image.convert("RGB")
        
       # 生成随机调整参数
        brightness=0.2
        contrast=0.2
        saturation=0.2
        hue=0.1
        self.brightness = (-brightness,brightness)
        self.contrast = (-contrast,contrast)
        self.saturation = (-saturation,saturation)
        self.hue = (-hue,hue)

        brightness_factor = torch.rand(1).item() * (self.brightness[1] - self.brightness[0]) + self.brightness[0]
        contrast_factor = torch.rand(1).item() * (self.contrast[1] - self.contrast[0]) + self.contrast[0]
        saturation_factor = torch.rand(1).item() * (self.saturation[1] - self.saturation[0]) + self.saturation[0]
        hue_factor = torch.rand(1).item() * (self.hue[1] - self.hue[0]) + self.hue[0]

        # 对需要同步变换的图像应用相同参数
        def apply_color_jitter(img):
            img = torchvision.transforms.functional.adjust_brightness(img, brightness_factor)
            img = torchvision.transforms.functional.adjust_contrast(img, contrast_factor)
            img = torchvision.transforms.functional.adjust_saturation(img, saturation_factor)
            img = torchvision.transforms.functional.adjust_hue(img, hue_factor)
            return img


        # 转换为tensor并规范化
        example["real_images"] = transforms.ToTensor()(apply_color_jitter(real_image))
        
        example["real_masks"] = transforms.ToTensor()(real_mask)
        
        if cloth_warp_image is not None:
            example["cloth_warp_images"] = transforms.ToTensor()(apply_color_jitter(cloth_warp_image))
        
        if cloth_warp_mask is not None:
            example["cloth_warp_masks"] = transforms.ToTensor()(cloth_warp_mask)
        
        if condition_image is not None:
            example["condition_images"] = transforms.ToTensor()(apply_color_jitter(condition_image))

        if extra_cond1_image is not None:
            example["extra_cond1_images"] = transforms.ToTensor()(apply_color_jitter(extra_cond1_image))

        if extra_cond2_image is not None:
            example["extra_cond2_images"] = transforms.ToTensor()(apply_color_jitter(extra_cond2_image))

        if extra_cond3_image is not None:
            example["extra_cond3_images"] = transforms.ToTensor()(apply_color_jitter(extra_cond3_image))

        return example


def get_parameter_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_model(args, global_step, unet, accelerator, is_final=False, image_encoder =None):
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

    if image_encoder is not None:
        unwrapped_image_encoder = accelerator.unwrap_model(image_encoder)
        unwrapped_image_encoder.save_pretrained(os.path.join(save_path, "image_encoder.pt"))
    
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
        "other_loss_type": args.other_loss_type,
        "other_loss_weight": args.other_loss_weight,
        "use_warp_as_condition": args.use_warp_as_condition,
        "use_origin_condition": args.use_origin_condition,
        "extra_cond1": args.extra_cond1,
        "extra_cond2": args.extra_cond2,
        "extra_cond3": args.extra_cond3,
        "pretrained_clip_model_path": args.pretrained_clip_model_path,
        "pretrained_image_encoder_path": args.pretrained_image_encoder_path,
        "image_encoder_lora_r": args.image_encoder_lora_r,
        "image_encoder_lora_alpha": args.image_encoder_lora_alpha,
        "image_encoder_lora_dropout": args.image_encoder_lora_dropout,
        "train_image_encoder": args.train_image_encoder,
        "predict_together": args.predict_together,
        "reverse_right":args.reverse_right,
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
    args.output_dir = os.path.join(args.output_dir, time.strftime("%m%d_%H%M"))
    
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

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 在初始化accelerator之后添加
    if args.other_loss_type == "lpips":
        lpips_model = lpips.LPIPS(net='vgg').to(accelerator.device)
        lpips_model.requires_grad_(False)
        lpips_model.eval()
    elif args.other_loss_type == "vgg":
        # --- 初始化VGG ---
        vgg = torchvision.models.vgg16(pretrained=True).features[:16].to(device=accelerator.device, dtype=weight_dtype)
        vgg.eval()  # 固定VGG参数
        for param in vgg.parameters():
            param.requires_grad = False


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

    if args.pretrained_image_encoder_path:
        image_encoder = CLIPPeftModel.from_pretrained(base_model_path=args.pretrained_clip_model_path,
                                                      finetune_path=args.pretrained_image_encoder_path)
        print("加载预训练image encoder成功")
    else:
        image_encoder = CLIPPeftModel(
            clip_model_name='ViT-B/32',
            checkpoint_path=args.pretrained_clip_model_path,
            lora_r=args.image_encoder_lora_r,
            lora_alpha=args.image_encoder_lora_alpha,
            lora_dropout=args.image_encoder_lora_dropout,
            lora_trainable_modules=["all","in_proj","out_proj"]
        )

    image_encoder.to(accelerator.device)
    if args.train_image_encoder:
        image_encoder.train()
        image_encoder.train_mode()
    else:
        image_encoder.eval()
        image_encoder.eval_mode()

    print("1111111 image_encoder.show_trainable_params()")
    image_encoder.show_trainable_params()
    print("1111111 end of image_encoder.show_trainable_params()")

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
        instance_prompt="",
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        random_transform=args.random_transform,
        extra_cond1=args.extra_cond1,
        extra_cond2=args.extra_cond2,
        extra_cond3=args.extra_cond3,
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

    # 在开始训练前进行测试生成
    # 验证图片名称列表
    validation_image_names = os.listdir(os.path.join(args.validation_root_dir, "real_images"))
    validation_image_length = len(validation_image_names)
    if accelerator.is_main_process:
        print("正在进行预训练测试...")
        random_index = random.randint(0, validation_image_length - 1)
        # 加载验证图片
        validation_image = Image.open(os.path.join(args.validation_root_dir, "real_images", validation_image_names[random_index])).convert("RGB")
        validation_mask = Image.open(os.path.join(args.validation_root_dir, "real_masks", validation_image_names[random_index])).convert("L")
        validation_condition_image = Image.open(os.path.join(args.validation_root_dir, "condition_images", validation_image_names[random_index])).convert("RGB")
        
        validation_extra_cond1_image = None
        validation_extra_cond2_image = None
        validation_extra_cond3_image = None
        if "extra_cond1" in args and args.extra_cond1 is not None:
            validation_extra_cond1_image = Image.open(os.path.join(args.extra_cond1.replace("vton", "vton_test"), validation_image_names[random_index])).convert("RGB")
        if "extra_cond2" in args and args.extra_cond2 is not None:
            validation_extra_cond2_image = Image.open(os.path.join(args.extra_cond2.replace("vton", "vton_test"), validation_image_names[random_index])).convert("RGB")
        if "extra_cond3" in args and args.extra_cond3 is not None:
            validation_extra_cond3_image = Image.open(os.path.join(args.extra_cond3.replace("vton", "vton_test"), validation_image_names[random_index])).convert("RGB")

        validation_cloth_warp_image = None
        validation_cloth_warp_mask = None
        if args.use_warp_cloth:
            validation_cloth_warp_image = Image.open(os.path.join(args.validation_root_dir, "cloth_warp_images", validation_image_names[random_index])).convert("RGB")
            validation_cloth_warp_mask = Image.open(os.path.join(args.validation_root_dir, "cloth_warp_masks", validation_image_names[random_index])).convert("L")

        # 运行推理
        result = run_inference_2(
            unet=accelerator.unwrap_model(unet),
            vae=vae,
            image_encoder=accelerator.unwrap_model(image_encoder),
            noise_scheduler=noise_scheduler,
            device=accelerator.device,
            weight_dtype=weight_dtype,
            image=validation_image,
            mask=validation_mask,
            condition_image=validation_condition_image,
            cloth_warp_image=validation_cloth_warp_image,
            cloth_warp_mask=validation_cloth_warp_mask,
            use_warp_as_condition=args.use_warp_as_condition,
            use_origin_condition=args.use_origin_condition,
            extra_cond1=validation_extra_cond1_image,
            extra_cond2=validation_extra_cond2_image,
            extra_cond3=validation_extra_cond3_image,
            show_whole_image=True,
            predict_together = args.predict_together,
            reverse_right = args.reverse_right,
        )[0]
        
        # 保存推理结果
        result.save(os.path.join(args.output_dir, "pretrain_test.png"))
        print(f"预训练测试图片已保存到: \n{os.path.join(args.output_dir, 'pretrain_test.png')}\n")

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

                
            with accelerator.accumulate(unet, image_encoder):
                # 基础输入处理
                real_images = prepare_image(batch["real_images"]).to(device=accelerator.device, dtype=weight_dtype)
                real_images_copy = real_images
                real_masks = prepare_mask_image(batch["real_masks"]).to(device=accelerator.device, dtype=weight_dtype)
                real_masks_copy = real_masks
                real_images, params = adaptive_crop_with_margin(
                    real_images, real_masks_copy, 
                    margin_ratio=0.05, 
                    target_size=(512, 384)
                )

                real_masks, _ = adaptive_crop_with_margin(
                    real_masks, real_masks_copy, 
                    margin_ratio=0.05, 
                    target_size=(512, 384)
                )

                real_images = real_images.to(device=accelerator.device, dtype=weight_dtype)
                real_masks = real_masks.to(device=accelerator.device, dtype=weight_dtype)

                condition_images = prepare_image(batch["condition_images"]).to(device=accelerator.device, dtype=weight_dtype)
                cloth_warp_images = prepare_image(batch["cloth_warp_images"]).to(device=accelerator.device, dtype=weight_dtype)
                cloth_warp_masks = prepare_mask_image(batch["cloth_warp_masks"]).to(device=accelerator.device, dtype=weight_dtype)
                
                extra_cond1_images = prepare_image(batch["extra_cond1_images"]).to(device=accelerator.device, dtype=weight_dtype)
                # 决定哪些样本的condition image要变成全黑
                condition_dropout_mask = torch.rand_like(condition_images) < args.condition_image_drop_out
                condition_images = torch.where(condition_dropout_mask, torch.zeros_like(condition_images), condition_images)
                # 决定哪些样本的cloth_warp_images要变成全黑
                cloth_warp_dropout_mask = torch.rand_like(cloth_warp_images) < args.cloth_warp_drop_out
                cloth_warp_images = torch.where(cloth_warp_dropout_mask, torch.zeros_like(cloth_warp_images), cloth_warp_images)
                # 决定哪些样本的extra_cond1_images要变成全黑
                extra_cond1_dropout_mask = torch.rand_like(extra_cond1_images) < args.extra_cond1_drop_out
                extra_cond1_images = torch.where(extra_cond1_dropout_mask, torch.zeros_like(extra_cond1_images), extra_cond1_images)
                
                # 在像素空间进行拼接
                # (B, 3, H, W)
                real_images_2 = torch.cat([real_images, condition_images], dim=-2)

                masked_real_images_1 = real_images * (real_masks < 0.5) # 保留黑色部分
                masked_real_images_2 = torch.cat([masked_real_images_1, condition_images], dim=-2)
                masks_2 = torch.cat([real_masks, torch.zeros_like(real_masks)], dim=-2)
                if args.reverse_right:
                    masks_2_reverse = torch.cat([torch.zeros_like(real_masks), real_masks_copy], dim=-2)
                else:
                    masks_2_reverse = torch.cat([real_masks_copy, torch.zeros_like(real_masks)], dim=-2)

                #! 把底图去掉试试
                # warped_masked_real_images_1 = (torch.ones_like(real_images_copy) * ((real_masks_copy >= 0.5) ^ (cloth_warp_masks >= 0.5))) + (real_images_copy * (real_masks_copy < 0.5))  + (cloth_warp_images * (cloth_warp_masks >= 0.5)) 
                warped_masked_real_images_1 = (real_images_copy * (real_masks_copy < 0.5))  + (cloth_warp_images * (cloth_warp_masks >= 0.5)) 
                # warped_masked_real_images_1 =  cloth_warp_images
                #! 先尝试上面是extra_cond1_images，下面是warped_masked_real_images
                if args.reverse_right:
                    warped_masked_real_images_2 = torch.cat([extra_cond1_images,warped_masked_real_images_1], dim=-2)
                    warped_masked_real_images_2_target = torch.cat([extra_cond1_images,real_images_copy], dim=-2)
                else:
                    warped_masked_real_images_2 = torch.cat([warped_masked_real_images_1,extra_cond1_images], dim=-2)
                    warped_masked_real_images_2_target = torch.cat([real_images_copy,extra_cond1_images], dim=-2)


                if args.predict_together:
                    real_images_4 = torch.cat([real_images_2, warped_masked_real_images_2_target], dim=-1)
                else:
                    real_images_4 = torch.cat([real_images_2, warped_masked_real_images_2], dim=-1)
                masked_real_images_4 = torch.cat([masked_real_images_2, warped_masked_real_images_2], dim=-1)
                if args.predict_together:
                    masks_4 = torch.cat([masks_2, masks_2_reverse], dim=-1)
                else:
                    masks_4 = torch.cat([masks_2, torch.zeros_like(masks_2)], dim=-1)

                #! 可以在第一格dream，也可以同时dream，这里是第一格dream
                dream_mask = torch.cat([masks_2, torch.zeros_like(masks_2)], dim=-1)
                
                # VAE编码基础输入
                real_image_latents = vae.encode(real_images_4).latent_dist.sample()
                masked_real_images_latents = vae.encode(masked_real_images_4).latent_dist.sample()
                mask_latent = torch.nn.functional.interpolate(masks_4, size=real_image_latents.shape[-2:], mode="nearest")
                dream_mask_latent = torch.nn.functional.interpolate(dream_mask, size=real_image_latents.shape[-2:], mode="nearest")

                # 添加噪声
                noise = torch.randn_like(real_image_latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (real_image_latents.shape[0],), device=real_image_latents.device)
                noisy_latents = noise_scheduler.add_noise(real_image_latents, noise, timesteps)

                # 连接所有输入
                latent_model_input = torch.cat([
                    noisy_latents,
                    mask_latent,
                    masked_real_images_latents,
                ], dim=1)

                #! 如果需要测试image_encoder，则需要传入condition_images
                # image_encoder_hidden_states = image_encoder(condition_images)
                image_encoder_hidden_states = None

                # 预测噪声残差
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states=image_encoder_hidden_states).sample

                # 计算DREAM损失
                target = noise
                # 确保mask_latent的维度与noise_pred匹配
                dream_weights = 1.0 + (args.dream_lambda - 1.0) * dream_mask_latent
                
                # 改用Huber Loss（Smooth L1）
                # loss =  F.smooth_l1_loss(noise_pred.float(), target.float(), reduction="none", beta=1.5)
                # 混合MSE和MAE
                mse_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                # mae_loss = F.l1_loss(noise_pred.float(), target.float(), reduction="none")
                loss = mse_loss

                loss = (loss * dream_weights).mean()
                
                #! 新增：添加颜色损失

                other_loss = 0
                if args.other_loss_type == "lpips":
                    generated_images  = vae.decode(noisy_latents / vae.config.scaling_factor, return_dict=False)[0]
                    target_images  = vae.decode(target / vae.config.scaling_factor, return_dict=False)[0]
                    other_loss = lpips_model(generated_images, target_images).mean()
                    other_loss = (other_loss * dream_weights).mean()
                elif args.other_loss_type == "canny":
                    generated_images  = vae.decode(noisy_latents / vae.config.scaling_factor, return_dict=False)[0]
                    target_images  = vae.decode(target / vae.config.scaling_factor, return_dict=False)[0]
                    edge_generated = canny_edge_detector(generated_images)
                    edge_target = canny_edge_detector(target_images)
                    other_loss = F.mse_loss(edge_generated, edge_target).mean()
                    other_loss = (other_loss * dream_weights).mean()
                elif args.other_loss_type == "vgg":
                    # 假设VAE输出范围是[-1,1]
                    generated_images = vae.decode(noisy_latents / vae.config.scaling_factor, return_dict=False)[0]
                    target_images = vae.decode(target / vae.config.scaling_factor, return_dict=False)[0]

                    # --- 损失函数定义 ---
                    def normalize_vgg_input(x):
                        return (x + 1) / 2  # 转换到[0,1]

                    def histogram_loss(fake, real, bins=50):
                        fake = (fake + 1) / 2  # 转换到[0,1]
                        real = (real + 1) / 2
                        # 原有直方图计算逻辑...
                        return loss

                    # --- 计算损失 ---
                    with torch.no_grad():  # VGG不参与梯度计算
                        gen_vgg = vgg(normalize_vgg_input(generated_images))
                        target_vgg = vgg(normalize_vgg_input(target_images))

                    l1_loss = F.l1_loss(generated_images, target_images)
                    perceptual_loss = F.l1_loss(gen_vgg, target_vgg)
                    hist_loss = histogram_loss(generated_images, target_images)

                    #! 权重分配（需调参）
                    other_loss = 0.5 * l1_loss + 0.3 * perceptual_loss + 0.2 * hist_loss
                else:
                    other_loss = 0

                total_loss = loss  +  other_loss * args.other_loss_weight


                # 在反向传播前检查损失是否有梯度
                if not total_loss.requires_grad:
                    raise ValueError("损失没有梯度！请检查模型参数是否正确解冻。")

                accelerator.backward(total_loss)
                
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
                    save_model(args, global_step, unet, accelerator, image_encoder=image_encoder)

                # 添加验证步骤
                if args.validation_steps is not None and global_step % args.validation_steps == 0 and accelerator.is_main_process:
                    print("正在生成验证图片...")
                    random_index = np.random.randint(0, len(validation_image_names))
                    print(f"生成的图片的原路径是：\n{os.path.join(args.validation_root_dir, 'real_images', validation_image_names[random_index])}")
                    print( "catvton生成的路径是："+"\n/mnt/pub_data/results/vton_test/inference_results_catvton/" + validation_image_names[random_index])
                    output_path = os.path.join(args.output_dir, f"validation_step_{global_step}.png")
                    print("本次训练生成图片的路径是："+"\n"+output_path)
                    # 加载验证图片
                    validation_image = Image.open(os.path.join(args.validation_root_dir, "real_images", validation_image_names[random_index])).convert("RGB")
                    validation_mask = Image.open(os.path.join(args.validation_root_dir, "real_masks", validation_image_names[random_index])).convert("L")
                    validation_condition_image = Image.open(os.path.join(args.validation_root_dir, "condition_images", validation_image_names[random_index])).convert("RGB")
                    
                    # 加载cloth_warp相关的图片
                    validation_cloth_warp_image = None
                    validation_cloth_warp_mask = None
                    if args.use_warp_cloth:
                        validation_cloth_warp_image = Image.open(os.path.join(args.validation_root_dir, "cloth_warp_images", validation_image_names[random_index])).convert("RGB")
                        validation_cloth_warp_mask = Image.open(os.path.join(args.validation_root_dir, "cloth_warp_masks", validation_image_names[random_index])).convert("L")

                    validation_extra_cond1_image = None
                    validation_extra_cond2_image = None
                    validation_extra_cond3_image = None
                    if "extra_cond1" in args and args.extra_cond1 is not None:
                        validation_extra_cond1_image = Image.open(os.path.join(args.extra_cond1.replace("vton", "vton_test"), validation_image_names[random_index])).convert("RGB")
                    if "extra_cond2" in args and args.extra_cond2 is not None:
                        validation_extra_cond2_image = Image.open(os.path.join(args.extra_cond2.replace("vton", "vton_test"), validation_image_names[random_index])).convert("RGB")
                    if "extra_cond3" in args and args.extra_cond3 is not None:
                        validation_extra_cond3_image = Image.open(os.path.join(args.extra_cond3.replace("vton", "vton_test"), validation_image_names[random_index])).convert("RGB")

                    # 生成验证图片
                    with torch.autocast(accelerator.device.type):
                        result = run_inference_2(
                            unet=accelerator.unwrap_model(unet),
                            vae=vae,
                            image_encoder=accelerator.unwrap_model(image_encoder),
                            noise_scheduler=noise_scheduler,
                            device=accelerator.device,
                            weight_dtype=weight_dtype,
                            image=validation_image,
                            mask=validation_mask,
                            condition_image=validation_condition_image,
                            cloth_warp_image=validation_cloth_warp_image,
                            cloth_warp_mask=validation_cloth_warp_mask,
                            use_warp_as_condition=args.use_warp_as_condition,
                            use_origin_condition=args.use_origin_condition,
                            extra_cond1=validation_extra_cond1_image,
                            extra_cond2=validation_extra_cond2_image,
                            extra_cond3=validation_extra_cond3_image,
                            show_whole_image=True,
                            predict_together = args.predict_together,
                            reverse_right = args.reverse_right,
                        )[0]

                    # 保存验证图片
                    result.save(output_path)
                    
                    # 确保模型回到训练模式
                    unet.train()
                    if args.train_text_encoder:
                        text_encoder.train()

            logs = {"loss": total_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # 在训练结束时保存最终模型
    if accelerator.is_main_process:
        save_model(args, global_step, unet, accelerator, is_final=True, image_encoder=image_encoder)


if __name__ == "__main__":
    main()

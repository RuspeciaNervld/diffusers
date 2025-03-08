import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from PIL import Image
import os
import time
from tqdm import tqdm
import sys
import torchvision.transforms as transforms
import json
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catvton_base_infer import run_inference_2
from train_color_loss import DreamBoothDataset, collate_fn, adapt_unet_with_catvton_attn

# 全局配置
CONFIG = {
    # 模型和路径配置
    "test_data_root" :"/home/nervld/gitclone/diffusers/data/vton_test_small",
    "pretrained_model_path": "booksforcharlie/stable-diffusion-inpainting",
    "catvton_attn_path": "/home/nervld/gitclone/diffusers/models/catvton_unet_attn",
    "my_unet_attn_path": "/home/nervld/gitclone/diffusers/models/my_unet_attn/with warp/nolpips6000_dream50_batch2_lr1e-5", # 有的话会覆盖catvton_attn_path
    "output_dir": "/home/nervld/gitclone/diffusers/output/vton_test_small/nolpips6000_dream50_batch2_lr1e-5",
    
    # 条件控制配置
    "use_warp_cloth": True,
    "use_warp_as_condition": False,
    "extra_cond1": "/home/nervld/gitclone/diffusers/data/vton_test_small/detail_images",
    "extra_cond2": None,
    "extra_cond3": None,


    # 训练相关配置
    "resolution": 512,
    "batch_size": 4,
    "device": "cuda",
    "mixed_precision": "bf16",  # 修改为 "no" 以避免 bitsandbytes 问题
    "trainable_modules": "attention",
}

def load_trainable_params(unet, checkpoint_path):
    """加载可训练参数"""
    from safetensors.torch import load_file
    
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

def run_inference_on_test_dataset(test_data_root):
    
    # 获取real_images的文件名
    real_images_file_names = sorted(os.listdir(os.path.join(test_data_root, "real_images")))
    # print(f"real_images_file_names: {real_images_file_names}")
    """
    在测试数据集上运行推理
    
    Args:
        test_data_root: 测试数据集根目录
    """
    print("初始化模型...")
    
    # 设置权重类型
    weight_dtype = torch.float32
    if CONFIG["mixed_precision"] == "fp16":
        weight_dtype = torch.float16
    elif CONFIG["mixed_precision"] == "bf16":
        weight_dtype = torch.bfloat16

    # 加载模型
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    unet = UNet2DConditionModel.from_pretrained(CONFIG["pretrained_model_path"], subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(CONFIG["pretrained_model_path"], subfolder="scheduler")

    # 加载CatVTON attention权重
    if CONFIG['catvton_attn_path'] is not None:
        print(f"加载CatVTON attention权重: {CONFIG['catvton_attn_path']}")
        catvton_attn_path = CONFIG["catvton_attn_path"]
    else:
        catvton_attn_path = None
    adapt_unet_with_catvton_attn(
        unet=unet,
        catvton_attn_path=catvton_attn_path,
        trainable_modules=CONFIG["trainable_modules"]
    )

    # 加载MyUnet attention权重
    if CONFIG['my_unet_attn_path'] is not None:
        print(f"加载MyUnet attention权重: {CONFIG['my_unet_attn_path']}")
        load_trainable_params(unet, CONFIG["my_unet_attn_path"])
    else:
        print(f"未指定MyUnet attention权重路径")

    # 移动模型到设备
    vae.to(CONFIG["device"], dtype=weight_dtype)
    unet.to(CONFIG["device"], dtype=weight_dtype)


    # 创建测试数据集
    test_dataset = DreamBoothDataset(
        instance_data_root=test_data_root,
        instance_prompt=None,
        tokenizer=None,
        size=CONFIG["resolution"],
        center_crop=True,
        random_transform=False,
        extra_cond1=CONFIG["extra_cond1"],
        extra_cond2=CONFIG["extra_cond2"],
        extra_cond3=CONFIG["extra_cond3"],
    )

    # 创建数据加载器
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # 设置模型为评估模式
    unet.eval()
    vae.eval()
    
    print(f"开始处理测试数据集，共 {len(test_dataset)} 张图片...")
    print("\n运行配置:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="处理测试数据")):
            # 将数据移到正确的设备和类型
            real_images = batch["real_images"].to(CONFIG["device"], dtype=weight_dtype)
            real_masks = batch["real_masks"].to(CONFIG["device"], dtype=weight_dtype)
            condition_images = batch["condition_images"].to(CONFIG["device"], dtype=weight_dtype)
            if CONFIG["use_warp_cloth"]:
                cloth_warp_images = batch["cloth_warp_images"].to(CONFIG["device"], dtype=weight_dtype)
                cloth_warp_masks = batch["cloth_warp_masks"].to(CONFIG["device"], dtype=weight_dtype)
            else:
                cloth_warp_images = None
                cloth_warp_masks = None
            if CONFIG["extra_cond1"] is not None:
                extra_cond1 = batch["extra_cond1_images"].to(CONFIG["device"], dtype=weight_dtype)
            else:
                extra_cond1 = None
            if CONFIG["extra_cond2"] is not None:
                extra_cond2 = batch["extra_cond2_images"].to(CONFIG["device"], dtype=weight_dtype)
            else:
                extra_cond2 = None
            if CONFIG["extra_cond3"] is not None:
                extra_cond3 = batch["extra_cond3_images"].to(CONFIG["device"], dtype=weight_dtype)
            else:
                extra_cond3 = None

            # 运行推理
            result = run_inference_2(
                unet=unet,
                vae=vae,
                noise_scheduler=noise_scheduler,
                device=CONFIG["device"],
                weight_dtype=weight_dtype,
                image=real_images,
                mask=real_masks,
                condition_image=condition_images,
                cloth_warp_image=cloth_warp_images,
                cloth_warp_mask=cloth_warp_masks,
                extra_cond1=extra_cond1,
                extra_cond2=extra_cond2,
                extra_cond3=extra_cond3,
                use_warp_as_condition=CONFIG["use_warp_as_condition"],
            )

            for i in range(len(result)):
                # 保存结果，应该按照命名顺序按照原来real_images的文件名保存
                idxx = batch_idx * CONFIG["batch_size"] + i
                save_path = os.path.join(CONFIG["output_dir"], f"{real_images_file_names[idxx]}")
                result[i].save(save_path)

    # 计算并打印处理时间
    total_time = time.time() - start_time
    avg_time_per_image = total_time / len(test_dataset)
    print(f"\n处理完成！")
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"平均每张图片处理时间: {avg_time_per_image:.2f} 秒")
    print(f"结果已保存到: {CONFIG['output_dir']}")

    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # 只需要提供测试数据集路径
    test_data_root = CONFIG["test_data_root"]  # 替换为实际的测试数据集路径
    run_inference_on_test_dataset(test_data_root)

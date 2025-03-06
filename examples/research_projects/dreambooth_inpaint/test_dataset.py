import os
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from train_dreambooth_inpaint_catvton_base import DreamBoothDataset
from transformers import CLIPTokenizer

def visualize_batch(batch, save_path=None):
    """可视化一个批次的数据"""
    # 创建一个4x4的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # 转换tensor到numpy进行显示
    def tensor_to_pil(tensor):
        # 将[-1,1]范围转换回[0,1]
        tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1)
        if tensor.shape[0] == 4:  # RGBA
            # 分离RGB和A通道
            rgb = tensor[:3]
            alpha = tensor[3:]
            # 使用白色背景
            white_bg = torch.ones_like(rgb)
            # 根据alpha混合
            blended = rgb * alpha + white_bg * (1 - alpha)
            return blended
        return tensor

    # 显示real_images
    real_img = tensor_to_pil(batch["real_images"])
    axes[0, 0].imshow(real_img.permute(1, 2, 0))
    axes[0, 0].set_title("Real Image (RGBA)")
    axes[0, 0].axis("off")

    # 显示real_masks
    mask_img = batch["real_masks"]
    axes[0, 1].imshow(mask_img.squeeze(), cmap='gray')
    axes[0, 1].set_title("Mask (1 channel)")
    axes[0, 1].axis("off")

    # 显示cloth_warp
    cloth_img = tensor_to_pil(batch["cloth_warp"])
    axes[1, 0].imshow(cloth_img.permute(1, 2, 0))
    axes[1, 0].set_title("Cloth Warp (RGBA)")
    axes[1, 0].axis("off")

    # 显示合成效果（将mask应用到real_image上）
    masked_real = real_img * (1 - mask_img)
    axes[1, 1].imshow(masked_real.permute(1, 2, 0))
    axes[1, 1].set_title("Masked Real Image")
    axes[1, 1].axis("off")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # 设置参数
    instance_data_root = "/home/nervld/gitclone/diffusers/data/catvton"  # 修改为你的数据目录
    tokenizer = CLIPTokenizer.from_pretrained("booksforcharlie/stable-diffusion-inpainting", subfolder="tokenizer")
    
    # 创建两个数据集实例，一个使用随机变换，一个不使用
    dataset_with_random = DreamBoothDataset(
        instance_data_root=instance_data_root,
        instance_prompt="test prompt",
        tokenizer=tokenizer,
        size=512,
        center_crop=False,
        random_transform=True,
    )
    
    dataset_no_random = DreamBoothDataset(
        instance_data_root=instance_data_root,
        instance_prompt="test prompt",
        tokenizer=tokenizer,
        size=512,
        center_crop=False,
        random_transform=False,
    )
    
    # 创建输出目录
    output_dir = "dataset_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化带随机变换的样本
    print("测试带随机变换的数据集:")
    for i in range(min(3, len(dataset_with_random))):
        print(f"正在处理第 {i+1} 个样本 (带随机变换)")
        sample = dataset_with_random[i]
        
        print(f"真实图像尺寸: {sample['real_images'].shape}")
        print(f"掩码尺寸: {sample['real_masks'].shape}")
        print(f"变形服装尺寸: {sample['cloth_warp'].shape}")
        
        save_path = os.path.join(output_dir, f"sample_random_{i+1}.png")
        visualize_batch(sample, save_path)
        
        print(f"真实图像: {dataset_with_random.real_images_path[i].name}")
        print(f"掩码: {dataset_with_random.real_masks_path[i].name}")
        print(f"服装: {dataset_with_random.cloth_warp_path[i].name}")
        print("-" * 50)
    
    # 可视化不带随机变换的样本
    print("\n测试不带随机变换的数据集:")
    for i in range(min(3, len(dataset_no_random))):
        print(f"正在处理第 {i+1} 个样本 (不带随机变换)")
        sample = dataset_no_random[i]
        
        print(f"真实图像尺寸: {sample['real_images'].shape}")
        print(f"掩码尺寸: {sample['real_masks'].shape}")
        print(f"变形服装尺寸: {sample['cloth_warp'].shape}")
        
        save_path = os.path.join(output_dir, f"sample_no_random_{i+1}.png")
        visualize_batch(sample, save_path)
        
        print(f"真实图像: {dataset_no_random.real_images_path[i].name}")
        print(f"掩码: {dataset_no_random.real_masks_path[i].name}")
        print(f"服装: {dataset_no_random.cloth_warp_path[i].name}")
        print("-" * 50)

if __name__ == "__main__":
    main() 
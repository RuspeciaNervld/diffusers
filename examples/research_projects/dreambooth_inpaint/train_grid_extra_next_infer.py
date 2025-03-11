from diffusers.utils.torch_utils import randn_tensor
import inspect
# 导入jupyter的tqdm
import tqdm
from PIL import Image
import torch
from typing import Union

def prepare_extra_step_kwargs(noise_scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(
        inspect.signature(noise_scheduler.step).parameters.keys()
    )
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(
        inspect.signature(noise_scheduler.step).parameters.keys()
    )
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def compute_vae_encodings(image: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    """
    Args:
        images (torch.Tensor): image to be encoded
        vae (torch.nn.Module): vae model
    Returns:
        torch.Tensor: latent encoding of the image
    """
    pixel_values = image.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return model_input

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

import numpy as np
import PIL
# 准备图像（转换为 Batch 张量）
def prepare_image(image):
    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(dtype=torch.float32)

        if image.min() >= 0 and image.max() <= 1:
            # print("image.min() >= 0 and image.max() <= 1")
            image = image * 2 - 1.0

    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image

def prepare_mask_image(mask_image):
    if isinstance(mask_image, torch.Tensor):
        if mask_image.ndim == 2:
            # Batch and add channel dim for single mask
            mask_image = mask_image.unsqueeze(0).unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
            # Single mask, the 0'th dimension is considered to be
            # the existing batch size of 1
            mask_image = mask_image.unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
            # Batch of mask, the 0'th dimension is considered to be
            # the batching dimension
            mask_image = mask_image.unsqueeze(1)

        # Binarize mask
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
    else:
        # preprocess mask
        if isinstance(mask_image, (PIL.Image.Image, np.ndarray)):
            mask_image = [mask_image]

        if isinstance(mask_image, list) and isinstance(mask_image[0], PIL.Image.Image):
            mask_image = np.concatenate(
                [np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0
            )
            mask_image = mask_image.astype(np.float32) / 255.0
        elif isinstance(mask_image, list) and isinstance(mask_image[0], np.ndarray):
            mask_image = np.concatenate([m[None, None, :] for m in mask_image], axis=0)

        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image)

    return mask_image


@torch.no_grad()
def run_inference_2(
    unet,
    vae,
    image_encoder,
    noise_scheduler,
    device='cuda',
    weight_dtype=torch.float32,
    image: Union[torch.Tensor, Image.Image]=None,  # real_images
    mask: Union[torch.Tensor, Image.Image]=None,   # real_masks
    condition_image: Union[torch.Tensor, Image.Image]=None,  # condition_images
    cloth_warp_image: Union[torch.Tensor, Image.Image]=None,  # cloth_warp_images
    cloth_warp_mask: Union[torch.Tensor, Image.Image]=None,  # cloth_warp_masks
    # 除了上面的是必须存在的，下面的根据需要拼接
    extra_cond1: Union[torch.Tensor, Image.Image]=None,  # extra_condition_images
    extra_cond2: Union[torch.Tensor, Image.Image]=None,  # extra_condition_images
    num_inference_steps: int = 50,
    guidance_scale: float = 2.5,
    generator=None,
    eta=1.0,
    show_whole_image:bool = False,
    predict_together:bool = False,
    reverse_right:bool = True,
    **kwargs
):
    # 打印推理参数，重点是用哪些latent
    print("像素空间四宫格：原图+warp图；condition+extra_cond1")

    # 把模型移动到cuda
    unet.to(device)
    vae.to(device)

    # 准备基础输入
    print("预测阶段准备基础输入")
    real_images = prepare_image(image).to(device=device, dtype=weight_dtype)
    real_images_copy = real_images
    real_masks = prepare_mask_image(mask).to(device=device, dtype=weight_dtype)
    real_masks_copy = real_masks

    #! margin ratio
    real_images, params = adaptive_crop_with_margin(
        real_images, real_masks_copy, 
        margin_ratio=0.05, 
        target_size=(512, 384)
    )
    print(real_images.shape)
    
    real_masks, _ = adaptive_crop_with_margin(
        real_masks, real_masks_copy, 
        margin_ratio=0.05, 
        target_size=(512, 384)
    )
    real_images = real_images.to(device=device, dtype=weight_dtype)
    real_masks = real_masks.to(device=device, dtype=weight_dtype)

    condition_images = prepare_image(condition_image).to(device=device, dtype=weight_dtype)
    cloth_warp_images = prepare_image(cloth_warp_image).to(device=device, dtype=weight_dtype)
    cloth_warp_masks = prepare_mask_image(cloth_warp_mask).to(device=device, dtype=weight_dtype)

    extra_cond1_images = prepare_image(extra_cond1).to(device=device, dtype=weight_dtype)
    extra_cond2_images = prepare_image(extra_cond2).to(device=device, dtype=weight_dtype)

    # 在像素空间进行拼接
    # (B, 3, H, W)
    print(condition_images.shape)
    real_images_2 = torch.cat([real_images, condition_images], dim=-2)
    masked_real_images_1 = real_images * (real_masks < 0.5) # 保留黑色部分
    masked_real_images_2 = torch.cat([masked_real_images_1, condition_images], dim=-2)
    masks_2 = torch.cat([real_masks, torch.zeros_like(real_masks)], dim=-2)
    if reverse_right:
        masks_2_reverse = torch.cat([torch.zeros_like(real_masks), real_masks_copy], dim=-2)
    else:
        masks_2_reverse = torch.cat([real_masks_copy, torch.zeros_like(real_masks)], dim=-2)
    
    #! 把底图去掉试试
    # warped_masked_real_images_1 = (real_images_copy * (real_masks_copy < 0.5))  + (cloth_warp_images * (cloth_warp_masks >= 0.5)) 
    warped_masked_real_images_1 = (extra_cond2_images * (cloth_warp_masks < 0.5))  + (cloth_warp_images * (cloth_warp_masks >= 0.5)) 
    # warped_masked_real_images_1 = (torch.ones_like(real_images_copy) * ((real_masks_copy >= 0.5) ^ (cloth_warp_masks >= 0.5))) + (real_images_copy * (real_masks_copy < 0.5))  + (cloth_warp_images * (cloth_warp_masks >= 0.5)) 
    # warped_masked_real_images_1 =  cloth_warp_images
    #! 先尝试上面是extra_cond1_images，下面是warped_masked_real_images
    if reverse_right:
        warped_masked_real_images_2 = torch.cat([extra_cond1_images,warped_masked_real_images_1], dim=-2)
        warped_masked_real_images_2_target = torch.cat([extra_cond1_images,real_images_copy], dim=-2)
    else:
        warped_masked_real_images_2 = torch.cat([warped_masked_real_images_1,extra_cond1_images], dim=-2)
        warped_masked_real_images_2_target = torch.cat([real_images_copy,extra_cond1_images], dim=-2)

    if predict_together:
        real_images_4 = torch.cat([real_images_2, warped_masked_real_images_2_target], dim=-1)
    else:
        real_images_4 = torch.cat([real_images_2, warped_masked_real_images_2], dim=-1)
    masked_real_images_4 = torch.cat([masked_real_images_2, warped_masked_real_images_2], dim=-1)
    if predict_together:
        masks_4 = torch.cat([masks_2, masks_2_reverse], dim=-1)
    else:
        masks_4 = torch.cat([masks_2, torch.zeros_like(masks_2)], dim=-1)
    
    # VAE编码基础输入
    # 按照dim=-1分割
    real_images_4_1 = real_images_4.split(real_images_4.shape[-1] // 2, dim=-1)[0]
    real_images_4_2 = real_images_4.split(real_images_4.shape[-1] // 2, dim=-1)[1]

    masked_real_images_4_1 = masked_real_images_4.split(masked_real_images_4.shape[-1] // 2, dim=-1)[0]
    masked_real_images_4_2 = masked_real_images_4.split(masked_real_images_4.shape[-1] // 2, dim=-1)[1]

    masks_4_1 = masks_4.split(masks_4.shape[-1] // 2, dim=-1)[0]
    masks_4_2 = masks_4.split(masks_4.shape[-1] // 2, dim=-1)[1]

    real_image_latents_1 = compute_vae_encodings(real_images_4_1, vae)
    real_image_latents_2 = compute_vae_encodings(real_images_4_2, vae)
    masked_real_images_latents_1 = compute_vae_encodings(masked_real_images_4_1, vae)
    masked_real_images_latents_2 = compute_vae_encodings(masked_real_images_4_2, vae)
    mask_latent_1 = torch.nn.functional.interpolate(masks_4_1, size=real_image_latents_1.shape[-2:], mode="nearest")
    mask_latent_2 = torch.nn.functional.interpolate(masks_4_2, size=real_image_latents_2.shape[-2:], mode="nearest")
    
    real_image_latents = torch.cat([real_image_latents_1, real_image_latents_2], dim=-1)
    masked_real_images_latents = torch.cat([masked_real_images_latents_1, masked_real_images_latents_2], dim=-1)
    mask_latent = torch.cat([mask_latent_1, mask_latent_2], dim=-1)


    if image_encoder is not None:
        image_encoder.eval()
        image_hidden_states = image_encoder(condition_images)
        # print(image_hidden_states.shape)
        # print("image_encoder.show_trainable_params()")
        # image_encoder.show_trainable_params()

    if do_classifier_free_guidance := (guidance_scale > 1.0):
        # uncond_real_images_2 = torch.cat([masked_real_images_1, torch.zeros_like(condition_images)], dim=-2)
        # uncond_real_images_4 = torch.cat([uncond_real_images_2, torch.zeros_like(extra_cond1_images)], dim=-1)

        # 把condition_images变成全黑 -1
        uncond_masked_real_images_2 = torch.cat([masked_real_images_1, torch.full_like(condition_images, -1.0)], dim=-2)
        uncond_masked_real_images_4 = torch.cat([uncond_masked_real_images_2, torch.full_like(uncond_masked_real_images_2,-1.0)], dim=-1)

        uncond_masked_real_images_4_1 = uncond_masked_real_images_4.split(uncond_masked_real_images_4.shape[-1] // 2, dim=-1)[0]
        uncond_masked_real_images_4_2 = uncond_masked_real_images_4.split(uncond_masked_real_images_4.shape[-1] // 2, dim=-1)[1]

        uncond_masked_real_images_latents_1 = compute_vae_encodings(uncond_masked_real_images_4_1, vae)
        uncond_masked_real_images_latents_2 = compute_vae_encodings(uncond_masked_real_images_4_2, vae)

        # uncond_real_image_latents = compute_vae_encodings(uncond_real_images_4, vae)
        uncond_masked_real_images_latents = torch.cat([uncond_masked_real_images_latents_1, uncond_masked_real_images_latents_2], dim=-1)

        masked_latent_concat = torch.cat([uncond_masked_real_images_latents, masked_real_images_latents])
        mask_latent_concat = torch.cat([mask_latent] * 2)

        uncond_image_hidden_states = torch.zeros_like(image_hidden_states)

    # 准备初始噪声
    latents = randn_tensor(
        real_image_latents.shape,
        generator=generator,
        device=device,
        dtype=weight_dtype,
    )

    # 设置时间步
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps
    latents = latents * noise_scheduler.init_noise_sigma


    # 去噪循环的额外参数
    extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, generator, eta)
    num_warmup_steps = len(timesteps) - num_inference_steps * noise_scheduler.order

    # 去噪循环
    with tqdm.tqdm(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            non_inpainting_latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
            image_hidden_states_final = (torch.cat([image_hidden_states ,uncond_image_hidden_states] ) if do_classifier_free_guidance else image_hidden_states)
            non_inpainting_latent_model_input = noise_scheduler.scale_model_input(non_inpainting_latent_model_input, t)
            inpainting_latent_model_input = torch.cat([non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat], dim=1)

            noise_pred = unet(
                inpainting_latent_model_input,
                t.to(device),
                encoder_hidden_states=image_hidden_states_final,
                return_dict=False,
            )[0]
            
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % noise_scheduler.order == 0):
                progress_bar.update()
    
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents.to(vae.device, dtype=vae.dtype)).sample
    image_latent_left = latents.split(latents.shape[-1] // 2, dim=-1)[0]
    image_latent_right = latents.split(latents.shape[-1] // 2, dim=-1)[1]
    image_left = vae.decode(image_latent_left.to(vae.device, dtype=vae.dtype)).sample
    # 最终的图像，只取real_images对应的部分
    if not show_whole_image:
        image_latent = latents.split(latents.shape[-1] // 2, dim=-1)[0]
        image = image_latent.split(image_latent.shape[-2] // 2, dim=-2)[0]  # 根据latent_append_num来分割
        processed_images = image  # 假设处理后图像

        # 逆变换还原
        restored_images = inverse_transform(processed_images, params)

        # 验证尺寸
        print(f"原图尺寸: {real_images_copy.shape[2:]}")
        print(f"处理后尺寸: {image.shape[2:]}")
        print(f"还原后尺寸: {restored_images.shape[2:]}")

        # 将还原后的图片通过mask贴到原图上
        image = real_images_copy * (real_masks_copy < 0.5) + restored_images * (real_masks_copy >= 0.5)

    image = (image / 2 + 0.5).clamp(0, 1)
    image_left = (image_left / 2 + 0.5).clamp(0, 1)
    
    # 转换为PIL图像
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = numpy_to_pil(image)
    image_left = image_left.cpu().permute(0, 2, 3, 1).float().numpy()
    image_left = numpy_to_pil(image_left)

    # 将image_left保存到固定位置
    image_left[0].save("image_left.png")
    import os
    print("image_left.png saved to\n", os.path.abspath("image_left.png") + "\n")

    return image

import torch
import torch.nn.functional as F

def adaptive_crop_with_margin(real_images, real_masks, margin_ratio=0.1, target_size=(512, 384)):
    """
    带余量的mask自适应裁剪与缩放
    Args:
        real_images (Tensor): 输入图像 [B,C,H,W], 值域[0,1]
        real_masks (Tensor): 二值mask [B,1,H,W], 值0/1
        margin_ratio (float): 余量比例（相对于裁剪区域尺寸）
        target_size (tuple): 目标缩放尺寸 (H, W)
    Returns:
        cropped_resized (Tensor): 裁剪缩放后的图像 [B,C,target_H,target_W]
        transform_params (dict): 逆变换参数
    """
    device = real_images.device
    B, C, H, W = real_images.shape
    transform_params = []

    # 初始化结果张量
    cropped_resized = torch.zeros((B, C, target_size[0], target_size[1]), device=device)

    for b in range(B):
        # --- 步骤1：计算有效区域坐标 ---
        mask = (real_masks[b,0] >= 0.5).float()  # 二值化 [H,W]
        y_coords, x_coords = torch.where(mask == 1)
        
        if len(y_coords) == 0:  # 无mask情况处理
            cropped_resized[b] = F.interpolate(real_images[b:b+1], size=target_size, mode='bilinear')[0]
            transform_params.append(None)
            continue

        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        # --- 步骤2：计算扩展余量 ---
        crop_h = y_max - y_min
        crop_w = x_max - x_min
        h_margin = int(crop_h * margin_ratio)
        w_margin = int(crop_w * margin_ratio)

        # 边界保护
        y_start = max(0, y_min - h_margin)
        y_end = min(H, y_max + h_margin)
        x_start = max(0, x_min - w_margin)
        x_end = min(W, x_max + w_margin)

        # --- 步骤3：执行裁剪 ---
        cropped = real_images[b:b+1, :, y_start:y_end, x_start:x_end]  # [1,C,h_crop,w_crop]
        
        # --- 步骤4：缩放至目标尺寸 ---
        resized = F.interpolate(cropped, size=target_size, mode='bilinear', align_corners=False)
        cropped_resized[b] = resized[0]

        # --- 记录变换参数 ---
        params = {
            "original_shape": (H, W),
            "crop_coords": (y_start, y_end, x_start, x_end),
            "original_crop_size": (cropped.shape[2], cropped.shape[3]),
            "target_size": target_size
        }
        transform_params.append(params)

    return cropped_resized, transform_params

def inverse_transform(processed_images, transform_params):
    """
    逆变换：将处理后的图像还原至原图位置
    Args:
        processed_images (Tensor): 处理后的图像 [B,C,target_H,target_W]
        transform_params (list): 变换参数列表
    Returns:
        restored (Tensor): 还原后的图像 [B,C,original_H,original_W]
    """
    device = processed_images.device
    B, C, _, _ = processed_images.shape
    restored = torch.zeros((B, C, 
                          transform_params[0]["original_shape"][0], 
                          transform_params[0]["original_shape"][1]), device=device)

    for b in range(B):
        if transform_params[b] is None:
            restored[b] = F.interpolate(processed_images[b:b+1], 
                                      size=transform_params[b]["original_shape"], 
                                      mode='bilinear')[0]
            continue

        # --- 逆缩放 ---
        resized_back = F.interpolate(
            processed_images[b:b+1], 
            size=transform_params[b]["original_crop_size"],
            mode='bilinear',
            align_corners=False
        )

        # --- 贴回原图位置 ---
        y_start, y_end, x_start, x_end = transform_params[b]["crop_coords"]
        restored[b:b+1, :, y_start:y_end, x_start:x_end] = resized_back

    return restored


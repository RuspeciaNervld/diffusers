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

def prepare_masked_image_tensor(image, mask, device = 'cuda', weight_dtype=torch.float32) -> torch.Tensor:
    real_images = prepare_image(image).to(device=device, dtype=weight_dtype)
    real_masks = prepare_image(mask).to(device=device, dtype=weight_dtype)
    
    # 把mask转为单通道
    real_masks = real_masks[:, 0:1, :, :]

    # 准备masked_image
    masked_real_images = real_images * (real_masks < 0.5)
    return masked_real_images

def prepare_warpped_masked_image_tensor(image, mask, cloth_warp_image, cloth_warp_mask, device = 'cuda', weight_dtype=torch.float32) -> torch.Tensor:
    real_images = prepare_image(image).to(device=device, dtype=weight_dtype)
    real_masks = prepare_image(mask).to(device=device, dtype=weight_dtype)
    cloth_warp_images = prepare_image(cloth_warp_image).to(device=device, dtype=weight_dtype)
    cloth_warp_masks = prepare_image(cloth_warp_mask).to(device=device, dtype=weight_dtype)
    
    # 把mask转为单通道
    real_masks = real_masks[:, 0:1, :, :]
    cloth_warp_masks = cloth_warp_masks[:, 0:1, :, :]
    
    # 准备masked_image
    masked_real_images = real_images * (real_masks < 0.5)
    
    # 将cloth_warp_images通过cloth_warp_masks贴到masked_real_images上
    cloth_warp_part = cloth_warp_images * (cloth_warp_masks >= 0.5)
    masked_part = masked_real_images + cloth_warp_part
    
    return masked_part


@torch.no_grad()
def run_inference(
    unet,
    vae,
    noise_scheduler,
    device='cuda',
    weight_dtype=torch.float32,
    image: Union[torch.Tensor, Image.Image]=None,  # real_images
    mask: Union[torch.Tensor, Image.Image]=None,   # real_masks
    condition_image: Union[torch.Tensor, Image.Image]=None,  # condition_images
    openpose_image: Union[torch.Tensor, Image.Image]=None,  # openpose_images
    cloth_warp_image: Union[torch.Tensor, Image.Image]=None,  # cloth_warp_images
    cloth_warp_mask: Union[torch.Tensor, Image.Image]=None,  # cloth_warp_masks
    canny_image: Union[torch.Tensor, Image.Image]=None,  # canny_images
    latent_append_num: int = 1, # 1就是默认的CatVTON，2就是CatVTON+Openpose，3就是CatVTON+Canny+Openpose
    num_inference_steps: int = 50,
    guidance_scale: float = 2.5,
    generator=None,
    eta=1.0,
    **kwargs
):
    # 把模型移动到cuda
    unet.to(device)
    vae.to(device)

    # 准备基础输入
    real_images = prepare_image(image).to(device=device, dtype=weight_dtype)
    real_masks = prepare_image(mask).to(device=device, dtype=weight_dtype)
    if condition_image is not None:
        condition_images = prepare_image(condition_image).to(device=device, dtype=weight_dtype)
    else:
        condition_images = None
    real_masks = real_masks[:, 0:1, :, :]  # 把mask转为单通道
    
    # VAE编码基础输入
    real_image_latents = compute_vae_encodings(real_images, vae)
    if condition_images is not None:
        condition_latents = compute_vae_encodings(condition_images, vae)
    else:
        condition_latents = torch.full_like(real_image_latents, -1.0)
    
    # 准备masked_image
    masked_real_images = real_images * (real_masks < 0.5)
    
    # 处理cloth_warp
    if cloth_warp_image is not None and cloth_warp_mask is not None:
        cloth_warp_images = prepare_image(cloth_warp_image).to(device=device, dtype=weight_dtype)
        cloth_warp_masks = prepare_image(cloth_warp_mask).to(device=device, dtype=weight_dtype)
        cloth_warp_masks = cloth_warp_masks[:, 0:1, :, :]
        
        # 将cloth_warp_images通过cloth_warp_masks贴到masked_real_images上
        cloth_warp_part = cloth_warp_images * (cloth_warp_masks >= 0.5)
        masked_part = masked_real_images + cloth_warp_part
    else:
        masked_part = masked_real_images
    
    masked_part_latents = compute_vae_encodings(masked_part, vae)
    
    # 根据latent_append_num决定拼接方式
    latents_to_concat = [real_image_latents, condition_latents]
    masks_to_concat = [mask_latent := torch.nn.functional.interpolate(real_masks, size=real_image_latents.shape[-2:], mode="nearest"),
                      torch.zeros_like(mask_latent)]
    masked_latents_to_concat = [masked_part_latents, condition_latents]
    
    if latent_append_num >= 2:  # 需要Openpose
        if openpose_image is not None:
            openpose_latents = compute_vae_encodings(prepare_image(openpose_image).to(device=device, dtype=weight_dtype), vae)
        else:
            openpose_latents = torch.full_like(condition_latents, -1.0)
        latents_to_concat.append(openpose_latents)
        masks_to_concat.append(torch.zeros_like(mask_latent))
        masked_latents_to_concat.append(openpose_latents)
    
    if latent_append_num >= 3:  # 需要Canny
        if canny_image is not None:
            canny_latents = compute_vae_encodings(prepare_image(canny_image).to(device=device, dtype=weight_dtype), vae)
        else:
            canny_latents = torch.full_like(condition_latents, -1.0)
        latents_to_concat.append(canny_latents)
        masks_to_concat.append(torch.zeros_like(mask_latent))
        masked_latents_to_concat.append(canny_latents)
    
    # 拼接latents
    latent_model_input_p1 = torch.cat(latents_to_concat, dim=-2)
    mask_latent_concat = torch.cat(masks_to_concat, dim=-2)
    masked_latent_concat = torch.cat(masked_latents_to_concat, dim=-2)

    # 准备初始噪声
    latents = randn_tensor(
        latent_model_input_p1.shape,
        generator=generator,
        device=device,
        dtype=weight_dtype,
    )

    # 设置时间步
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps
    latents = latents * noise_scheduler.init_noise_sigma

    # Classifier-Free Guidance
    if do_classifier_free_guidance := (guidance_scale > 1.0):
        # 创建无条件分支
        zero_latents = [torch.zeros_like(l) for l in latents_to_concat[1:]]  # 跳过real_image_latents
        uncond_masked_latent_concat = torch.cat([masked_part_latents] + zero_latents, dim=-2)
        masked_latent_concat = torch.cat([uncond_masked_latent_concat, masked_latent_concat])
        mask_latent_concat = torch.cat([mask_latent_concat] * 2)

    # 去噪循环的额外参数
    extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, generator, eta)
    num_warmup_steps = len(timesteps) - num_inference_steps * noise_scheduler.order

    # 去噪循环
    with tqdm.tqdm(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            non_inpainting_latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
            non_inpainting_latent_model_input = noise_scheduler.scale_model_input(non_inpainting_latent_model_input, t)
            inpainting_latent_model_input = torch.cat([non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat], dim=1)
            
            noise_pred = unet(
                inpainting_latent_model_input,
                t.to(device),
                encoder_hidden_states=None,
                return_dict=False,
            )[0]
            
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % noise_scheduler.order == 0):
                progress_bar.update()

    # 解码最终的潜变量（只取real_images对应的部分）
    latents = latents.split(latents.shape[-2] // (latent_append_num + 1), dim=-2)[0]  # 根据latent_append_num来分割
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents.to(vae.device, dtype=vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    
    # 转换为PIL图像
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = numpy_to_pil(image)

    return image
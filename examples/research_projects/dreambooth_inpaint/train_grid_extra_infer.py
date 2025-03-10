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

        # # 如果像素值在0-1之间，则乘以255
        # if image.min() >= 0 and image.max() <= 1:
        #     image = image * 255.0
        # # 如果像素值在0-255之间，则除以127.5
        # if image.min() >= 0 and image.max() <= 255:
        #     image = image.to(dtype=torch.float32) / 127.5 - 1.0
        # else:
        #     image = image.to(dtype=torch.float32)

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
    use_warp_as_condition: bool = False,
    use_origin_condition: bool = True,
    # 除了上面的是必须存在的，下面的根据需要拼接
    extra_cond1: Union[torch.Tensor, Image.Image]=None,  # extra_condition_images
    extra_cond2: Union[torch.Tensor, Image.Image]=None,  # extra_condition_images
    extra_cond3: Union[torch.Tensor, Image.Image]=None,  # extra_condition_images
    num_inference_steps: int = 50,
    guidance_scale: float = 2.5,
    generator=None,
    eta=1.0,
    show_whole_image:bool = False,
    **kwargs
):
    # 打印推理参数，重点是用哪些latent
    print("像素空间四宫格：原图+warp图；condition+extra_cond1")

    # 把模型移动到cuda
    unet.to(device)
    vae.to(device)

    # 准备基础输入
    real_images = prepare_image(image).to(device=device, dtype=weight_dtype)
    real_masks = prepare_mask_image(mask).to(device=device, dtype=weight_dtype)
    condition_images = prepare_image(condition_image).to(device=device, dtype=weight_dtype)
    cloth_warp_images = prepare_image(cloth_warp_image).to(device=device, dtype=weight_dtype)
    cloth_warp_masks = prepare_mask_image(cloth_warp_mask).to(device=device, dtype=weight_dtype)
    extra_cond1_images = prepare_image(extra_cond1).to(device=device, dtype=weight_dtype)

    # 在像素空间进行拼接
    # (B, 3, H, W)
    real_images_2 = torch.cat([real_images, condition_images], dim=-2)
    masked_real_images_1 = real_images * (real_masks < 0.5) # 保留黑色部分
    masked_real_images_2 = torch.cat([masked_real_images_1, condition_images], dim=-2)
    masks_2 = torch.cat([real_masks, torch.zeros_like(real_masks)], dim=-2)
    masks_2_reverse = torch.cat([torch.zeros_like(real_masks), real_masks], dim=-2)
    
    warped_masked_real_images_1 = masked_real_images_1 + (cloth_warp_images * (cloth_warp_masks >= 0.5))
    #! 先尝试上面是extra_cond1_images，下面是warped_masked_real_images
    warped_masked_real_images_2 = torch.cat([extra_cond1_images,warped_masked_real_images_1], dim=-2)
    warped_masked_real_images_2_target = torch.cat([extra_cond1_images,real_images], dim=-2)

    #! 这里可以是warped_masked_real_images_2，也可以是warped_masked_real_images_2_target，取决于想不想让warp部分也预测
    real_images_4 = torch.cat([real_images_2, warped_masked_real_images_2], dim=-1)
    masked_real_images_4 = torch.cat([masked_real_images_2, warped_masked_real_images_2], dim=-1)
    #! 如果上面用了warped_masked_real_images_2_target，这里mask需要选下面一个
    masks_4 = torch.cat([masks_2, torch.zeros_like(masks_2)], dim=-1)
    # masks_4 = torch.cat([masks_2, masks_2_reverse], dim=-1)
    
    # VAE编码基础输入
    real_image_latents = compute_vae_encodings(real_images_4, vae)
    masked_real_images_latents = compute_vae_encodings(masked_real_images_4, vae)
    mask_latent = torch.nn.functional.interpolate(masks_4, size=real_image_latents.shape[-2:], mode="nearest")

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

        # uncond_real_image_latents = compute_vae_encodings(uncond_real_images_4, vae)
        uncond_masked_real_images_latents = compute_vae_encodings(uncond_masked_real_images_4, vae)

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
            
            # print("inpainting_latent_model_input.shape", inpainting_latent_model_input.shape)
            # print("image_hidden_states.shape", image_hidden_states.shape)
            # print("image_hidden_states_final.shape", image_hidden_states_final.shape)
            # print("non_inpainting_latent_model_input.shape", non_inpainting_latent_model_input.shape)
            # print("mask_latent_concat.shape", mask_latent_concat.shape)
            # print("masked_latent_concat.shape", masked_latent_concat.shape)

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

    # 解码最终的潜变量（只取real_images对应的部分）
    if not show_whole_image:
        latents = latents.split(latents.shape[-2] // 2, dim=-2)[0]  # 根据latent_append_num来分割
        latents = latents.split(latents.shape[-1] // 2, dim=-1)[0]  # 根据latent_append_num来分割
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents.to(vae.device, dtype=vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    
    # 转换为PIL图像
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = numpy_to_pil(image)

    return image
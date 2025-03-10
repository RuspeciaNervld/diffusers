import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm  # 可选，用于进度条

source_base_dir = "/home/nervld/gitclone/diffusers/data/vton_test/real_images"       # 原图目录
source_prediction_dir = "/home/nervld/gitclone/diffusers/output/vton_test_small/像素四宫格_同步学习_step_1000" # 预测图目录
mask_dir = "/home/nervld/gitclone/diffusers/data/vton_test/real_masks"                    # 掩膜目录（白色为蒙版区域）
output_dir = source_prediction_dir + "_update"               # 输出目录

# 创建输出目录（若不存在）
os.makedirs(output_dir, exist_ok=True)


# 获取预测目录中所有文件名
pred_files = os.listdir(source_prediction_dir)

for file_name in tqdm(pred_files):
    # 检查原图和掩膜是否存在同名文件
    base_path = os.path.join(source_base_dir, file_name)
    pred_path = os.path.join(source_prediction_dir, file_name)
    mask_path = os.path.join(mask_dir, file_name)
    
    if not (os.path.exists(base_path) and os.path.exists(mask_path)):
        print(f"跳过文件 {file_name}（原图或掩膜缺失）")
        continue

    # 加载图像和掩膜（使用OpenCV处理）
    base_img = cv2.imread(base_path)           # 原图（BGR格式）
    pred_img = cv2.imread(pred_path)           # 预测图（BGR格式）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 单通道掩膜[1,2](@ref)

    # 将掩膜二值化（确保白色区域为255）
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 调整掩膜尺寸（若与原图尺寸不一致）
    if mask_binary.shape != base_img.shape[:2]:
        mask_binary = cv2.resize(mask_binary, (base_img.shape[1], base_img.shape[0]))

    # 将预测图按掩膜复制到原图（仅保留白色区域）
    masked_pred = cv2.bitwise_and(pred_img, pred_img, mask=mask_binary)
    masked_base = cv2.bitwise_and(base_img, base_img, mask=cv2.bitwise_not(mask_binary))
    combined = cv2.add(masked_base, masked_pred)

    # 保存结果
    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, combined)

print(output_dir)
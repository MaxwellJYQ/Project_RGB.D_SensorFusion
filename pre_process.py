import os
import cv2
import numpy as np

# 输入文件夹和输出文件夹路径
input_folder = 'dataset/output'
output_folder = 'dataset/output'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith("_depth.png"):
        # 读取输入图片
        input_image = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_UNCHANGED)

        # 生成高斯噪声
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gaussian_noise = np.random.normal(mean, sigma, input_image.shape).astype(np.uint16)

        # 添加高斯噪声
        noisy_image = cv2.add(input_image, gaussian_noise)

        # 8倍降采样
        downsampled_image = cv2.resize(noisy_image, None, fx=1/8, fy=1/8, interpolation=cv2.INTER_LINEAR)

        # 插值回来
        upsampled_image = cv2.resize(downsampled_image, (noisy_image.shape[1], noisy_image.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 保存处理后的图片到输出文件夹
        output_filename = os.path.join(output_folder, os.path.splitext(filename)[0] + '_noise.png')
        cv2.imwrite(output_filename, upsampled_image)

        print(f"Processed: {filename}")

print("All images processed.")

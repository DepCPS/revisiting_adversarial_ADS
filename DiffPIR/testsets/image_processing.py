import os
import cv2
import numpy as np
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

# -------------------------------
# 1. 中值模糊处理函数
# -------------------------------
def apply_median_blur(img, kernel_size=3):
    """
    对图像应用中值模糊
    :param img: 输入图像
    :param kernel_size: 核大小，须为奇数
    :return: 模糊后的图像
    """
    return cv2.medianBlur(img, kernel_size)

# -------------------------------
# 2. 位深度降低函数
# -------------------------------
def bit_depth_reduction(img, bits=5):
    """
    降低图像位深度，默认保留5位
    :param img: 输入图像（8位图像）
    :param bits: 保留的位数
    :return: 位深度降低后的图像
    """
    shift = 8 - bits  # 原图通常为8位
    return np.left_shift(np.right_shift(img, shift), shift)

# -------------------------------
# 3. 随机化处理函数
# -------------------------------
def randomization(img, scale_range=(0.9, 1.1)):
    """
    通过随机缩放和填充/裁剪来对图像进行随机化处理
    :param img: 输入图像
    :param scale_range: 缩放比例范围
    :return: 随机化后的图像
    """
    h, w = img.shape[:2]
    scale = random.uniform(*scale_range)
    new_h, new_w = int(h * scale), int(w * scale)
    # 随机缩放
    img_resized = cv2.resize(img, (new_w, new_h))
    
    if new_h < h or new_w < w:
        # 缩放后图像尺寸较小时，随机填充到原始尺寸
        top = random.randint(0, h - new_h)
        bottom = h - new_h - top
        left = random.randint(0, w - new_w)
        right = w - new_w - left
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_REFLECT)
        img_final = img_padded[0:h, 0:w]  # 裁剪到原始尺寸
    else:
        # 缩放后图像尺寸较大时，随机裁剪到原始尺寸
        start_y = random.randint(0, new_h - h)
        start_x = random.randint(0, new_w - w)
        img_final = img_resized[start_y:start_y+h, start_x:start_x+w]
    
    return img_final

# -------------------------------
# 4. 卷积自编码器模型构建
# -------------------------------
def build_autoencoder(input_shape):
    """
    构建一个简单的卷积自编码器
    :param input_shape: 输入图像的形状 (height, width, channels)
    :return: 编译好的自编码器模型
    """
    input_img = Input(shape=input_shape)

    # 编码器部分
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # 解码器部分
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

# -------------------------------
# 5. 主函数：分别处理图像并保存到不同文件夹
# -------------------------------
def main(input_folder, output_folder_base):
    # 定义4种处理方法对应的输出子文件夹
    output_folder_median = os.path.join(output_folder_base, 'median_blur')
    output_folder_bitdepth = os.path.join(output_folder_base, 'bit_depth_reduction')
    output_folder_random = os.path.join(output_folder_base, 'randomization')
    output_folder_autoencoder = os.path.join(output_folder_base, 'autoencoder')

    # 如果输出文件夹不存在，则创建它们
    for folder in [output_folder_median, output_folder_bitdepth, output_folder_random, output_folder_autoencoder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # 为了构建自编码器，需要确定图像尺寸。这里取输入文件夹中的第一张图像作为样本
    sample_img_path = None
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            sample_img_path = os.path.join(input_folder, filename)
            break
    if sample_img_path is None:
        print("未找到图像文件！")
        return

    sample_img = cv2.imread(sample_img_path)
    height, width, channels = sample_img.shape
    input_shape = (height, width, channels)
    
    # 构建自编码器模型（如有预训练权重，请取消下面注释并提供正确路径）
    autoencoder = build_autoencoder(input_shape)
    # autoencoder.load_weights('path_to_weights.h5')
    
    # 遍历输入文件夹中的所有图像
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"读取图像失败: {img_path}")
                continue
            
            # 如果图像尺寸与样本尺寸不一致，则统一resize
            if img.shape != input_shape:
                img = cv2.resize(img, (width, height))
            
            # 1. 中值模糊处理
            img_median = apply_median_blur(img, kernel_size=3)
            cv2.imwrite(os.path.join(output_folder_median, filename), img_median)
            
            # 2. 位深度降低处理
            img_bitdepth = bit_depth_reduction(img, bits=5)
            cv2.imwrite(os.path.join(output_folder_bitdepth, filename), img_bitdepth)
            
            # 3. 随机化处理
            img_random = randomization(img, scale_range=(0.9, 1.1))
            cv2.imwrite(os.path.join(output_folder_random, filename), img_random)
            
            # 4. 卷积自编码器去噪处理
            # 归一化图像到[0,1]
            img_norm = img.astype('float32') / 255.0
            # 扩展batch维度
            img_input = np.expand_dims(img_norm, axis=0)
            # 利用自编码器进行预测
            img_auto = autoencoder.predict(img_input)
            img_auto = np.squeeze(img_auto, axis=0)
            # 还原到0-255范围
            img_auto = (img_auto * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_folder_autoencoder, filename), img_auto)
            
            print(f"已处理并保存图像: {filename}")

if __name__ == '__main__':
    # 修改为你的输入文件夹路径和保存结果的基础文件夹路径
    input_folder = 'gauss'
    output_folder_base = 'result_gauss'
    main(input_folder, output_folder_base)

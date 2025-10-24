import os
import numpy as np
from PIL import Image
from scipy.io import savemat

# 设置图像所在的文件夹路径
image_folder = '/media/super/DATA/tianxiao/Fast-DDPM_laplas/data/pelvic/pelvic_val/CT_val'

# 获取所有jpg文件名
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 初始化图像数据列表
images = []

# 读取每个图像并处理
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    # 使用PIL读取图像
    img = Image.open(image_path).convert('RGB')  # 转换为RGB模式，若是灰度图则可以去掉convert('RGB')
    
    # 将图像转换为numpy数组并归一化到 [0, 1] 范围
    img_array = np.array(img) / 255.0  # 图像数据归一化
    
    # 如果是灰度图（单通道），去掉多余的维度
    if img_array.ndim == 3:
        img_array = np.mean(img_array, axis=-1)  # 将RGB图像转为灰度图

    # 添加到图像列表
    images.append(img_array)

# 将图像列表转换为 numpy 数组
images_array = np.array(images)

# 保存为.mat文件
savemat('Pelvic_val_CT.mat', {'images': images_array})

print('Conversion to .mat file completed!')

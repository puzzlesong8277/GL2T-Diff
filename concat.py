import os
from PIL import Image

# 图像文件夹路径
image_folder = '/media/super/DATA/tianxiao/Fast-DDPM_laplas/compare_img/BraTS_T1toT2'

# 读取文件夹中的所有图像文件名
image_files = os.listdir(image_folder)

# 按照规则分类图像文件
source_images = sorted([f for f in image_files if 'st' in f])
reference_images = sorted([f for f in image_files if 'gt' in f])
pt1_images = sorted([f for f in image_files if 'pt1' in f])
pt2_images = sorted([f for f in image_files if 'pt2' in f])
pt3_images = sorted([f for f in image_files if 'pt3' in f])

# 每一行图像的顺序：[source, reference, pix2pix, Fast-DDPM, ours]
rows = list(zip(source_images, reference_images, pt3_images, pt2_images, pt1_images))

# 打开所有图像并计算最终拼接图像的尺寸
images = [[Image.open(os.path.join(image_folder, img)) for img in row] for row in rows]
width, height = images[0][0].size
total_width = width * 5
total_height = height * len(rows)

# 创建一个新的图像以拼接所有图像
new_image = Image.new('RGB', (total_width, total_height))

# 将每个图像按顺序粘贴到新图像中
for i, row in enumerate(images):
    for j, img in enumerate(row):
        new_image.paste(img, (j * width, i * height))

# 保存拼接后的图像
new_image.save('/media/super/DATA/tianxiao/Fast-DDPM_laplas/compare_img/BraTS_T1toT2/concatenated_image.png')

print("Image concatenation complete!")



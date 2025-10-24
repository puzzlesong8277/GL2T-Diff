import re
import csv

# 定义日志文件路径和输出的 CSV 文件路径
log_file_path = '/media/super/DATA/tianxiao/Fast-DDPM_laplas/image_metrics_Pelvic_MRItoCT.log'  # 这里替换成你的 log 文件路径
csv_file_path = '/media/super/DATA/tianxiao/Fast-DDPM_laplas/image_metrics_Pelvic_MRItoCT_output.csv'  # 这里替换成你想保存的 CSV 文件路径

# 读取 log 文件
with open(log_file_path, 'r') as log_file:
    log_lines = log_file.readlines()

# 用于存储提取的数据
data = []

# 正则表达式匹配 PSNR 和 SSIM 数值
pattern = re.compile(r"Image:\s+([\w\-_]+),\s+PSNR:\s+([0-9\.]+),\s+SSIM:\s+([0-9\.]+)")

# 遍历每一行
for line in log_lines:
    match = pattern.search(line)
    if match:
        image_name = match.group(1)
        psnr = match.group(2)
        ssim = match.group(3)
        # 将提取的内容添加到数据列表
        data.append([image_name, psnr, ssim])

# 将数据写入 CSV 文件
with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # 写入表头
    writer.writerow(['Image', 'PSNR', 'SSIM'])
    # 写入数据
    writer.writerows(data)

print(f"CSV file has been created: {csv_file_path}")

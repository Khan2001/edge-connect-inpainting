# 导入所需库
import os
import argparse
import numpy as np

'''
这段代码的作用是将指定路径（path）下的所有图片文件保存到一个txt文件中，并按文件名排序。
'''

# 创建命令行参数对象
parser = argparse.ArgumentParser()
# 添加命令行参数
parser.add_argument('--path', type=str, help='path to the dataset') # 数据集路径
parser.add_argument('--output', type=str, help='path to the file list') # 输出文件列表的路径
args = parser.parse_args()

# 支持的文件扩展名
ext = {'.jpg', '.png'}
# 存储所有图片文件的路径列表
images = []

# 遍历指定路径下的所有文件和子目录
for root, dirs, files in os.walk(args.path):
    print('loading ' + root) # 输出正在加载的目录
    for file in files:
        if os.path.splitext(file)[1] in ext: # 如果文件扩展名是支持的扩展名之一
            images.append(os.path.join(root, file)) # 将文件的完整路径添加到列表中

# 按文件名排序
images = sorted(images)

# 将所有文件路径保存到txt文件中
np.savetxt(args.output, images, fmt='%s')

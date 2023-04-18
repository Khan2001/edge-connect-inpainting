# 导入所需库
import os
import argparse
import numpy as np

'''
这段代码的作用是将指定路径（path）下的图片数据集按照训练集、验证集和测试集的比例进行划分，
并将划分后的文件列表保存到指定路径（output）下的三个txt文件中（train.flist，val.flist和test.flist）。
'''

# 创建命令行参数对象
parser = argparse.ArgumentParser()
# 添加命令行参数
parser.add_argument('--path', type=str, help='path to the dataset') # 数据集路径
parser.add_argument('--train', type=int, default=28, help='number of train in a iter') # 每轮迭代中训练集的数量，默认为28
parser.add_argument('--val', type=int, default=1, help='number of val in a iter') # 每轮迭代中验证集的数量，默认为1
parser.add_argument('--test', type=int, default=1, help='number of test in a iter') # 每轮迭代中测试集的数量，默认为1
parser.add_argument('--output', type=str, help='path to the three file lists') # 输出文件列表的路径
args = parser.parse_args()

# 支持的文件扩展名
ext = {'.jpg', '.png'}
# 计算数据集总数
total = args.train + args.val + args.test
# 存储训练集、验证集和测试集的文件路径列表
images_train = []
images_val = []
images_test = []

num = 1
# 遍历指定路径下的所有文件和子目录
for root, dirs, files in os.walk(args.path):
    print('loading ' + root) # 输出正在加载的目录
    for file in files:
        if os.path.splitext(file)[1] in ext: # 如果文件扩展名是支持的扩展名之一
            path = os.path.join(root, file) # 获取文件的完整路径
            # 将文件添加到训练集、验证集或测试集中
            if num % total > (args.val + args.test) or num % total == 0:
                images_train.append(path)
            elif num % total <= args.val and num % total > 0:
                images_val.append(path)
            else:
                images_test.append(path)
            num += 1

# 按文件名排序
images_train.sort()
images_val.sort()
images_test.sort()

# 输出各集合中的数据个数
print('train number =', len(images_train))
print('val number =', len(images_val))
print('test number =', len(images_test))

# 如果输出文件夹不存在，则创建
if not os.path.exists(args.output):
    os.mkdir(args.output)

# 将各个集合中的文件列表保存到txt文件中
np.savetxt(os.path.join(args.output, 'train.flist'), images_train, fmt='%s')
np.savetxt(os.path.join(args.output, 'val.flist'), images_val, fmt='%s')
np.savetxt(os.path.join(args.output, 'test.flist'), images_test, fmt='%s')

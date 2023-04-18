import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from glob import glob
from ntpath import basename
from scipy.misc import imread # 导入imread模块，用于读取图像文件
from skimage.measure import compare_ssim # 导入compare_ssim模块，用于计算结构相似度
from skimage.measure import compare_psnr # 导入compare_psnr模块，用于计算峰值信噪比
from skimage.color import rgb2gray # 导入rgb2gray模块，用于将彩色图像转换为灰度图像


'''
几种与图像质量和相似性相关的指标，包括结构相似性指标（SSIM）和峰值信噪比（PSNR）。
'''

def parse_args(): # 定义parse_args函数，用于解析命令行参数并返回args对象
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', help='Path to ground truth data', type=str) # 添加--data-path参数，表示真实数据集的路径
    parser.add_argument('--output-path', help='Path to output data', type=str) # 添加--output-path参数，表示输出结果的路径
    parser.add_argument('--debug', default=0, help='Debug', type=int) # 添加--debug参数，表示是否开启调试模式，默认不开启
    args = parser.parse_args() # 解析命令行参数
    return args # 返回args对象


def compare_mae(img_true, img_test): # 定义compare_mae函数，用于计算平均绝对误差
    img_true = img_true.astype(np.float32) # 将图像类型转换为float32
    img_test = img_test.astype(np.float32) # 将图像类型转换为float32
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test) # 返回平均绝对误差


args = parse_args() # 解析命令行参数并将结果赋值给args
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg)) # 打印每个参数的名称和取值

path_true = args.data_path # 真实数据集的路径
path_pred = args.output_path # 输出结果的路径

psnr = [] # 初始化psnr列表
ssim = [] # 初始化ssim列表
mae = [] # 初始化mae列表
names = [] # 初始化names列表
index = 1 # 初始化index为1

files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png')) # 获取真实数据集中所有jpg和png格式的文件名，并将结果存储在files列表中
for fn in sorted(files): # 循环遍历每个文件名
    name = basename(str(fn)) # 获取文件名（不包含路径）
    names.append(name) # 将文件名添加到names列表中

    img_gt = (imread(str(fn)) / 255.0).astype(np.float32) # 读取真实图像，并将像素值范围归一化到[0,1]，然后将数据类型转换为float32
    img_pred = (imread(path_pred + '/' + basename(str(fn))) / 255.0).astype(np.float32) # 读取输出图像，并将像素值范围归一化到[0,1]，然后将数据类型转换为float32

    img_gt = rgb2gray(img_gt) # 将真实图像转换为灰度图像
    img_pred = rgb2gray(img_pred) # 将输出图像转换为灰度图像

    if args.debug != 0: # 如果开启了调试模式
        plt.subplot('121') # 创建一个子图
        plt.imshow(img_gt) # 显示真实图像
        plt.title('Groud truth') # 设置标题
        plt.subplot('122') # 创建另一个子图
        plt.imshow(img_pred) # 显示输出图像
        plt.title('Output') # 设置标题
        plt.show() # 显示图像

    psnr.append(compare_psnr(img_gt, img_pred, data_range=1)) # 计算峰值信噪比，并将结果添加到psnr列表中
    ssim.append(compare_ssim(img_gt, img_pred, data_range=1, win_size=51)) # 计算结构相似度，并将结果添加到ssim列表中
    mae.append(compare_mae(img_gt, img_pred)) # 计算平均绝对误差，并将结果添加到mae列表中
    if np.mod(index, 100) == 0: # 如果处理的图像数是100的倍数
        print(
            str(index) + ' images processed', # 打印已处理的图像数
            "PSNR: %.4f" % round(np.mean(psnr), 4), # 打印平均峰值信噪比
            "SSIM: %.4f" % round(np.mean(ssim), 4), # 打印平均结构相似度
            "MAE: %.4f" % round(np.mean(mae), 4), # 打印平均平均绝对误差
        )
    index += 1 # 处理下一张图像

np.savez(args.output_path + '/metrics.npz', psnr=psnr, ssim=ssim, mae=mae, names=names) # 将psnr、ssim、mae和names保存为一个.npz文件
print(
    "PSNR: %.4f" % round(np.mean(psnr), 4), # 打印平均峰值信噪比
    "PSNR Variance: %.4f" % round(np.var(psnr), 4), # 打印峰值信噪比的方差
    "SSIM: %.4f" % round(np.mean(ssim), 4), # 打印平均结构相似度
    "SSIM Variance: %.4f" % round(np.var(ssim), 4), # 打印结构相似度的方差
    "MAE: %.4f" % round(np.mean(mae), 4), # 打印平均平均绝对误差
    "MAE Variance: %.4f" % round(np.var(mae), 4) # 打印平均绝对误差的方差
)
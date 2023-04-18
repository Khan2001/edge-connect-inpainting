import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray
from .utils import create_mask

# 定义数据集类
class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist) # 加载原始图像列表
        self.edge_data = self.load_flist(edge_flist) # 加载边缘图像列表
        self.mask_data = self.load_flist(mask_flist) # 加载掩码图像列表

        self.input_size = config.INPUT_SIZE # 输入图像大小
        self.sigma = config.SIGMA # 边缘检测中的高斯滤波器参数
        self.edge = config.EDGE # 是否使用边缘图像
        self.mask = config.MASK # 是否使用掩码图像
        self.nms = config.NMS # 是否进行非极大值抑制

        # 在测试模式下，掩码图像和原始图像一一对应
        # 掩码图像不需随机加载
        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # 加载图像
        img = imread(self.data[index])

        # 如果需要，进行尺寸调整
        if size != 0:
            img = self.resize(img, size, size)

        # 创建灰度图像
        img_gray = rgb2gray(img)

        # 加载掩码图像
        mask = self.load_mask(img, index)

        # 加载边缘图像
        edge = self.load_edge(img_gray, index, mask)

        # 数据增强
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        # 转换图像为张量
        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)

    # 加载边缘图像
    def load_edge(self, img, index, mask):
        # 获取sigma的值
        sigma = self.sigma

        # 如果是测试模式，则使用掩码图像（掩盖掉掩码区域）防止Canny算法检测掩码区域的边缘
        mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # 判断边缘类型：无边缘
        if self.edge == 1:
            # 如果sigma为-1，则返回全零数组
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float)

            # 如果sigma为0，则随机选择1-4之间的整数作为高斯滤波器参数sigma
            if sigma == 0:
                sigma = random.randint(1, 4)

            # 调用canny函数生成边缘信息，并将其返回
            return canny(img, sigma=sigma, mask=mask).astype(np.float)

        # 判断边缘类型：外部边缘
        else:
            # 获取图像的高度和宽度
            imgh, imgw = img.shape[0:2]
            # 从已有的边缘数据集中随机选择一张，并将其调整到和原图像大小相同
            edge = imread(self.edge_data[index])
            edge = self.resize(edge, imgh, imgw)

            # 判断是否进行非极大值抑制
            if self.nms == 1:
                # 对于原图像和边缘图像分别调用canny函数生成边缘信息，然后进行点乘操作
                edge = edge * canny(img, sigma=sigma, mask=mask)

            return edge

    # 加载掩码图像
    def load_mask(self, img, index):
        # 获取图像的高度和宽度
        imgh, imgw = img.shape[0:2]
        # 获取掩码类型
        mask_type = self.mask

        # 判断掩码类型：外部掩码 + 随机块
        if mask_type == 4:
            # 以50%的概率选择1或3
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # 判断掩码类型：外部掩码 + 随机块 + 一半
        elif mask_type == 5:
            # 随机选择1-3的整数值
            mask_type = np.random.randint(1, 4)

        # 判断掩码类型：随机块
        if mask_type == 1:
            # 调用create_mask函数生成随机块掩码，并返回
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # 判断掩码类型：一半
        if mask_type == 2:
            # 随机选择左/右一半区域，并调用create_mask函数生成对应的掩码
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # 判断掩码类型：外部掩码
        if mask_type == 3:
            # 随机从已有的掩码数据集中选择一张，并将其调整到和原图像大小相同，再进行二值化处理
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255  # 由于插值需要进行阈值处理
            return mask

        # 判断掩码类型：测试模式（不随机加载掩码图像）
        if mask_type == 6:
            # 加载指定索引的掩码图像，并对其进行调整、灰度化和二值化处理
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    # 将numpy数组转换为PyTorch的张量
    def to_tensor(self, img):
        img = Image.fromarray(img) # 以numpy数组作为参数创建PIL图像
        img_t = F.to_tensor(img).float() # 将PIL图像转换为PyTorch的张量
        return img_t

    # 调整图像大小
    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2] # 获取图像的高度和宽度

        if centerCrop and imgh != imgw: # 如果需要进行中心裁剪且图像不是正方形
            # 中心裁剪
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width]) # 调整图像大小

        return img

    # 加载图像文件列表
    def load_flist(self, flist):
        if isinstance(flist, list): # 如果flist是列表，则直接返回
            return flist

        # flist: 图像文件路径、图像目录路径、文本文件flist路径
        if isinstance(flist, str):
            if os.path.isdir(flist): # 如果flist是目录路径，则获取目录下所有jpg和png文件的路径列表
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist): # 如果flist是文件路径，则尝试使用numpy.genfromtxt从文件中读取路径列表
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except: # 如果读取失败，则将文件路径包装成列表返回
                    return [flist]

        return []

    # 创建数据迭代器
    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self, # 数据集
                batch_size=batch_size, # 批大小
                drop_last=True # 如果数据集大小不能被批大小整除，丢弃最后一个不完整的批
            )

            for item in sample_loader: # 遍历数据集迭代器
                yield item # 返回一个批的数据

import torch
import torch.nn as nn


# 边缘精度评估模块，用于衡量边缘图像的准确性
class EdgeAccuracy(nn.Module):
    """
    衡量边缘图像的准确性
    """
    def __init__(self, threshold=0.5):
        super(EdgeAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        # 将输入和输出转换为二值边缘图像，使用阈值进行转换
        labels = (inputs > self.threshold)
        outputs = (outputs > self.threshold)

        # 计算相关和选定的边缘数量
        relevant = torch.sum(labels.float())
        selected = torch.sum(outputs.float())

        # 如果相关和选定的边缘数量都为0，则返回完美的精确度和召回率
        if relevant == 0 and selected == 0:
            return 1, 1

        # 计算真正的边缘数量、召回率和精确度
        true_positive = ((outputs == labels) * labels).float()
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        precision = torch.sum(true_positive) / (selected + 1e-8)

        return precision, recall


# 峰值信噪比评估模块，用于衡量两幅图像之间的信噪比
class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        # 根据最大像素值计算最大可能的PSNR值
        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()
        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        # 计算两幅图像之间的均方误差
        mse = torch.mean((a.float() - b.float()) ** 2)

        # 如果均方误差为0，则返回最大可能的PSNR值
        if mse == 0:
            return 0

        # 计算并返回PSNR值
        return self.max_val - 10 * torch.log(mse) / self.base10

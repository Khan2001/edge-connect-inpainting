import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import canny
import torchvision.transforms.functional as F
import torch.nn as nn


def create_dir(dir):
    """
    创建目录

    Args:
        dir: 目录名
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    """
    创建蒙版

    Args:
        width: 图像宽度
        height: 图像高度
        mask_width: 蒙版宽度
        mask_height: 蒙版高度
        x: 蒙版左上角横坐标（默认随机）
        y: 蒙版左上角纵坐标（默认随机）

    Returns:
        蒙版矩阵
    """
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask


def get_model_list(dirname, key_phase, key_model):
    """
    获取模型列表

    Args:
        dirname: 模型所在目录
        key_phase: 阶段名称，取值为'EdgeModel'或'InpaintModel'
        key_model: 模型名称，取值为'gen'或'dis'

    Returns:
        最新的模型文件名
    """
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key_phase in f and key_model in f and ".pth" in f]
    if len(gen_models) == 0:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def stitch_images(inputs, *outputs, img_per_row=2):
    """
    拼接图片

    Args:
        inputs: 输入图像列表
        outputs: 输出图像列表（可变参数）
        img_per_row: 每行显示的图像数量

    Returns:
        拼接后的图像
    """
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB',
                    (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imshow(img, title=''):
    """
    显示图片

    Args:
        img: 待显示的图像
        title: 窗口标题（默认为空）
    """
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    """
    保存图片

    Args:
        img: 待保存的图像
        path: 保存路径
    """
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)


def canny_edge(img, mask, sigma=2, training=False):
    # 在测试模式下，需要对掩码图像进行处理，以便正确处理被掩蔽的区域
    # 控制 Canny 边缘检测器不要检测掩码区域的边缘
    mask = None if training else (1 - mask / 255).astype(np.bool)

    # 如果 sigma=-1，则返回大小与输入相同的全零数组
    if sigma == -1:
        return np.zeros(img.shape).astype(np.float)

    # 如果 sigma=0，则随机选择一个 [1,4] 之间的整数作为阈值
    if sigma == 0:
        sigma = random.randint(1, 4)

    # 调用 canny 函数提取输入图像的边缘，并将结果转换为浮点数数组后返回
    return canny(img, sigma=sigma, mask=mask).astype(np.float)


def output_align(input, output):
    """
    在测试时，有时输出比不规则大小的输入少几个像素，这里是为了填充它们
    """
    # 如果输出大小与输入不同，则沿着宽度和高度方向进行填充
    if output.size() != input.size():
        diff_width = input.size(-1) - output.size(-1)
        diff_height = input.size(-2) - output.size(-2)
        m = nn.ReplicationPad2d((0, diff_width, 0, diff_height))
        output = m(output)

    # 返回大小与输入相同的输出
    return output


def to_tensor(img):
    # 将图像转换为 PyTorch 模型所需的张量格式
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    img_t = img_t.unsqueeze(0)

    # 返回扩展过维度的张量，大小为 [1,C,H,W]
    return img_t


class Progbar(object):
    """
    显示进度条

    参数：
        target: 总步数 None表示未知
        width: 进度条宽度
        verbose: 输出模式  0：静默 1：详细 2：半详细
        stateful_metrics: 可迭代的字符串名称的指标不应该在时间上平均的指标，将显示为-is，在显示之前所有其他的都将被平均化
        interval: 最小视觉进度更新间隔（以秒为单位）
    """
    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        # 判断是否可以使用动态显示（即在终端中更新进度条）
        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                    sys.stdout.isatty()) or
                                    'ipykernel' in sys.modules or
                                    'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """
        更新进度条
        
        参数：
            current：当前步骤的索引。
            values：一个元组列表，其中每个元组包含两个值：name 和 value_for_last_step。
                    如果 name 在 stateful_metrics 列表中，那么 value_for_last_step 将直接显示。否则，将显示该指标的历史平均值。
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

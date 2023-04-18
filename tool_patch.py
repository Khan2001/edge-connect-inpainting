"""
===============================================================================
使用 Edge-Connect 算法的交互式图像修补工具。

终端输入：
    python tool_patch.py --path <权重目录路径>
                        --edge（可选打开边缘窗口）


使用方法：
    终端输入命令后，将显示两个窗口，一个用于输入input，一个用于输出output。如果输入 `--edge` arg，将显示另一个边缘窗口edge。

    首先，在输入窗口中，使用鼠标左键涂黑缺失的部分，然后按下“n”修补图像（一次或多次）。
    在输入窗口中，用鼠标左键绘制黑色遮罩表示缺陷。
    在边缘窗口中，用鼠标左键绘制边缘，右键擦除边缘。
    使用“[”和“]”调整笔刷的大小，类似PhotoShop的操作。
    最后，按下“s”键保存输出。

    '[' - 使画笔厚度更小
    ']' - 使画笔厚度更大
    'n' - 要修补图像的黑色部分（只需使用输入图像）
    'e' - 要修补图像的黑色部分（使用输入图像并编辑边缘图像）
    'r' - 重置修补
    's' - 保存输出
    'q' - 退出
===============================================================================
"""

from __future__ import print_function

import argparse
import glob

from easygui import *
import numpy as np
import cv2 as cv
import sys
import os
import shutil
from src.config import Config
from main import main

BLUE = [255, 0, 0]  # rectangle color
RED = [0, 0, 255]  # PR BG
GREEN = [0, 255, 0]  # PR FG
BLACK = [0, 0, 0]  # sure BG
WHITE = [255, 255, 255]  # sure FG

DRAW_MASK = {'color': BLACK, 'val': 255}

radius = 3  # brush radius
drawing = False
drawing_edge_l = False
drawing_edge_r = False
value = DRAW_MASK
THICKNESS = -1  # 实心圆


def onmouse_input(event, x, y, flags, param):
    """
    只要鼠标在input窗口上移动(点击），此函数就会被回调执行
    """
    # 为方法体外的变量赋值，声明global
    global img, img2, drawing, value, mask, ix, iy, rect_over
    # print(x,y)

    # 如果鼠标左键按下，则开始绘制曲线
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        # 在img上画圆
        cv.circle(img, (x, y), radius, value['color'], THICKNESS, lineType=cv.LINE_AA)
        # 在mask上画圆
        cv.circle(mask, (x, y), radius, value['val'], THICKNESS, lineType=cv.LINE_AA)

    # 如果鼠标正在拖动并且已经开始了绘制，则在图像上绘制曲线
    elif drawing is True and event == cv.EVENT_MOUSEMOVE:
        # 在img上画圆
        cv.circle(img, (x, y), radius, value['color'], THICKNESS, lineType=cv.LINE_AA)
        # 在mask上画圆
        cv.circle(mask, (x, y), radius, value['val'], THICKNESS, lineType=cv.LINE_AA)

    # 如果鼠标左键松开，则停止绘制曲线
    elif drawing is True and event == cv.EVENT_LBUTTONUP:
        drawing = False
        # 在img上画圆
        cv.circle(img, (x, y), radius, value['color'], THICKNESS, lineType=cv.LINE_AA)
        # 在mask上画圆
        cv.circle(mask, (x, y), radius, value['val'], THICKNESS, lineType=cv.LINE_AA)



def onmouse_edge(event, x, y, flags, param):
    """
    只要鼠标在edge窗口上移动(点击），此函数就会被回调执行
    """
    # 为方法体外的变量赋值，声明global
    global img, img2, drawing_edge_l, drawing_edge_r, value, mask, ix, iy, edge
    # print(x, y)

    # 如果鼠标左键按下，则开始在edge上画曲线
    if event == cv.EVENT_LBUTTONDOWN:
        drawing_edge_l = True
        # 在edge上画点
        edge[y, x] = 255

    # 如果鼠标正在拖动并且已经开始了绘制，则在edge上绘制曲线
    elif drawing_edge_l is True and event == cv.EVENT_MOUSEMOVE:
        # 在edge上画点
        edge[y, x] = 255

    # 如果鼠标左键松开，则停止在edge上绘制曲线
    elif drawing_edge_l is True and event == cv.EVENT_LBUTTONUP:
        drawing_edge_l = False
        # 在edge上画点
        edge[y, x] = 255

    # 如果鼠标右键按下，则开始在edge上绘制圆形
    elif event == cv.EVENT_RBUTTONDOWN:
        drawing_edge_r = True
        # 在edge上画圆
        cv.circle(edge, (x, y), radius, BLACK, THICKNESS, lineType=cv.LINE_AA)

    # 如果鼠标正在拖动并且已经开始了绘制，则在edge上绘制圆形
    elif drawing_edge_r is True and event == cv.EVENT_MOUSEMOVE:
        cv.circle(edge, (x, y), radius, BLACK, THICKNESS, lineType=cv.LINE_AA)

    # 如果鼠标右键松开，则停止在edge上绘制圆形
    elif drawing_edge_r is True and event == cv.EVENT_RBUTTONUP:
        drawing_edge_r = False
        cv.circle(edge, (x, y), radius, BLACK, THICKNESS, lineType=cv.LINE_AA)



def check_load(args):
    """
    检查目录和权重文件。加载配置文件。
    """
    # 检查目录是否存在
    if not os.path.exists(args.path):
        raise NotADirectoryError('路径 <' + str(args.path) + '> 不存在！')

    # 查找边缘模型的权重文件
    edge_weight_files = list(glob.glob(os.path.join(args.path, 'EdgeModel_gen*.pth')))
    if len(edge_weight_files) == 0:
        raise FileNotFoundError('无法在路径 ' + args.path + ' 下找到权重文件 <EdgeModel_gen*.pth>！')
    
    # 查找修复模型的权重文件
    inpaint_weight_files = list(glob.glob(os.path.join(args.path, 'InpaintingModel_gen*.pth')))
    if len(inpaint_weight_files) == 0:
        raise FileNotFoundError('无法在路径 ' + args.path + ' 下找到权重文件 <InpaintingModel_gen*.pth>！')

    # 配置文件的路径
    config_path = os.path.join(args.path, 'config.yml')
    # 如果配置文件不存在，则将示例配置文件复制一份
    if not os.path.exists(config_path):
        shutil.copyfile('./config.yml.example', config_path)

    # 加载配置文件
    config = Config(config_path)

    return config



def load_model(config):
    """
    加载模型
    """
    model = main(mode=4, config=config)
    return model


def model_process(img, mask, edge=None):
    """
    处理图像
    :param img: 输入的图像，维度为3
    :param mask: 掩码，维度为2
    :return:
    首先将掩码二值化（即将所有非零像素都设置为 255），然后将 BGR 图像转换为 RGB 格式。
    如果传入了 edge 参数，则表示需要边缘保护修复，否则不需要。
    最后，使用 OpenCV 将结果图像从 RGB 转换为 BGR 格式，并返回修复后的图像和边缘。
    """
    # print(img.shape, mask.shape)
    mask[mask > 0] = 255
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if edge is None:
        result, edges = model.test_img_with_mask(img, mask)
    else:
        result, edges = model.test_img_with_mask(img, mask, edge_edit=edge)
    result = cv.cvtColor(result, cv.COLOR_RGB2BGR)
    return result, edges


# 定义主函数
if __name__ == '__main__':

    # 输出程序使用说明
    print(__doc__)

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path of model weights files <.pth>')
    parser.add_argument('-e', '--edge', action='store_true', help='open the edge edit window')
    args = parser.parse_args()

    # 检查模型权重文件路径是否存在，并加载模型
    config = check_load(args)
    model = load_model(config)

    # 让用户选择要处理的图片，并读取该图片，进行缩放
    image = fileopenbox(msg='Select a image', title='Select', filetypes=[['*.png', '*.jpg', '*.jpeg', 'Image Files']])
    if image is None:
        print('\nPlease select a image.')
        exit()
    else:
        print('Image selected: ' + image)

    img = cv.imread(image)

    # 如果图片尺寸太小，则进行放大操作
    if max(img.shape[0], img.shape[1]) < 256:
        img = cv.resize(img, dsize=(0, 0), fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
    img2 = img.copy()  # 备份原始图片
    mask = np.zeros(img.shape[:2], dtype=np.uint8)  # 掩膜初始化为背景
    output = np.zeros(img.shape, np.uint8)  # 输出图片

    # 创建输入输出窗口，并设置鼠标回调函数
    cv.namedWindow('output')
    cv.namedWindow('input')
    cv.setMouseCallback('input', onmouse_input)
    cv.moveWindow('input', img.shape[1] + 20, 90)  # 移动输入窗口位置

    # 如果开启了边界模式，则创建一个边界窗口，并设置鼠标回调函数
    if args.edge:
        edge = np.zeros(img.shape, np.uint8)
        cv.namedWindow('edge')
        cv.setMouseCallback('edge', onmouse_edge)
        cv.moveWindow('edge', img.shape[1] + 40, 90)

    # 进入图像分割循环处理
    while 1:

        # 显示输出和输入窗口，如果开启了边界模式，则同时显示边界窗口
        cv.imshow('output', output)
        cv.imshow('input', img)
        if args.edge:
            cv.imshow('edge', edge)

        # 等待200ms并接收按键事件
        k = cv.waitKey(200)

        # 根据按键事件执行相应操作
        if k == 27 or k == ord('q'):  # 按下“esc”或“q”，退出程序
            break

        elif k == ord('r'):  # 重置所有变量
            print("resetting \n")
            drawing = False
            value = DRAW_MASK
            img = img2.copy()
            mask = np.zeros(img.shape[:2], dtype=np.uint8)  # 掩膜初始化为背景
            output = np.zeros(img.shape, np.uint8)  # 输出图片
            if args.edge:
                edge = np.zeros(img.shape, np.uint8)
                drawing_edge_r = False
                drawing_edge_l = False

        elif k == ord('n'):  # 使用输入图片进行分割
            print("\nPatching using input image...")
            output, edge = model_process(img, mask)
            print("\nPatched!")
        elif k == ord('e') and args.edge:  # 使用输入图片和边界进行分割
            print("\nPatching using input and edge...")
            output, _ = model_process(img, mask, edge)
            print("\nPatched!")
        elif k == ord('['):
            radius = 1 if radius == 1 else radius - 1
            print('Brush thickness is', radius)
        elif k == ord(']'):
            radius += 1
            print('Brush thickness is', radius)
        elif k == ord('s'): # 保存输出结果到文件
            path = filesavebox('save', 'save the output.', default='patched_' + os.path.basename(image),
                                filetypes=[['*.jpg', 'jpg'], ['*.png', 'png']])
            if path:
                if not path.endswith('.jpg') and not path.endswith('.png'):
                    path = str(path) + '.png'
                cv.imwrite(path, output)
                print('Patched image is saved to', path)

    # 销毁所有窗口
    cv.destroyAllWindows()

import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.edge_connect import EdgeConnect


def main(mode=None, config=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
                    4: demo_patch,
    """

    # 如果mode等于4，则调用load_config_demo函数加载配置；否则，调用load_config函数加载配置。
    if mode == 4:
        config = load_config_demo(mode, config=config)
    else:
        config = load_config(mode)

    # 初始化环境
    if (config.DEVICE == 1 or config.DEVICE is None) and torch.cuda.is_available():
        # 设置CUDA_VISIBLE_DEVICES环境变量，指定使用哪些GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
        # 将DEVICE设置为cuda设备
        config.DEVICE = torch.device("cuda")
        # 启用cudnn自动调节器
        torch.backends.cudnn.benchmark = True  
    else:
        # 将DEVICE设置为cpu设备
        config.DEVICE = torch.device("cpu")
    print('DEVICE is:', config.DEVICE)

    # 将cv2运行线程数设置为1（防止与pytorch dataloader死锁）
    cv2.setNumThreads(0)

    # 初始化随机种子
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # 启用cudnn自动调节器
    torch.backends.cudnn.benchmark = True

    # 构建并初始化模型
    model = EdgeConnect(config)
    model.load()

    # 模型训练
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # 模型测试
    elif config.MODE == 2:
        print('\nstart testing...\n')
        with torch.no_grad():
            model.test()

    # 模型评估
    elif config.MODE == 3:
        print('\nstart eval...\n')
        with torch.no_grad():
            model.eval()

    # 模型演示
    elif config.MODE == 4:
        if config.DEBUG:
            config.print()
        print('model prepared.')
        return model



def load_config(mode=None):
    r"""加载模型配置

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    # 创建并初始化参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints',
                        help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4],
                        help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')

    # 如果mode等于2，则在参数解析器中添加一些额外的参数：输入图像、掩码图像、边缘图像和输出目录。
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
        parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
        parser.add_argument('--output', type=str, help='path to the output directory')

    # 解析命令行参数
    args = parser.parse_args()

    # 配置文件路径
    config_path = os.path.join(args.path, 'config.yml')

    # 如果检查点目录不存在，则创建该目录
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # 如果配置文件不存在，则复制配置文件示例
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # 加载配置文件
    config = Config(config_path)

    # 训练模式
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model 

        if config.SKIP_PHASE2 is None:
            config.SKIP_PHASE2 = 0
        if config.MODEL == 2 and config.SKIP_PHASE2 == 1:
            raise Exception("MODEL is 2, cannot skip phase2! trun config.SKIP_PHASE2 off or just use MODEL 3.")

    # 测试模式
    elif mode == 2:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3
        config.INPUT_SIZE = 0

        if args.input is not None:
            config.TEST_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.edge is not None:
            config.TEST_EDGE_FLIST = args.edge

        if args.output is not None:
            config.RESULTS = args.output

    # 评估模式
    elif mode == 3:
        config.MODE = 3
        config.MODEL = args.model if args.model is not None else 3

    return config



def load_config_demo(mode, config):
    r"""loads model config

    Args:
        mode (int): 4: demo_patch
    """
    print('load_config_demo----->')
    if mode == 4:
        config.MODE = 4
        config.MODEL = 3
        config.INPUT_SIZE = 0

    return config


if __name__ == "__main__":
    main()

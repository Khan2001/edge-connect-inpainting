import os
import yaml

# 定义 Config 类
class Config(dict):
    def __init__(self, config_path):
        # 打开配置文件并读取其内容
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            # 将 YAML 转换为字典
            self._dict = yaml.unsafe_load(self._yaml)
            # 设置 PATH 字段为给定路径所在的目录
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        # 如果字典中存在与请求的属性名相同的键，则返回该键对应的值
        if self._dict.get(name) is not None:
            return self._dict[name]

        # 如果 DEFAULT_CONFIG 字典中存在相应的默认值，则返回该值
        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]

        # 否则返回 None
        return None

    # 打印模型的配置信息
    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')

# 默认参数配置
DEFAULT_CONFIG = {
    'MODE': 1,                      # 1: train, 2: test, 3: eval 
                                    # 模式：1-训练，2-测试，3-评估
    'MODEL': 1,                     # 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model 
                                    # 模型类型：1-边缘模型，2-修复模型，3-边缘修复模型，4-联合模型
    'MASK': 3,                      # 1: random block, 2: half, 3: external, 4: (external, random block), 5: (external, random block, half)
                                    # 掩码类型：1-随机块，2-半遮挡，3-外部遮挡，4-（外部遮挡，随机块），5-（外部遮挡，随机块，半遮挡）
    'EDGE': 1,                      # 边缘检测算法类型：1-Canny，2-外部边缘
    'NMS': 1,                       # 非极大值抑制控制参数：0-不使用非极大值抑制，1-在外部边缘上应用非极大值抑制
    'SEED': 10,                     # 随机种子
    'GPU': [0],                     # GPU ID 列表
    'DEBUG': 0,                     # 是否启用调试模式
    'VERBOSE': 0,                   # 是否在输出控制台中启用详细模式

    'LR': 0.0001,                   # 学习率
    'D2G_LR': 0.1,                  # 判别器/生成器学习率比例
    'BETA1': 0.0,                   # Adam 优化器 beta1 参数
    'BETA2': 0.9,                   # Adam 优化器 beta2 参数
    'BATCH_SIZE': 8,                # 训练时的输入批次大小
    'INPUT_SIZE': 256,              # 训练时的输入图像尺寸，0 表示使用原始尺寸
    'SIGMA': 2,                     # Canny 边缘检测中使用的高斯滤波器的标准差（0 表示随机，-1 表示不使用边缘）
    'MAX_ITERS': 2e6,               # 训练模型的最大迭代次数

    'EDGE_THRESHOLD': 0.5,          # 边缘检测阈值
    'L1_LOSS_WEIGHT': 1,            # L1 损失函数的权重
    'FM_LOSS_WEIGHT': 10,           # 特征匹配损失的权重
    'STYLE_LOSS_WEIGHT': 1,         # 风格损失的权重
    'CONTENT_LOSS_WEIGHT': 1,       # 感知损失的权重
    'INPAINT_ADV_LOSS_WEIGHT': 0.01,# 对抗性损失的权重

    'GAN_LOSS': 'nsgan',            # 生成对抗网络的损失函数类型，可选值包括 'nsgan'、'lsgan' 和 'hinge'。
    'GAN_POOL_SIZE': 0,             # 伪图像池大小

    'SAVE_INTERVAL': 1000,          # 保存模型的间隔迭代数（0 表示不保存）
    'SAMPLE_INTERVAL': 1000,        # 采样图像的间隔迭代数（0 表示不采样）
    'SAMPLE_SIZE': 12,              # 采样图像的数量
    'EVAL_INTERVAL': 0,             # 模型评估的间隔迭代数（0 表示不评估）
    'LOG_INTERVAL': 10,             # 在训练状态输出到控制台的间隔迭代数（0 表示不输出）
}
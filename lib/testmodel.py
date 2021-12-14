from lib.utils import Bicubic
import torch
from lib.models import SRResNet,Generator


def get_test_model(device):
    models=[]
    models_name=[]

    ##SRRESNET模型载入
    # 模型参数
    large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
    small_kernel_size = 3   # 中间层卷积的核大小
    n_channels = 64         # 中间层通道数
    n_blocks = 16           # 残差模块数量
    scaling_factor = 4  # 放大比例

    # 预训练模型
    srresnet_checkpoint = "./results/srresnet.pth"
    # 加载模型SRResNet
    checkpoint = torch.load(srresnet_checkpoint)
    srresnet = SRResNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)
    srresnet = srresnet.to(device)
    srresnet.load_state_dict(checkpoint['model'])

    srresnet.eval()
    models.append(srresnet)
    models_name.append("srresnet")

    ##SRGAN_SAMPLE模型载入

    # 生成器模型参数(与SRResNet相同)
    large_kernel_size_g = 9   # 第一层卷积和最后一层卷积的核大小
    small_kernel_size_g = 3   # 中间层卷积的核大小
    n_channels_g = 64         # 中间层通道数
    n_blocks_g = 16           # 残差模块数量
    scaling_factor = 4         # 放大比例

    # 预训练模型
    srgan_checkpoint = "./results/srgan_sample.pth"

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srgan_checkpoint)
    generator = Generator(large_kernel_size=large_kernel_size_g,
                        small_kernel_size=small_kernel_size_g,
                        n_channels=n_channels_g,
                        n_blocks=n_blocks_g,
                        scaling_factor=scaling_factor)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'])

    generator.eval()

    models.append(generator)
    models_name.append("srgan_sample")

    ##SRGAN模型载入

    # 生成器模型参数(与SRResNet相同)
    large_kernel_size_g = 9   # 第一层卷积和最后一层卷积的核大小
    small_kernel_size_g = 3   # 中间层卷积的核大小
    n_channels_g = 64         # 中间层通道数
    n_blocks_g = 16           # 残差模块数量
    scaling_factor = 4         # 放大比例

    # 预训练模型
    srgan_checkpoint = "./results/srgan.pth"

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srgan_checkpoint)
    generator = Generator(large_kernel_size=large_kernel_size_g,
                        small_kernel_size=small_kernel_size_g,
                        n_channels=n_channels_g,
                        n_blocks=n_blocks_g,
                        scaling_factor=scaling_factor)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'])

    generator.eval()

    models.append(generator)
    models_name.append("srgan")

    ##SRRESNET_V2模型载入

    # 模型参数
    large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
    small_kernel_size = 3   # 中间层卷积的核大小
    n_channels = 64         # 中间层通道数
    n_blocks = 16           # 残差模块数量

    # 预训练模型
    checkpoint = "./results/srresnet_v2.pth"

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(checkpoint)
    srresnet_v2 = SRResNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)
    srresnet_v2 = srresnet_v2.to(device)
    srresnet_v2.load_state_dict(checkpoint['model'])

    srresnet_v2.eval()
    models.append(srresnet_v2)
    models_name.append("srresnet_v2")

    #srgan_v2_sample
    # 生成器模型参数(与SRResNet相同)
    large_kernel_size_g = 9   # 第一层卷积和最后一层卷积的核大小
    small_kernel_size_g = 3   # 中间层卷积的核大小
    n_channels_g = 64         # 中间层通道数
    n_blocks_g = 16           # 残差模块数量

    # 预训练模型
    srgan_checkpoint = "./results/srgan_v2_sample.pth"

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srgan_checkpoint)
    generator = Generator(large_kernel_size=large_kernel_size_g,
                        small_kernel_size=small_kernel_size_g,
                        n_channels=n_channels_g,
                        n_blocks=n_blocks_g,
                        scaling_factor=scaling_factor)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'])

    generator.eval()

    models.append(generator)
    models_name.append("srgan_v2_sample")

    #srgan_v2
    # 生成器模型参数(与SRResNet相同)
    large_kernel_size_g = 9   # 第一层卷积和最后一层卷积的核大小
    small_kernel_size_g = 3   # 中间层卷积的核大小
    n_channels_g = 64         # 中间层通道数
    n_blocks_g = 16           # 残差模块数量

    # 预训练模型
    srgan_checkpoint = "./results/srgan_v2.pth"

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srgan_checkpoint)
    generator = Generator(large_kernel_size=large_kernel_size_g,
                        small_kernel_size=small_kernel_size_g,
                        n_channels=n_channels_g,
                        n_blocks=n_blocks_g,
                        scaling_factor=scaling_factor)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'])

    generator.eval()

    models.append(generator)
    models_name.append("srgan_v2")


    #Bicubic模型载入

    scaling_factor=4


    bicubic_model=Bicubic(scaling_factor,device)
    models.append(bicubic_model)
    models_name.append("bicubic")

    return models,models_name
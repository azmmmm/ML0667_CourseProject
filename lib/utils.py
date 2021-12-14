#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from PIL import Image
import os
import json
import random
from torch.autograd._functions import tensor
import torchvision.transforms.functional as FT
import torch
import math
from PIL import ImageFilter
import io
import imageio
import numpy as np
from lib.niqe.niqe import niqe
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 常量
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456,
                                   0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224,
                                  0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor(
    [0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor(
    [0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


def create_data_lists(train_folders, test_folders, min_size, output_folder):
    """
    创建训练集和测试集列表文件.
        参数 train_folders: 训练文件夹集合; 各文件夹中的图像将被合并到一个图片列表文件里面
        参数 test_folders: 测试文件夹集合; 每个文件夹将形成一个图片列表文件
        参数 min_size: 图像宽、高的最小容忍值
        参数 output_folder: 最终生成的文件列表,json格式
    """
    print("\n正在创建文件列表... 请耐心等待.\n")
    train_images = list()
    for d in train_folders:
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
    print("训练集中共有 %d 张图像\n" % len(train_images))
    with open(os.path.join(output_folder, 'train_images.json'), 'w') as j:
        json.dump(train_images, j)

    for d in test_folders:
        test_images = list()
        test_name = d.split("/")[-1]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
        print("在测试集 %s 中共有 %d 张图像\n" %
              (test_name, len(test_images)))
        with open(os.path.join(output_folder, test_name + '_test_images.json'),'w') as j:
            json.dump(test_images, j)

    print("生成完毕。训练集和测试集文件列表已保存在 %s 下\n" % output_folder)


def convert_image(img, source, target):
    """
    转换图像格式.(该函数有严重的问题，最好不要改动)

    :参数 img: 输入图像
    :参数 source: 数据源格式, 共有3种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]' 
                   
    :参数 target: 数据目标格式, 共5种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]' 
                   (3) '[-1, 1]' 
                   (4) 'imagenet-norm' (由imagenet数据集的平均值和方差进行标准化)
                   (5) 'y-channel' (亮度通道Y，采用YCbCr颜色空间, 用于计算PSNR 和 SSIM)
    :返回: 转换后的图像
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'
                      }, "无法转换图像源格式 %s!" % source
    assert target in {
        'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm', 'y-channel'
    }, "无法转换图像目标格式t %s!" % target

    # 转换图像数据至 [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)   #把一个取值范围是[0,255]的PIL.Image 转换成形状为[C,H,W]的Tensor，取值范围是[0,1.0]

    elif source == '[0, 1]':
        pass  # 已经在[0, 1]范围内无需处理

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.
    
    # 从 [0, 1] 转换至目标格式
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # 无需处理

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],
                           rgb_weights) / 255. + 16.

    return img



class ImageTransforms(object):
    """
    图像变换.
    """

    def __init__(self, split, crop_size, scaling_factor, lr_img_type,
                 hr_img_type):
        """
        :参数 split: 'train' 或 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸
        :参数 scaling_factor: 放大比例
        :参数 lr_img_type: 低分辨率图像预处理方式
        :参数 hr_img_type: 高分辨率图像预处理方式
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        """
        对图像进行裁剪和下采样形成低分辨率图像
        :参数 img: 由PIL库读取的图像
        :返回: 特定形式的低分辨率和高分辨率图像
        """

        # 裁剪
        if self.split == 'train':
            # 从原图中随机裁剪一个子块作为高分辨率图像
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            img = img.crop((left, top, right, bottom))
        else:
            # 从图像中尽可能大的裁剪出能被放大比例整除的图像
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            img = img.crop((left, top, right, bottom))

        # 下采样（双三次差值）
        lr_img = img.resize((int(img.width / self.scaling_factor),
                                int(img.height / self.scaling_factor)),
                               Image.BICUBIC)

        # 安全性检查
        assert img.width == lr_img.width * self.scaling_factor and img.height == lr_img.height * self.scaling_factor

        # 转换图像
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        img = convert_image(img, source='pil', target=self.hr_img_type)

        return lr_img, img

#模拟jpeg压缩
def jpegBlur(im,q):
  buf = io.BytesIO()
  imageio.imwrite(buf,im,format='jpeg',quality=q)
  s = buf.getbuffer()
  im=imageio.imread(s,format='jpeg')
  return Image.fromarray(im)


def gasuss_noise(image:Image, mean=0, sigma=0.001):
    ''' 
        添加高斯噪声
        image:原始图像
        mean : 均值 
        sigma : 高斯模糊标准差， 0.01表示像素点在 1%范围内波动
    '''
    image=np.array(image)
    image = np.array(image/255, dtype=float)#将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
    noise = np.random.normal(mean, sigma, image.shape)#创建一个均值为mean，方差为var呈高斯分布的图像矩阵
    out = image + noise#将噪声和原始图像进行相加得到加噪后的图像
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)#clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
    out = np.uint8(out*255)#解除归一化，乘以255将加噪后的图像的像素值恢复
    #cv.imshow("gasuss", out)
    noise = noise*255
    return Image.fromarray(out)

class ImageTransforms_v2(object):
    """
    图像变换.
    """

    def __init__(self, split, crop_size, scaling_factor, lr_img_type,
                 hr_img_type,gasuss_noise_sigma=0.001):
        """
        :参数 split: 'train' 或 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸
        :参数 scaling_factor: 放大比例
        :参数 lr_img_type: 低分辨率图像预处理方式
        :参数 hr_img_type: 高分辨率图像预处理方式
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.gasuss_noise_sigma=gasuss_noise_sigma
        assert self.split in {'train', 'test'}

    def __call__(self, img):
        """
        对图像进行裁剪和下采样形成低分辨率图像
        :参数 img: 由PIL库读取的图像
        :返回: 特定形式的低分辨率和高分辨率图像
        """

        # 裁剪
        if self.split == 'train':
            # 从原图中随机裁剪一个子块作为高分辨率图像
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            img = img.crop((left, top, right, bottom))
        else:
            # 从图像中尽可能大的裁剪出能被放大比例整除的图像
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            img = img.crop((left, top, right, bottom))

        # ----------------------- degradation process ----------------------- #
        #高斯模糊
        lr_img=img.filter(ImageFilter.GaussianBlur())


        # 下采样（双三次差值）
        lr_img = lr_img.resize((int(img.width / self.scaling_factor),
                                int(img.height / self.scaling_factor)),
                               Image.BICUBIC)
        #高斯噪音
        lr_img=gasuss_noise(lr_img,mean=0,sigma=self.gasuss_noise_sigma)

        #模拟jpeg压缩
        lr_img=jpegBlur(lr_img,random.randint(70,90))

        # 安全性检查
        assert img.width == lr_img.width * self.scaling_factor and img.height == lr_img.height * self.scaling_factor

        # 转换图像
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        img = convert_image(img, source='pil', target=self.hr_img_type)

        return lr_img, img

class AverageMeter(object):
    """
    跟踪记录类，用于统计一组数据的平均值、累加和、数据个数.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    丢弃梯度防止计算过程中梯度爆炸.

    :参数 optimizer: 优化器，其梯度将被截断
    :参数 grad_clip: 截断值
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state, filename):
    """
    保存训练结果.

    :参数 state: 逐项预保存内容
    """

    torch.save(state, filename)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    调整学习率.

    :参数 optimizer: 需要调整的优化器
    :参数 shrink_factor: 调整因子，范围在 (0, 1) 之间，用于乘上原学习率.
    """

    print("\n调整学习率.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("新的学习率为 %f\n" % (optimizer.param_groups[0]['lr'], ))

class Bicubic():
    """
    #标准双三次插值上采样
    """
    def __init__(self, scaling_factor,device):
        """
        :参数 scaling_factor: 放大比例
        """
        self.scaling_factor = scaling_factor
        self.device=device

    def __call__(self, img):
        '''
        img:# # (1, 3, w, h ), imagenet-normed
        return: img_tensor #(1,3, w*scaling_factor, h*scaling_factor), in [-1, 1] 
        '''
        img_01=img*imagenet_std_cuda+imagenet_mean_cuda # (1, 3, w, h ), [0, 1]
        img_01=img_01[0] ## (3, w, h ), [0, 1]
        
        img_PIL=convert_image(img_01,source='[0, 1]',target='pil')
        #print(img_PIL.width)
        img_PIL=img_PIL.resize((int(img_PIL.width * self.scaling_factor),int(img_PIL.height * self.scaling_factor)),Image.BICUBIC)
        
        img_tensor=convert_image(img_PIL,source='pil',target='[-1, 1]')# (3, w*scaling_factor, h*scaling_factor), in [-1, 1] 
        img_tensor=torch.tensor([img_tensor.numpy()]).to(self.device) #(1,3, w*scaling_factor, h*scaling_factor), in [-1, 1] 
        return img_tensor

def get_NIQE_from_PIL(img:Image,device):
    
    img_tensor=convert_image(img,source='pil',target='[-1, 1]')# (3, w, h), in [-1, 1] 
    img_tensor=torch.tensor([img_tensor.numpy()]).to(device) #(1,3, w, h), in [-1, 1] 

    sr_imgs=img_tensor  # (1, 3, w, h), in [-1, 1]                
    #计算NIQE
    sr_imgs_PIL=convert_image(sr_imgs.squeeze(0).cpu().detach(),source='[-1, 1]',target='pil')
    img = np.array(sr_imgs_PIL.convert('LA'))[:,:,0] # ref
    niqe_score=np.float64(niqe(img))
    return niqe_score
    
 

def degradation(img:Image,scaling_factor,gasuss_noise_sigma=0.001,radius=2):
        #高斯模糊
        lr_img=img.filter(ImageFilter.GaussianBlur(radius=radius))


        # 下采样（双三次差值）
        lr_img = lr_img.resize((int(img.width / scaling_factor),
                                int(img.height / scaling_factor)),
                               Image.BICUBIC)
        #高斯噪音
        lr_img=gasuss_noise(lr_img,mean=0,sigma=gasuss_noise_sigma)

        #模拟jpeg压缩
        lr_img=jpegBlur(lr_img,random.randint(70,90))
        return lr_img

#超分处理
def super_revolution(model:list,model_name,imgDir,input,output,device):
    # 加载图像
    img = Image.open(imgDir+input, mode='r')
    img = img.convert('RGB')
 
    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)
 

    # 转移数据至设备
    lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed
 
    # 模型推理
    score=pd.DataFrame(columns=['NIQE'])
    for i in range(0,len(model)):
        with torch.no_grad():
            sr_img = model[i](lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]   
            sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
            sr_img.save(imgDir+output+'_'+model_name[i]+'.jpg')
            
            #sr_imgs_PIL=convert_image(sr_imgs.squeeze(0).cpu().detach(),source='[-1, 1]',target='pil')
            
            img = np.array(sr_img.convert('LA'))[:,:,0] # ref
            niqe_score=np.float64(niqe(img))

        score=score.append(pd.DataFrame([niqe_score],index=[model_name[i]],columns=['NIQE']))
        del img,sr_img
    return(score)
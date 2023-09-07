import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image





# dataset block 数据集模块
# 三段式 1.初始化 2.获取数据__getitem__ 3.获取数据长度
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        # image path 读入图片路径
        self.files = glob.glob(os.path.join(root) + "/*/*.*")
        self.name = list(set([os.path.basename(os.path.dirname(file_name)) for file_name in self.files]))
        self.name.sort()
        # random image 打乱图片顺序/可以省略
        random.shuffle(self.files)

    def __getitem__(self, index):

        name = self.name
        # open image 打开图片
        img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        # change label 转换标签为 one-hot
        label = name.index(os.path.basename(os.path.dirname(self.files[index % len(self.files)])))
        
        if self.transform is not None:
            img = self.transform(img)

        return img , label

    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    
    dataset = ImageDataset('data')
    print(dataset.name)
    
import os

import PIL.Image as Image
import torch
import torch.utils.data as data
from torchvision.transforms import transforms

# data.Dataset:
# 所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)


class GGNDataset(data.Dataset):

    def __init__(self, path, transform=None):
        # os.listdir(path)返回指定路径下的文件和文件夹列表。"/"是真除法,"//"对结果取整
        n = len(os.listdir(path)) // 6
        imgs = []
        for i in range(n):
            # 注意这里的格式化字符串用法
            img1 = os.path.join(path, '%d.1.ct.35.jpg' % i)
            img2 = os.path.join(path, '%d.2.ct.35.jpg' % i)
            img3 = os.path.join(path, '%d.3.ct.35.jpg' % i)
            img4 = os.path.join(path, '%d.2.ct.65.jpg' % i)
            mask = os.path.join(path, '%d.mask.35.jpg' % i)
            imgs.append([img1, img2, img3, img4, mask])

        self.imgs = imgs
        self.data_transform = transform

    def __getitem__(self, index):
        img1_path, img2_path, img3_path, img4_path, mask_path = self.imgs[index]
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        img3 = Image.open(img3_path)
        img4 = Image.open(img4_path)
        mask = Image.open(mask_path).convert('1')
        if self.data_transform is not None:
            # 多张图合成一张多通道图
            img_3d = torch.cat((self.data_transform(img1), self.data_transform(img2), self.data_transform(img3)), 0)
            img_2d = torch.cat((self.data_transform(img2), self.data_transform(img4)), 0)
            mask = transforms.ToTensor()(mask)

        # 返回的是tensor
        return img_3d, img_2d, mask

    def __len__(self):
        return len(self.imgs)

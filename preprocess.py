import os

import nrrd
import numpy as np
import pandas as pd
import pydicom
from imageio import imwrite
from matplotlib import pyplot as plt
from PIL import Image

PATH = "D:\Dataset"
tables = pd.read_csv(r"coordinates\GGN_original.csv")


def maxmin_normalize(x):
    min = np.min(x)
    max = np.max(x)
    x = (x - min) / (max - min)
    return x


def crop(array, x, y, scale):
    top = round(scale/2)
    bottom = scale - top
    return array[round(y - bottom):round(y + top), round(x - bottom):round(x + top)]


def show():
    for ct_id, x, y, _, _, _, mask_index, z_trans, _ in tables.iloc:
        # 读入CT源文件及mask
        mask, _ = nrrd.read(os.path.join(PATH, ct_id, '{}.nrrd\{}.nrrd'.format(mask_index, mask_index)))
        ct_image = pydicom.read_file(os.path.join(PATH, ct_id, '{}.dcm'.format(z_trans + 1))).pixel_array

        # CT图像归一化，mask横纵坐标调换
        ct_image[ct_image > 1200] = 1200
        ct_image[ct_image < -600] = -600
        nodule_mask = np.transpose(mask[:, :, z_trans])

        # 遮罩结果展示
        plt.figure('CT_ID:{} Mask Index: {} Slice:{}'.format(ct_id, mask_index, z_trans))
        plt.subplot(2, 3, 1)
        plt.imshow(ct_image, "gray")
        plt.subplot(2, 3, 2)
        plt.imshow(1000 * nodule_mask, "gray")
        plt.subplot(2, 3, 3)
        plt.imshow(2000 * nodule_mask + ct_image, "gray")

        # 进行裁切
        mask_crop = crop(nodule_mask, x, y, 35)
        nodule_crop = crop(ct_image, x, y, 35)

        # 裁切结果展示
        plt.subplot(2, 3, 4)
        plt.imshow(mask_crop, "gray")
        plt.subplot(2, 3, 5)
        plt.imshow(nodule_crop, "gray")
        plt.show()


def save_pictures():
    count_train = 0
    count_val = 0
    for ct_id, x, y, _, _, _, mask_index, z_trans, types in tables.iloc:
        # 判断训练集和测试集
        count = count_train if types == 'train' else count_val

        # 读入mask，横纵坐标调换
        mask, _ = nrrd.read(os.path.join(PATH, ct_id, '{}.nrrd\{}.nrrd'.format(mask_index, mask_index)))
        nodule_mask = np.transpose(mask[:, :, z_trans])

        # 裁切35×35及65×65的mask
        mask_crop = crop(nodule_mask, x, y, 35)
        imwrite('dataset//%s//%d.mask.35.jpg' % (types, count), mask_crop)
        mask_crop = crop(nodule_mask, x, y, 65)

        # 下采样到35×35
        mask_crop = np.array(Image.fromarray(mask_crop).resize((35, 35), resample=0))
        imwrite('dataset//%s//%d.mask.65.jpg' % (types, count), mask_crop)

        # 裁切上下三张CT图
        for i in range(-1, 2):
            # 读入CT图并归一化
            ct_image = pydicom.read_file(os.path.join(PATH, ct_id, '%d.dcm' % (z_trans + 1 + i))).pixel_array
            ct_image[ct_image > 1200] = 1200
            ct_image[ct_image < -600] = -600

            # 裁切 35×35 CT图
            nodule_crop = crop(ct_image, x, y, 35)
            imwrite('dataset//%s//%d.%d.ct.35.jpg' % (types, count, i+2), maxmin_normalize(nodule_crop)*255)

            # 裁切 65×65 CT图
            if i == 0:
                nodule_crop = crop(ct_image, x, y, 65)
                # 下采样到35×35
                nodule_crop = np.array(Image.fromarray(nodule_crop).resize((35, 35), resample=0))
                imwrite('dataset//%s//%d.%d.ct.65.jpg' % (types, count, i+2), maxmin_normalize(nodule_crop)*255)

        print("已处理%d个结节" % (count_train+count_val+1))
        if types == 'train':
            count_train += 1
        else:
            count_val += 1


if __name__ == '__main__':
    save_pictures()

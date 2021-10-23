import os

import nrrd
import numpy as np
import pandas as pd
import pydicom
from matplotlib import pyplot as plt
from imageio import imwrite

PATH = "D:\Dataset"
tables = pd.read_csv(r"nodules\GGN_original.csv")


def show():
    for ct_id, x, y, z, diam, slice_num, mask_index, z_trans in tables.iloc:
        # 读入CT源文件及mask
        mask, mask_options = nrrd.read(os.path.join(PATH, ct_id, '{}.nrrd\{}.nrrd'.format(mask_index, mask_index)))
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
        mask_crop = nodule_mask[round(y - 17):round(y + 18), round(x - 17):round(x + 18)]
        nodule_crop = ct_image[round(y - 17):round(y + 18), round(x - 17):round(x + 18)]

        # 裁切结果展示
        plt.subplot(2, 3, 4)
        plt.imshow(mask_crop, "gray")
        plt.subplot(2, 3, 5)
        plt.imshow(nodule_crop, "gray")
        plt.show()


def maxmin_normalize(x):
    min = np.min(x)
    max = np.max(x)
    x = (x - min) / (max - min)
    return x


def save_pictures():
    count = 0
    for ct_id, x, y, z, diam, slice_num, mask_index, z_trans in tables.iloc:
        # 读入mask并预处理
        mask, mask_options = nrrd.read(os.path.join(PATH, ct_id, '{}.nrrd\{}.nrrd'.format(mask_index, mask_index)))
        nodule_mask = np.transpose(mask[:, :, z_trans])

        # 裁切35×35及65×65的mask
        mask_crop = nodule_mask[round(y - 17):round(y + 18), round(x - 17):round(x + 18)]
        imwrite('.\\dataset\\{}.mask.{}.jpg'.format(count, mask_crop.shape[0]), mask_crop)
        mask_crop = nodule_mask[round(y - 32):round(y + 33), round(x - 32):round(x + 33)]
        imwrite('.\\dataset\\{}.mask.{}.jpg'.format(count, mask_crop.shape[0]), mask_crop)

        # 裁切上下三张CT图
        for i in range(-1, 2):
            ct_image = pydicom.read_file(os.path.join(PATH, ct_id, '{}.dcm'.format(z_trans + 1 + i))).pixel_array
            # CT图像归一化，mask横纵坐标调换
            ct_image[ct_image > 1200] = 1200
            ct_image[ct_image < -600] = -600

            # 裁切 35×35 CT图
            nodule_crop = ct_image[round(y - 17):round(y + 18), round(x - 17):round(x + 18)]
            imwrite('.\\dataset\\{}.{}.ct.{}.jpg'.format(count, i+2, nodule_crop.shape[0]), maxmin_normalize(nodule_crop)*255)

            # 裁切 65×65 CT图
            if i == 0:
                nodule_crop = ct_image[round(y - 32):round(y + 33), round(x - 32):round(x + 33)]
                imwrite('.\\dataset\\{}.{}.ct.{}.jpg'.format(count, i+2, nodule_crop.shape[0]), maxmin_normalize(nodule_crop)*255)
        print("已处理{}个结节".format(count))
        count = count + 1


if __name__ == '__main__':
    save_pictures()

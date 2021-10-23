import os

import nrrd
import numpy as np
import pandas as pd
import pydicom
from matplotlib import pyplot as plt

PATH = "D:\Dataset"
tables = pd.read_csv(r"nodules\GGN_original.csv")
id_table = tables.iloc[:, 0]

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
    mask_crop = nodule_mask[round(x):round(x + 35), round(y - 35):round(y)]
    nodule_crop = ct_image[round(x):round(x + 35), round(y - 35):round(y)]

    # 裁切结果展示
    plt.subplot(2, 3, 4)
    plt.imshow(mask_crop, "gray")
    plt.subplot(2, 3, 5)
    plt.imshow(nodule_crop, "gray")
    plt.show()

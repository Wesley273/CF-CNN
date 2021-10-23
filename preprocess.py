import pandas as pd
import os
import numpy as np
import nrrd
import pydicom
from matplotlib import pyplot as plt

PATH = "D:\Dataset"
tables = pd.read_csv(r"nodules\GGN_original.csv")

for ct_id, x, y, z, diam in tables.iloc:
    # 读入文件
    try:
        mask, mask_options = nrrd.read(os.path.join(PATH, ct_id, 'LAC_2.nrrd\LAC_2.nrrd'))
    except Exception as e:
        pass
    slice_index = round(mask.shape[2] - z)
    ct_image = pydicom.read_file(os.path.join(PATH, ct_id, '{}.dcm'.format(slice_index + 1))).pixel_array

    # CT图像归一化，mask横纵坐标调换
    ct_image[ct_image > 1200] = 1200
    ct_image[ct_image < -600] = -600
    nodule_mask = np.transpose(mask[:, :, slice_index])

    # 遮罩结果
    plt.subplot(1, 3, 1)
    plt.imshow(ct_image, "gray")
    plt.subplot(1, 3, 2)
    plt.imshow(1000 * nodule_mask, "gray")
    plt.subplot(1, 3, 3)
    plt.imshow(2000 * nodule_mask + ct_image, "gray")
    plt.show()

    # 进行裁切
    mask_crop = nodule_mask[round(x):round(x + 35), round(y - 35):round(y)]
    nodule_crop = ct_image[round(x):round(x + 35), round(y - 35):round(y)]

    # 裁切结果展示
    plt.subplot(1, 2, 1)
    plt.imshow(mask_crop, "gray")
    plt.subplot(1, 2, 2)
    plt.imshow(nodule_crop, "gray")
    plt.show()

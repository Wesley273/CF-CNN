import pydicom
from matplotlib import pyplot as plt
import nrrd
import numpy as np

#结节位置 x:166.7001 y:122.6418  z:185.625

#读入数据和标注
slices = 50
data = pydicom.read_file(r".\testDataset\{}.dcm".format(slices))
mask_path = r".\testDataset\LAC_1.nrrd"
mask, mask_options = nrrd.read(mask_path)

#显示CT图片
plt.imshow(data.pixel_array, "gray")
plt.show()

#显示mask
plt.imshow(np.transpose(mask[:, :, slices]), "gray")
plt.show()

#显示遮罩结果
plt.imshow(np.transpose(2000 * mask[:, :, slices]) + data.pixel_array, "gray")
plt.show()
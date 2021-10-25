from dataset import GGNDataset
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import GGNDataset
from cfcnn import Net

data_transform = transforms.Compose([
    transforms.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    # torchvision.transforms.Normalize(mean, std, inplace=False)
    transforms.Normalize([0.5], [0.5])
])

liver_dataset = GGNDataset("dataset", transform=data_transform)
print(liver_dataset[0])

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from cfcnn import Net
from dataset import GGNDataset

data_transform = transforms.Compose([
    transforms.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    # torchvision.transforms.Normalize(mean, std, inplace=False)
    transforms.Normalize([0.5], [0.5])
])

if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load(r"weight\weights_19.pth", map_location='cpu'))
    GGN_dataset = GGNDataset("dataset", transform=data_transform)
    dataloaders = DataLoader(GGN_dataset)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x, y, _ in dataloaders:
            output = model(x, y)
            output = torch.squeeze(output).numpy()
            plt.imshow(output)
            plt.pause(0.01)
        plt.show()

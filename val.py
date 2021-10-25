from PIL.Image import Image
import torch
import matplotlib.pyplot as plt
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


def show_result(model, dataloader):
    plt.ion()
    with torch.no_grad():
        for x, y, _ in dataloader:
            output = model(x, y)
            output = torch.squeeze(output).numpy()
            plt.imshow(output)
            plt.pause(0.01)
        plt.show()


def get_IoU(model, dataloader):
    for x, y, mask in dataloader:
        mask = torch.squeeze(mask).numpy()
        output = model(x, y)
        output = torch.squeeze(output).detach().numpy()
        output = output > 0.5
        mask = mask > 0.5
        intersection = (output & mask).sum()
        union = (output | mask).sum()
        print((intersection) / union)


if __name__ == '__main__':
    # 加载模型
    model = Net()
    model.load_state_dict(torch.load(r"weight\weights_a_19.pth", map_location='cpu'))
    GGN_dataset = GGNDataset("dataset", transform=data_transform)
    dataloader = DataLoader(GGN_dataset)
    model.eval()
    #show_result(model, dataloader)
    get_IoU(model, dataloader)

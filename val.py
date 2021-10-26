import matplotlib.pyplot as plt
import torch
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
    with torch.no_grad():
        for img_3d, img_2d, _ in dataloader:
            output = model(img_3d, img_2d)
            output = torch.squeeze(output).numpy()
            plt.imshow(output, 'gray')
            plt.show()


def get_IoU(model, dataloader):
    for img_3d, img_2d, mask in dataloader:
        mask = torch.squeeze(mask).numpy()
        output = model(img_3d, img_2d)
        output = torch.squeeze(output).detach().numpy()
        output = output > 0.5
        mask = mask > 0.5
        intersection = (output & mask).sum()
        union = (output | mask).sum()
        print((intersection) / union)


if __name__ == '__main__':
    # 加载模型
    model = Net()
    model.load_state_dict(torch.load(r"weight/weights_a_1999.pth", map_location='cpu'))
    GGN_dataset = GGNDataset(r"dataset/val", transform=data_transform)
    dataloader = DataLoader(GGN_dataset)
    model.eval()
    show_result(model, dataloader)
    get_IoU(model, dataloader)

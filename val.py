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
        for img_3d, img_2d, mask in dataloader:
            output = model(img_3d, img_2d)
            output = torch.squeeze(output).numpy()
            mask = torch.squeeze(mask).numpy()
            plt.subplot(1, 2, 1)
            plt.imshow(output, 'gray')
            plt.title('Prediction IoU=%.03f' % get_IoU(mask, output, 0.5))
            plt.subplot(1, 2, 2)
            plt.title('Ground Truth')
            plt.imshow(mask, 'gray')
            plt.show()


def get_IoU(x, y, threshold):
    x = x > threshold
    y = y > threshold
    intersection = (x & y).sum()
    union = (x | y).sum()
    return (intersection) / union


def print_IoU(model, dataloader):
    for img_3d, img_2d, mask in dataloader:
        output = model(img_3d, img_2d)
        output = torch.squeeze(output).detach().numpy()
        mask = torch.squeeze(mask).numpy()
        print(get_IoU(mask, output, 0.5))


if __name__ == '__main__':
    # 加载模型
    model = Net()
    model.load_state_dict(torch.load(r"weight/weights_a_1999.pth", map_location='cpu'))
    GGN_dataset = GGNDataset(r"dataset/val", transform=data_transform)
    dataloader = DataLoader(GGN_dataset)
    model.eval()
    print_IoU(model, dataloader)
    show_result(model, dataloader)

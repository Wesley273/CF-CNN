from cfcnn import Net
import torch
import PIL.Image as Image
from torchvision.transforms import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_transform = transforms.Compose([
    transforms.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    # torchvision.transforms.Normalize(mean, std, inplace=False)
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
if __name__ == '__main__':
    model = Net(3, 1)
    test = Image.open(r"test.jpg")
    test = x_transform(test)
    test = test.unsqueeze(0)
    model.eval()
    torch.no_grad()
    model(test)

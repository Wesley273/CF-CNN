import CFNet
import torch
import PIL.Image as Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    model = CFNet(3, 1)
    test = Image.open(r"test.jpg")
    test.size()
    model(test)

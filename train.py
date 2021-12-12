import copy

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from cfcnn import Net
from dataset import GGNDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_transform = transforms.Compose([
    transforms.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    # torchvision.transforms.Normalize(mean, std, inplace=False)
    transforms.Normalize([0.5], [0.5])
])

def get_test_loss(model,epoch):
    # 测试集loss计算
        test_loss = 0
        TEST_dataset = GGNDataset(r"dataset/val", transform=data_transform)
        dataloader_test = DataLoader(TEST_dataset)
        model.eval()
        for img_3d, img_2d, mask in dataloader_test:
            output = model(img_3d.to(device), img_2d.to(device))
            test_loss += criterion(output, mask.to(device)).item()
        print("epoch %d test loss:%0.3f" % (epoch, test_loss))

# 训练模型
def train_model(model, criterion, optimizer, dataloader, num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataloader.dataset)
        epoch_loss = 0
        step = 0
        for img_3d, img_2d, mask in dataloader:
            # 每个bacth都要将梯度(dw,db,...)清零
            optimizer.zero_grad()
            inputs_3d = img_3d.to(device)
            inputs_2d = img_2d.to(device)
            masks = mask.to(device)
            # 前向传播
            outputs = model(inputs_3d, inputs_2d)
            # 计算损失
            loss = criterion(outputs, masks)
            # 梯度下降,计算出梯度
            loss.backward()
            # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            optimizer.step()
            # loss.item()是为了取得一个元素张量的数值
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataloader.batch_size, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        get_test_loss(copy.deepcopy(model),epoch)
    # 保存模型参数
    torch.save(model.state_dict(), r'weight/weights_%d.pth' % epoch)
    return model


if __name__ == '__main__':
    model = Net().to(device)
    # 损失函数
    criterion = torch.nn.BCELoss()
    # 梯度下降
    optimizer = optim.Adam(model.parameters())
    # 加载数据集
    GGN_dataset = GGNDataset(r"dataset/train", transform=data_transform)
    # 设置DataLoader
    dataloader = DataLoader(GGN_dataset, batch_size=8, shuffle=True, num_workers=4)
    # 开始训练
    train_model(model, criterion, optimizer, dataloader, num_epochs=2000)

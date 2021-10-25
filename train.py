import PIL.Image as Image
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import NoduleDataset
from cfcnn import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_transform = transforms.Compose([
    transforms.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    # torchvision.transforms.Normalize(mean, std, inplace=False)
    transforms.Normalize([0.5], [0.5])
])


# 训练模型
def train_model(model, criterion, optimizer, dataloader, num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataloader.dataset)
        epoch_loss = 0
        step = 0  # minibatch数
        for x, y in dataloader:  # 分100次遍历数据集，每次遍历batch_size=4
            optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 梯度下降,计算出梯度
            optimizer.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += loss.item()  # loss.item()是为了取得一个元素张量的数值
            step += 1
            print("%d/%d,train_loss:%0.3f" %
                  (step, dataset_size // dataloader.batch_size, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    # 保存模型参数
    torch.save(model.state_dict(), '.\\weight\\weights_%d.pth' % epoch)
    return model


if __name__ == '__main__':
    model = Net().to(device)
    batch_size = 1
    # 损失函数
    criterion = torch.nn.BCELoss()
    # 梯度下降
    # model.parameters():Returns an iterator over module parameters
    optimizer = optim.Adam(model.parameters())
    # 加载数据集
    liver_dataset = NoduleDataset("dataset", transform=data_transform, target_transform=data_transform)
    dataloader = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # 开始训练
    train_model(model, criterion, optimizer, dataloader)

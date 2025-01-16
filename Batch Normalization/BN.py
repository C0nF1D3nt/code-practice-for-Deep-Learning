import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils import data
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    # 训练模式使用小批次的平均值和方差，预测模式使用训练模式中计算的滑动平均值来计算
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 在mean函数中，dim是多少，该维度就会压缩为1
            # 使用全连接层的情况，对不同batch求平均，mean的维度为(1, feature_len)
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，只保留通道那一维
            # 即每个通道分别进行标准化
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 进行标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层(B L)，4表示卷积层(B C H W)
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 需要进行训练的参数
        # 在全连接层情况下，数量与特征长度相同
        # 在卷积层情况下，数量与通道数相同
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 滑动平均值，在训练过程中迭代，最终用于预测过程
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5),
    # nn.BatchNorm2d(6),
    BatchNorm(6, num_dims=4),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    # nn.BatchNorm2d(16),
    BatchNorm(16, num_dims=4),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*4*4, 120),
    # nn.BatchNorm1d(120),
    BatchNorm(120, num_dims=2),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    # nn.BatchNorm1d(84),
    BatchNorm(84, num_dims=2),
    nn.Sigmoid(),
    nn.Linear(84, 10)
).to(device)

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

trans = transforms.Compose([
    transforms.ToTensor(),
])

train_data = torchvision.datasets.FashionMNIST('../data', train=True, download=True, transform=trans)
test_data = torchvision.datasets.FashionMNIST('../data', train=False, download=True, transform=trans)

batch_size = 256
train_iter = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

n_epochs = 10
lr = 1.0

loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

loss_list = []
train_acc_list = []
test_acc_list = []
for epoch in range(n_epochs):
    net.train()
    total_loss = 0.0
    train_acc = 0.0
    for i, (X, y) in enumerate(train_iter):
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss = loss_f(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_loss += loss.item() * X.shape[0]
            train_acc += (torch.argmax(y_hat, dim=1).reshape(y.shape) == y).sum()
    total_loss /= len(train_data)
    loss_list.append(total_loss)
    train_acc /= len(train_data)
    print(f'epoch {epoch + 1}: loss {total_loss}, train_acc {train_acc}')

    # test
    with torch.no_grad():
        net.eval()
        test_acc = 0.0
        for i, (X, y) in enumerate(test_iter):
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            test_acc += (torch.argmax(y_hat, dim=1).reshape(y.shape) == y).sum()
        test_acc /= len(test_data)
        print(f'test_acc {test_acc}')


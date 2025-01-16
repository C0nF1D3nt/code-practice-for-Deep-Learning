import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils import data
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # path1
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # path2
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # path3
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # path4
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.concat((p1, p2, p3, p4), dim=1)

class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.model = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            nn.Linear(1024, 10)
        )


    def forward(self, x):
        return self.model(x)

net = GoogleNet().to(device)

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(96)
])

train_data = torchvision.datasets.FashionMNIST('../data', train=True, download=True, transform=trans)
test_data = torchvision.datasets.FashionMNIST('../data', train=False, download=True, transform=trans)

batch_size = 128
train_iter = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

n_epochs = 10
lr = 0.1

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



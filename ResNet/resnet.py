import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils import data
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)

def resnet_block(input_channels, output_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, output_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(output_channels, output_channels))
    return blk

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.model = nn.Sequential(
            self.b1,
            self.b2,
            self.b3,
            self.b4,
            self.b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.model(x)

net = ResNet().to(device)

# X = torch.rand(size=(1, 1, 224, 224))
# for layer in net.model:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape:\t', X.shape)

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

batch_size = 256
train_iter = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

n_epochs = 10
lr = 0.05

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


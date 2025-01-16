import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils import data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(p=0.5),
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
).to(device)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
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



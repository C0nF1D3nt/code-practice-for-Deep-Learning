import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils import data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10)
    )

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

net = vgg(conv_arch).to(device)


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



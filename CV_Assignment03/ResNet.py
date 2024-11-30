import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import os
import torch.backends.cudnn as cudnn

os.environ["CUDA_VISIBLE_DEVICES"] = '0' # GPU Number 
start_time = time.time()
batch_size = 128
learning_rate = 0.1
root_dir = './content/app/'
#default_directory = 'drive/app/torch/save_models'

# Data Augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),               # Random Position Crop
    transforms.RandomHorizontalFlip(),                  # right and left flip
    transforms.ToTensor(),                              # change [0,255] Int value to [0,1] Float value
    transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
                         std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
])

transform_test = transforms.Compose([
    transforms.ToTensor(),                              # change [0,255] Int value to [0,1] Float value
    transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
                         std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
])

# automatically download
train_dataset = datasets.CIFAR10(root=root_dir,
                                 train=True,
                                 transform=transform_train,
                                 download=True)

test_dataset = datasets.CIFAR10(root=root_dir,
                                train=False,
                                transform=transform_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,            # at Training Procedure, Data Shuffle = True
                                           num_workers=4)           # CPU loader number

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,            # at Test Procedure, Data Shuffle = False
                                          num_workers=4)            # CPU loader number

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) #
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), #avgPooling?
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None),
            BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None),
            BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channels=16, out_channels=32, stride=2, downsample=True),
            BasicBlock(in_channels=32, out_channels=32, stride=1, downsample=None),
            BasicBlock(in_channels=32, out_channels=32, stride=1, downsample=None)

        )
        self.layer3 = nn.Sequential(
            BasicBlock(in_channels=32, out_channels=64, stride=2, downsample=True),
            BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None),
            BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None)
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ResNet()
optimizer = optim.SGD(model.parameters(), learning_rate,
                                momentum=0.9,
                                weight_decay=1e-4,
                                nesterov=True)
criterion = nn.CrossEntropyLoss()

if torch.cuda.device_count() > 0:
    print("USE", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True
else:
    print("USE ONLY CPU!")


def train(epoch):
    model.train()
    train_loss = 0 
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)

        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        if batch_idx % 10 == 0:
            print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)

        outputs = model(data)
        loss = criterion(outputs, target)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
    print('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
          .format(test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def save_checkpoint(directory, state, filename='latest.tar.gz'):

    if not os.path.exists(directory):
        os.makedirs(directory)

    model_filename = os.path.join(directory, filename)
    torch.save(state, model_filename)
    print("=> saving checkpoint")

def load_checkpoint(directory, filename='latest.tar.gz'):

    model_filename = os.path.join(directory, filename)
    if os.path.exists(model_filename):
        print("=> loading checkpoint")
        state = torch.load(model_filename)
        return state
    else:
        return None

start_epoch = 0

#checkpoint = load_checkpoint(default_directory)
#if not checkpoint:
#    pass
#else:
#    start_epoch = checkpoint['epoch'] + 1
#    model.load_state_dict(checkpoint['state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer'])

for epoch in range(start_epoch, 165):

    if epoch < 80:
        lr = learning_rate
    elif epoch < 120:
        lr = learning_rate * 0.1
    else:
        lr = learning_rate * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    train(epoch)
#    save_checkpoint(default_directory, {
#        'epoch': epoch,
#        'model': model,
#        'state_dict': model.state_dict(),
#        'optimizer': optimizer.state_dict(),
#    })
    test()  

now = time.gmtime(time.time() - start_time)
print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))

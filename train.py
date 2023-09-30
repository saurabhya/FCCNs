import torch
from torch import nn
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.backends import cudnn
from utils import *
import networks as models
from einops import rearrange
import networks_new as resnet_exp

import time
# from torch_lr_finder import LRFinder


batch_size= 64
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# transform_train = transforms.Compose([
#     transforms.ToTensor(),
#     ToHSV(),
#     ToRGB(),
#     transforms.Resize((222, 222)),
#     transforms.CenterCrop((220, 220)),
#     transforms.RandomHorizontalFlip(0.3), transforms.RandomRotation(10),
#     ToComplex(),
#
#     ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     ToHSV(),
#     ToRGB(),
#     transforms.Resize((220, 220)),
#     ToComplex(),
#     ])
#


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(38),
    transforms.CenterCrop(36),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    ToHSV(),
    ToComplex2(),
    # ToiRGB(),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(36),
    ToHSV(),
    ToComplex2(),
    # ToiRGB(),
    ])


trainset = torchvision.datasets.CIFAR10(
        root= "data", train=True, download= True, transform= transform_train
        )
trainloader = torch.utils.data.DataLoader(
        trainset, batch_size= batch_size, shuffle= True, num_workers= 8, pin_memory= False
        )


testset = torchvision.datasets.CIFAR10(
        root= "data", train=False, download= True, transform= transform_test
        )

testloader = torch.utils.data.DataLoader(
        testset, batch_size= 64, shuffle= True, num_workers= 8, pin_memory= False
        )

# net = models.VGG('vgg11', num_classes= 10).to(device)
net = models.CDS_E(num_classes= 10).to(device)
# net = models.testNet(num_classes= 10).to(device)
# net = models.toyNet().to(device)
# net = resnet_exp.resnet18(num_classes= 10).to(device)

# if device == torch.device('cuda:0'):
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark= True

criterion1= nn.CrossEntropyLoss().to(device)

def criterion2(y, theta):
    y = F.one_hot(y, num_classes= 10)
    loss = y * theta
    return loss.mean()

optimizer = optim.Adam(net.parameters(), lr= 1e-3)
# optimizer = optim.SGD(net.parameters(), lr= 1e-4, momentum= 0.8, weight_decay= 0.05)
# optimizer = optim.SGD(net.parameters(), lr= 1e-2, momentum= 0.9)


# lr_finder = LRFinder(net, optimizer, criterion, device= device)
# lr_finder.range_test(trainloader,start_lr=1e-5, end_lr= 0.1, num_iter= 100)
# lr_finder.plot()
# lr_finder.reset()

tick = time.time()



for epoch in range(50):
    epoch_loss = 0
    epoch_acc = 0.0
    total = 0
    step = 0
    correct = 0
    net.train()
    for i, (images, labels) in enumerate(trainloader, 0):
        images, labels = images.to(device), labels.to(device)
        # print(labels.shape)

        optimizer.zero_grad()

        outputs = net(images)
        outputs = rearrange(outputs, 'b c h w -> b (c h w)')
        assert outputs.dtype == torch.complex64
        outputs_phase = outputs.angle()
        outputs_magnitude = outputs.abs()

        # print(f"Shape of outputs_magnitude: {outputs_magnitude.shape}")

        loss1= criterion1(outputs_magnitude, labels)
        loss2 = criterion2(labels, outputs_phase)
        lamb = 0.2
        loss = (loss1 - loss2)
        epoch_loss += loss.item()
        step += 1

        _, predicted = torch.max(outputs_magnitude.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


        loss.backward()

        optimizer.step()


    print("\n=============>>>")
    print(f"Train Loss: {(epoch_loss / step):.4f} and Accuracy: {(correct / total) :.2f}")
    correct = 0.0
    total = 0

    net.eval()
    with torch.no_grad():
        step = 0
        val_loss = 0.0
        for i, (images, labels) in enumerate(testloader, 0):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            assert outputs.dtype == torch.complex64
            outputs = rearrange(outputs, 'b c h w -> b (c h w)')
            outputs_magnitude = outputs.abs()
            outputs_phase = outputs.angle()

            loss1= criterion1(outputs_magnitude, labels)
            loss2 = criterion2(labels, outputs_phase)
            lamb = 0.2
            loss = (loss1 - lamb*loss2)
            val_loss += loss.item()
            step += 1

            _, predicted = torch.max(outputs_magnitude.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss = val_loss / step
        print(f"Val Loss: {val_loss:.4f} and Accuracy: {(correct / total) :.2f}\n")

tock = time.time()

print(f"\n\nTotal time taken to run: {tock - tick}")

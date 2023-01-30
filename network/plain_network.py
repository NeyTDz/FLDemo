import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import parameter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import *
device = torch.device(DEVICE)

class Conv2Linear(nn.Module):
    def __init__(self):
        super(Conv2Linear, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Linear2Conv(nn.Module):
    def __init__(self, out_ch, width, height):
        super(Linear2Conv, self).__init__()
        self.out_ch = out_ch
        self.width = width
        self.height = height

    def forward(self, x):
        return x.reshape(x.size()[0], self.out_ch, self.width, self.height)


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc = nn.Sequential(
            Conv2Linear(),
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.fc(x)
        return x

def _conv2d_bn(in_channels, out_channels, kernel_size, stride, padding):
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    bn = nn.BatchNorm2d(num_features=out_channels)
    return nn.Sequential(conv, bn)

def _conv2d_bn_relu(in_channels, out_channels, kernel_size, stride, padding):
    conv2d_bn = _conv2d_bn(in_channels, out_channels, kernel_size, stride, padding)
    relu = nn.ReLU(inplace=True)
    layers = list(conv2d_bn.children())
    layers.append(relu)
    return nn.Sequential(*layers)

class _BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downscale=False):
        super(_BasicBlock, self).__init__()
        self.down_sampler = None
        stride = 1
        if downscale:
            self.down_sampler = _conv2d_bn(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
            stride = 2
        self.conv_bn_relu1 = _conv2d_bn_relu(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # don't relu here! relu on (H(x) + x)
        self.conv_bn2 = _conv2d_bn(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu_out = nn.ReLU(inplace=True)
        # residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        # residual = nn.BatchNorm2d(num_features=out_channels)
        # residual = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        if self.down_sampler:
            input = self.down_sampler(x)
        residual = self.conv_bn_relu1(x)
        residual = self.conv_bn2(residual)
        out = self.relu_out(input + residual)
        return out

class ResNet(nn.Module):
    def __init__(self, num_layer_stack=3):
        super(ResNet, self).__init__()
        self.conv1 = _conv2d_bn_relu(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.layer1 = self.__make_layers(num_layer_stack, in_channels=16, out_channels=16, downscale=False)
        self.layer2 = self.__make_layers(num_layer_stack, in_channels=16, out_channels=32, downscale=True)
        self.layer3 = self.__make_layers(num_layer_stack, in_channels=32, out_channels=64, downscale=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(in_features=64, out_features=10 if DATASET=='CIFAR10' else 100)

    def __make_layers(self, num_layer_stack, in_channels, out_channels, downscale):
        layers = []
        layers.append(_BasicBlock(in_channels=in_channels, out_channels=out_channels, downscale=downscale))
        for i in range(num_layer_stack - 1):
            layers.append(_BasicBlock(in_channels=out_channels, out_channels=out_channels, downscale=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y



class PlainNetwork(ResNet if "CIFAR" in DATASET else MnistNet):
    def __init__(self):
        super(PlainNetwork, self).__init__()
        self.flat_grad = None
        self.device = torch.device(DEVICE)

    def get_flat_grad(self,clipping=1.0):
        flat_grad = torch.tensor([]).to(self.device)
        for name, params in self.named_parameters():
            #print(name,params.shape)
            flat_grad = torch.cat([flat_grad, torch.flatten(params.grad.clamp(-clipping, clipping))], dim=0)
        flat_grad = flat_grad.cpu().numpy().flatten()
        #print(flat_grad.shape)
        return flat_grad      

    
    def update_grad(self,grad_dict):
        device = self.device
        '''
        grad_dict = {}
        start = 0
        for name, params in self.named_parameters():
            param_size = 1
            for s in params.size():
                param_size *= s
            grad_dict[name] = torch.from_numpy(flat_grads[start:start+param_size]).reshape(params.size())
            start += param_size
        '''
        for k, v in self.named_parameters():
            v.grad = grad_dict[k].to(device).type(dtype=v.grad.dtype)        
        #return grad_dict        

    def test(self,test_loader,lossf):
        device = self.device
        loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(device)
                targets = targets.to(device)
                output = self(data)
                loss += lossf(output, targets)
                correct += (output.argmax(1) == targets).sum()
                total += data.size(0)
        loss = loss.item()/len(test_loader)
        acc = correct.item()/total
        return loss,acc        

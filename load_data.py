import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import *

def get_transform(mean,std):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)]) 
    return transform

def get_dataloader(train_file,test_file,batch_size):

    train_loader = DataLoader(
        dataset=train_file,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_file,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader,test_loader

def load_dataset(dataset):
    train_file,test_file = None,None
    if dataset.upper() == 'MNIST':
        mean, std = (0.1307,), (0.3081,)
        train_file,test_file = load_mnist(get_transform(mean,std))
    elif dataset.upper() == 'FASHION-MNIST':
        mean, std = (0.2860,), (0.3530,)
        train_file,test_file = load_fashionmnist(get_transform(mean,std))
    elif dataset.upper() == "CIFAR10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        train_file,test_file = load_cifar10(get_transform(mean,std))
    elif dataset.upper() == "CIFAR100":
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        train_file,test_file = load_cifar100(get_transform(mean,std))
    elif dataset.upper() == "IMAGENET":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) 
    else:
        mean, std = None,None
        print("unknown dataset!")
    return train_file,test_file
 
def sep_dataset(client_idx,train_file):
    '''
    Seperate dataset for clients
    When CLIENT_NUM <= 20: sharding
    When CLIENT_NUM = 50,100: sharding & random overlap
    '''
    all_index = np.arange(TRAINDATA_SIZE)
    if CLIENT_NUM <= 20:
        each_size = len(train_file.data) // CLIENT_NUM
        index = all_index[client_idx * each_size:(client_idx+1) * each_size]
    elif CLIENT_NUM == 50:
        each_size = len(train_file.data) // 50 #MNIST:1000, CIFAR:1200
        local_index = all_index[client_idx * each_size:(client_idx+1) * each_size]
        other_index = np.delete(all_index,local_index)
        other_index = np.random.choice(other_index,each_size)
        index = np.concatenate((local_index,other_index))
    elif CLIENT_NUM == 100:
        each_size = len(train_file.data) // 50 #MNIST:1000, CIFAR:1200
        double_cidx = int(client_idx/2)
        local_index = all_index[double_cidx * each_size:(double_cidx+1) * each_size]
        other_index = np.delete(all_index,local_index)
        other_index = np.random.choice(other_index,each_size)
        index = np.concatenate((local_index,other_index))
    else:
        print("unknown client num")
        assert(0)
    part_train_file = torch.utils.data.Subset(train_file,index)
    return part_train_file
 
def load_mnist(transform):
    train_file = datasets.MNIST(
        root='./dataset/',
        train=True,
        transform=transform,
        download=True
    )
    test_file = datasets.MNIST(
        root='./dataset/',
        train=False,
        transform=transform
    )
    return train_file,test_file

def load_fashionmnist(transform):
    train_file = datasets.FashionMNIST(
        root='./dataset/',
        train=True,
        transform=transform,
        download=True
    )
    test_file = datasets.FashionMNIST(
        root='./dataset/',
        train=False,
        transform=transform
    )
    return train_file,test_file

def load_cifar10(transform):
    train_file = datasets.CIFAR10(
        root='./dataset/',
        train=True,
        transform=transform,
        download=True
    )
    test_file = datasets.CIFAR10(
        root='./dataset/',
        train=False,
        transform=transform
    )
    return train_file,test_file

def load_cifar100(transform):
    train_file = datasets.CIFAR100(
        root='./dataset/',
        train=True,
        transform=transform,
        download=True
    )
    test_file = datasets.CIFAR100(
        root='./dataset/',
        train=False,
        transform=transform
    )
    return train_file,test_file

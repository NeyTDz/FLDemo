import logging
import os
from pickle import TRUE
import torch

""" MSG CONSTANT """
READY_MSG = "ready"

""" NETWORK PARAM """
SERVER_IP = "127.0.0.1"
SERVER_PORT = 10975
CLIENT_IP = "127.0.0.1"
CLIENT_UDP_PORT = 12000
BUFF_SIZE = 1024*256
PORT_TABLE_COL = ['id','usend','urecv']

""" PROTOCOL PARAM"""
PROTOCOL = 'ALL' #SEP, ALL
CLIENT_NUM = 4


""" TRAIN PARAM"""
DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
DATASET = "MNIST"
TRAINDATA_SIZE = 50000 if "CIFAR" in DATASET else 60000
EPOCH = 10
BATCH_SIZE = 256 if CLIENT_NUM <= 10 else 128
LR = 1E-3
SEED = 999

""" LOG CONFIG """
LOG_DIR = "logs/{}/{}_clients/".format(PROTOCOL,CLIENT_NUM)
if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
LOG_LEVEL = logging.DEBUG


import numpy as np
import os
import pickle
from config import *


def seq_send(sock,msg):
    '''Send sequence msg larger BUFF_SIZE'''
    sendData = pickle.dumps(msg)
    remain = len(sendData)
    sock.send(pickle.dumps(remain))
    resp = sock.recv(BUFF_SIZE)
    if pickle.loads(resp) != READY_MSG:
        raise ValueError("receive unrecognized message!")
    sock.send(sendData)      

def seq_recv(sock):
    remain = pickle.loads(sock.recv(BUFF_SIZE))
    sock.send(pickle.dumps(READY_MSG))
    recvData = b""
    while remain > 0:
        each_data = sock.recv(BUFF_SIZE)
        remain -= len(each_data)
        recvData += each_data 
    msg = pickle.loads(recvData) 
    return msg

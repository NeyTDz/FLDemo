import numpy as np
import pickle
import time
import random
import torch
import torch.nn as nn
import argparse
import logging

from socket import socket, AF_INET, SOCK_STREAM, SOCK_DGRAM
from multiprocessing import Process, set_start_method
from concurrent.futures import thread

from load_data import load_dataset,sep_dataset,get_dataloader
from network.plain_network import PlainNetwork
from Client import Client
from config import *
from utils import *

device = torch.device(DEVICE)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def client_func(client:Client):
    set_seed(SEED)
    logging.basicConfig(level=LOG_LEVEL,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename="{}/client_{}.log".format(LOG_DIR,client.id),
                        filemode="w")   
    logging.debug("Debug mode")

    """ Init data&model"""
    train_file,test_file = load_dataset(DATASET)
    assert train_file
    train_file = sep_dataset(client.id,train_file)
    train_loader,test_loader = get_dataloader(train_file,test_file,BATCH_SIZE)

    model = PlainNetwork().to(device)
    optim = torch.optim.Adam(model.parameters(), LR)
    lossf = nn.CrossEntropyLoss()

    """ Init socket """
    udp_send_port, udp_recv_port = CLIENT_UDP_PORT+client.id*2, CLIENT_UDP_PORT+client.id*2+1
    udp_send_socket = socket(AF_INET, SOCK_DGRAM)  
    udp_send_socket.bind(("", udp_send_port))
    udp_recv_socket = socket(AF_INET, SOCK_DGRAM)  
    udp_recv_socket.bind(("", udp_recv_port)) 
    while True:
        try:
            tcp_client_socket_server = socket(AF_INET, SOCK_STREAM)
            tcp_client_socket_server.connect((SERVER_IP, SERVER_PORT))
            sock_msg = {'id':client.id,'usend':udp_send_port,'urecv':udp_recv_port}
            tcp_client_socket_server.send(pickle.dumps(sock_msg))
            init_msg = pickle.loads(tcp_client_socket_server.recv(BUFF_SIZE))
            client.init_stat(init_msg)
            break
        except:
            print("waiting for server...")
            time.sleep(0.5)  
   

    logging.debug("init success")

    """ Interaction between Clients(Fake concurent)"""
    
    interaction_time = 1 #(s)
    agent_data = []
    lookup_table = client.lookup_table
    counter = 0 # time counter
    interval = 0.05 # check recv buffer interval
    send_gap = 4 # send the agent params every send_gap intervals
    i = 0
    while(counter <= interaction_time):
        try:   
            #print(client.id,'in') 
            if int(counter / interval) % send_gap == 0 and counter > 0 and i < len(client.agent_ids):
                agent_id = client.agent_ids[i]
                to_port = int(lookup_table[lookup_table['id'] == agent_id]['urecv'])
                addr = ('127.0.0.1',to_port)
                data = pickle.dumps(client.id) #test: use id for data
                udp_send_socket.sendto(data,addr)
                i += 1
                logging.debug("send params to client {}".format(agent_id))    
            recv = udp_recv_socket.recvfrom(BUFF_SIZE,0x40) # 0x40 means not-blocking
        except BlockingIOError as e:
            recv = None
            time.sleep(interval)
            counter += interval
        if recv is not None:
            data, address = recv
            data = pickle.loads(data)
            agent_data.append(data)
            logging.debug("receive params from {}".format(address[1]))
    
    client.process_params(agent_data) # TODO

    

    for epoch in range(EPOCH):
        for step, (data, targets) in enumerate(train_loader):
            logging.debug("batch seq: {}".format(step))
            ########## ML Train##########
            optim.zero_grad()
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            loss = lossf(output, targets)
            loss.backward()
            flat_grad = model.get_flat_grad()
            client.grad = flat_grad

            # NOTE: add your design around Aggregate steps
            ########## Aggregate ##########
            ### generate local gradients and send to server
            msg = {'id':client.id,'local_grad':flat_grad}
            seq_send(tcp_client_socket_server,msg)
            logging.debug("send plain gradients to server")

            ### receive global gradients
            global_grad = seq_recv(tcp_client_socket_server)
            logging.debug("receive global gradients from server")

            ### update gradients of model
            grad_dict = client.process_grad(model,global_grad,PROTOCOL)
            model.update_grad(grad_dict) 
            optim.step()
            
            ########## ML Test ##########
            # batch test
            if step % 10 == 0:
                batch_loss,batch_acc = model.test(test_loader,lossf)
                #print("client {}, epoch: {}, step: {}, loss: {}, acc: {}".format(client.id, epoch, step, batch_loss, batch_acc))
                logging.info("client {}, epoch: {}, step: {}, loss: {}, acc: {}".format(client.id,epoch, step, batch_loss, batch_acc))             
        # epoch test
        epoch_loss,epoch_acc = model.test(test_loader,lossf)
        print("client {}, epoch: {}, loss: {:.4f}, acc: {:.4f}".format(client.id, epoch, epoch_loss, epoch_acc))
        logging.info("client {}, epoch: {}, loss: {}, acc: {}".format(client.id, epoch, epoch_loss, epoch_acc)) 
        for name, parameters in model.named_parameters():
            logging.debug("name: {}, size: {}, weights: {}".format(name, parameters.size(), parameters.flatten()[:5]))
        logging.info("--------------------------------------------------") 
        

    tcp_client_socket_server.close() 
    
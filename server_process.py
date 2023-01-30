
import numpy as np
import pickle
import time
from collections import defaultdict
import random
import logging

from socket import socket, AF_INET, SOCK_STREAM

from Server import Server
from config import *
from utils import *
import math

def server_func(server:Server):
    logging.basicConfig(level=LOG_LEVEL,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename="{}/server.log".format(LOG_DIR),
                        filemode="w")   
    logging.debug("Debug mode")

    """ Init socket """

    ServerSocket = socket()
    ServerSocket.bind(("", SERVER_PORT))
    ServerSocket.listen(CLIENT_NUM)

    server.init_stat()
    connections = {}
    orders = np.arange(CLIENT_NUM) if PROTOCOL == 'SEP' else np.zeros(CLIENT_NUM)
    np.random.shuffle(orders)
    for _ in range(CLIENT_NUM):
        conn, address = ServerSocket.accept()
        sock_msg = pickle.loads(conn.recv(BUFF_SIZE))
        print("Client {} connected. IP: {}, Port: {}".format(sock_msg['id'],address[0],address[1]))
        connections[sock_msg['id']] = {'conn':conn,'address':address}
        server.add_lookup_table(sock_msg)
        # NOTE: Other init variables should be added in init_msg 
        #init_msg = {'order':orders[conn_id]}

        #conn.send(pickle.dumps(init_msg))
    
    for conn_id in connections.keys():
        conn = connections[conn_id]['conn']
        init_msg = {'order':orders[conn_id],'lookup_table':server.lookup_table}
        conn.send(pickle.dumps(init_msg))
        
        
    
    logging.debug("init success")

    

    trans_stat = []

    batch_num = math.ceil(TRAINDATA_SIZE // CLIENT_NUM / BATCH_SIZE) if CLIENT_NUM < 50 \
                else math.ceil(TRAINDATA_SIZE // 25 / BATCH_SIZE)
    for epoch in range(EPOCH):
        print("---------------------------------------------")
        for b in range(batch_num):
            logging.debug("epoch: {}, batch seq: {}".format(epoch,b))
            #print("epoch:",epoch,"step:",b)

            # NOTE: add your design around Aggregate steps
            ########## Aggregate ##########
            ### receive plain gradients and aggregate
            global_grad = None
            for u in range(CLIENT_NUM):
                conn = connections[u]['conn']
                msg = seq_recv(conn)
                if u == 0:
                    global_grad = msg['local_grad']
                else:
                    global_grad += msg['local_grad']
            global_grad /= CLIENT_NUM
            logging.debug("aggregate global gradients")

            ### broadcast global gradients ###
            for u in range(CLIENT_NUM):
                conn = connections[u]['conn']
                seq_send(conn,global_grad)
            logging.debug("broadcast global gradients to clients") 
    
    
    
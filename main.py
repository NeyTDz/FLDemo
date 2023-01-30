
import numpy as np
import time
from multiprocessing import Process, set_start_method

from Client import Client
from Server import Server
from config import *
from utils import *

import server_process,client_process


if __name__ == "__main__":
    start_time = time.time()
    set_start_method('spawn')
    server = Server()
    clients = []
    for i in range(CLIENT_NUM):
        clients.append(Client(i))
    
    process_list = []
    p_server = Process(target=server_process.server_func, args=(server,))
    p_server.start()
    process_list.append(p_server)
    time.sleep(1)

    for i in range(CLIENT_NUM):
        p = Process(target=client_process.client_func, args=(clients[i], ))
        p.start()
        process_list.append(p)
        time.sleep(1)

    for p in process_list:
        p.join()

    end_time = time.time()
    print('Start:',time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(start_time)))
    print('End:  ',time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(end_time)))
    print('Cost: ',time.strftime("%H:%M:%S", time.gmtime(end_time-start_time)))

    print('end')

   
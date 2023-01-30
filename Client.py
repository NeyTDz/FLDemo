import numpy as np
import pandas as pd
import torch
from config import *
from utils import *
import random

class Client(object):
    def __init__(self,client_id):
        self.id = client_id
        self.host = CLIENT_IP
        self.grad_order = None
        self.grad = None
        self.lookup_table = pd.DataFrame(columns=PORT_TABLE_COL)
        self.agent_ids = [(self.id + 1) % CLIENT_NUM,(self.id + 2) % CLIENT_NUM]
        #self.port = gene_port(self.id,'1') # abort
    
    def init_stat(self,init_msg):
        self.grad_order = init_msg['order']
        self.lookup_table = init_msg['lookup_table']

    def process_params(self,agent_data):
        # TODO: add your code here
        pass
    
    def process_grad(self,model,global_grad,protocol):
        grad_dict = {}
        start = 0
        # TODO: add your design about partial update of clients
        if protocol == 'SEP':
            for i,(name, params) in enumerate(model.named_parameters()):
                param_size = 1
                for s in params.size():
                    param_size *= s
                if i // 2 == self.grad_order: # weight&bias
                    grad_dict[name] = torch.from_numpy(global_grad[start:start+param_size]).reshape(params.size())
                else:
                    grad_dict[name] = torch.from_numpy(self.grad[start:start+param_size]).reshape(params.size())
                start += param_size
        elif protocol == 'ALL':
            for i,(name, params) in enumerate(model.named_parameters()):
                param_size = 1
                for s in params.size():
                    param_size *= s
                grad_dict[name] = torch.from_numpy(global_grad[start:start+param_size]).reshape(params.size())
                start += param_size
        else:
            print('Invalid protocol')
            assert(0)
        return grad_dict




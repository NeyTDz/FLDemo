import numpy as np
import pandas as pd
from config import *
from utils import *

class Server(object):
    def __init__(self):
        self.host = SERVER_IP
        self.port = SERVER_PORT
        self.client_num = CLIENT_NUM
        self.lookup_table = pd.DataFrame(columns=PORT_TABLE_COL)
    
    def add_lookup_table(self,msg):
        self.lookup_table.loc[len(self.lookup_table)] = \
             [msg['id'],msg['usend'],msg['urecv']]
    
    def allocate_agents(self):
        # NOTE: For advice, you can do the allocation of agents by Server.
        #       The allocation result can be transmitted to clients by init_msg
        pass

    def init_stat(self):
        pass
        

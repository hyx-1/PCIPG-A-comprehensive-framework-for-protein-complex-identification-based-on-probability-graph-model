import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import random
# from pygcn.layers import GraphConvolution
# from dgl.nn import GraphConv, EdgeWeightNorm
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, GATConv, GCNConv, SAGEConv
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_mean_pool



class GAT(torch.nn.Module):
    def __init__(self,  hidden=512):
        super(GAT, self).__init__()
        self.nn1 = nn.Linear(128, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.gat_conv1 = GATConv(hidden, hidden, heads=4, concat=False)

        self.nn2 =  nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.gat_conv2 = GATConv(hidden, hidden, heads=4, concat=False)

        self.nn3 = nn.Linear(hidden, hidden) 
        self.bn3 = nn.BatchNorm1d(hidden)     
        self.gat_conv3 = GATConv(hidden, hidden, heads=4, concat=False)

        self.lin1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 2000)   



    def reset_parameters(self):

        self.nn1.reset_parameters()
        self.nn2.reset_parameters()
        self.nn3.reset_parameters()
        self.gat_conv1.reset_parameters()
        self.gat_conv2.reset_parameters()
        self.gat_conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.fc2.reset_parameters()


    def forward(self, x, edge_index, p=0.5):
        
        x = self.nn1(x)
        x = self.bn1(x)
        x = self.gat_conv1(x, edge_index)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.gat_conv2(x, edge_index)
        x = self.nn2(x)
        x = F.relu(x)
        x = self.bn2(x)

        # x, (edge_index, attention_weights) = self.gin_conv3(x, edge_index, return_attention_weights=True)
        x = self.gat_conv3(x, edge_index)
        x = self.nn3(x)
        x = F.relu(x)
        x = self.bn3(x)
        # x = self.gin_conv4(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x



class SAGE(nn.Module):
    def __init__(self):
        super(SAGE, self).__init__()
        hidden = 128
        self.conv1 = SAGEConv(18, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
  
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)


        self.sag1 = SAGPooling(hidden,0.5)
        self.sag2 = SAGPooling(hidden,0.5)
        self.sag3 = SAGPooling(hidden,0.5)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(0.5)
        # for param in self.parameters():
        #     print(type(param), param.size())


    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.bn1(x)
        y = self.sag1(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1] 

        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x) 
        x = self.bn2(x)
        y = self.sag2(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]  
        
        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x) 
        x = self.bn3(x)
        y = self.sag3(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]


        return global_mean_pool(y[0], y[3])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ppi_model(nn.Module):
    def __init__(self):
        super(ppi_model,self).__init__()
        self.BGNN = SAGE()
        self.TGNN = GAT()

    def forward(self, batch, p_x_all, p_edge_all, edge_index, p=0.5):
        edge_index = edge_index.to(device)
        batch = batch.to(torch.int64).to(device)
        x = p_x_all.to(torch.float32).to(device)
        edge = torch.LongTensor(p_edge_all.to(torch.int64)).to(device)
        embs = self.BGNN(x, edge, batch-1)
        del p_x_all, p_edge_all
        final = self.TGNN(embs, edge_index, p=0.5)
        return final
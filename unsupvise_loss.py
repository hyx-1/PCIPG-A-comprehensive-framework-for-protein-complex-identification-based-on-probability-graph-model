import torch
import math
from torch import nn
import torch.nn.functional as f

class Unsupvise_loss(nn.Module):
    def __init__(self):
        super(Unsupvise_loss, self).__init__()

    def forward(self, edges, F, p_no_comm=1e-4, neg_scale=1.0):
        device = F.device  # 确定F所在的设备
        p_no_comm_tensor = torch.tensor(p_no_comm, dtype=torch.float32, device=device)
        eps = torch.log(1 / (1 - p_no_comm_tensor))
        all_ones = torch.ones(F.shape[0], F.shape[0], dtype=torch.float32, device=device)
        A = torch.zeros_like(all_ones)
        A[edges[0], edges[1]] = 1
        A[edges[1], edges[0]] = 1
        
        FUV = torch.matmul(F, F.T)/math.sqrt(F.shape[1])
        mask = torch.eye(FUV.shape[0], device=device, dtype=torch.bool)
        FUV.masked_fill_(mask, float('-inf'))
        FUV = f.softmax(FUV, dim=-1)
        
        edges_loss = torch.zeros(1, dtype=torch.float32, device=device)
        non_edges_loss = torch.zeros(1, dtype=torch.float32, device=device)
        
        edges_loss = -torch.log1p(-torch.exp(-eps - FUV[edges[0], edges[1]]))
        edges_loss = torch.sum(edges_loss)
        
        non_edges_mask = (all_ones - A) - torch.eye(F.shape[0], device=device)

        non_edges_loss = FUV * non_edges_mask
        non_edges_loss = torch.sum(non_edges_loss)
        
        non_edges_num = non_edges_mask.sum()# 减去对角线元素的数量

        loss = (edges_loss / edges.shape[1] + non_edges_loss / non_edges_num) / (1 + neg_scale)
        return loss
    
class Unsupvise_weight_loss(nn.Module):
    def __init__(self):
        super(Unsupvise_weight_loss, self).__init__()

    def forward(self, edges, F, S, p_no_comm=1e-4, neg_scale=1.0):
        device = F.device 
        p_no_comm_tensor = torch.tensor(p_no_comm, dtype=torch.float32, device=device)
        eps = torch.log(1 / (1 - p_no_comm_tensor))
        all_ones = torch.ones(F.shape[0], F.shape[0], dtype=torch.float32, device=device)
        A = torch.zeros_like(all_ones)
        A[edges[0], edges[1]] = 1
        A[edges[1], edges[0]] = 1
        
        FUV = torch.matmul(F, F.T)/math.sqrt(F.shape[1])
        mask = torch.eye(FUV.shape[0], device=device, dtype=torch.bool)
        FUV.masked_fill_(mask, float('-inf'))
        FUV = f.softmax(FUV, dim=-1)
        
        edges_loss = torch.zeros(1, dtype=torch.float32, device=device)
        non_edges_loss = torch.zeros(1, dtype=torch.float32, device=device)
        
        edges_loss = -torch.log1p(-torch.exp(-eps - FUV[edges[0], edges[1]]))
        edges_loss = torch.sum(edges_loss)
        
        logpipj = -torch.log(FUV[edges[0], edges[1]])
        logpipj = torch.sum(logpipj)

        B = S[edges[0], edges[1]]
        weight = torch.tensor((torch.ones(1, B.shape[0])-B)/B)
        weight = weight.to('cuda:0')
        logsijpij = weight*FUV[edges[0], edges[1]]
        logsijpij = torch.sum(logsijpij)

        non_edges_mask = (all_ones - A) - torch.eye(F.shape[0], device=device)       
        non_edges_loss = FUV * non_edges_mask
        non_edges_loss = torch.sum(non_edges_loss)
        
        non_edges_num = non_edges_mask.sum()
        
        loss = ((edges_loss+logpipj+logsijpij) / edges.shape[1] + non_edges_loss / non_edges_num) / (1 + neg_scale)
        return loss





import os
import torch
import json
import numpy as np
import random
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from unsupvise_loss import Unsupvise_weight_loss
from sage_gat import ppi_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
seed = 42  
set_seed(seed)

writer = SummaryWriter('HCT116_sage_gat_train_v1')
with open('./protein_HCT116_name.json', 'r') as f:
    data = json.load(f)
    
protein_nodes = len(data)
S = torch.zeros(protein_nodes, protein_nodes)
with open('../PPI_Dataset/HCT116_weight.txt', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip().split()
        if line[4] in data and line[5] in data:
            S[data[line[4]], data[line[5]]] = S[data[line[5]], data[line[4]]]=float(line[-1])

def multi2big_x(x_ori):
    x_cat = torch.zeros(1, 7)
    x_num_index = torch.zeros(protein_nodes)
    for i in range(protein_nodes):
        x_now = torch.tensor(x_ori[i])
        x_num_index[i] = torch.tensor(x_now.size(0))
        x_cat = torch.cat((x_cat, x_now), 0)
    return x_cat[1:, :], x_num_index

def multi2big_batch(x_num_index):
    num_sum = x_num_index.sum()
    num_sum = num_sum.int()
    batch = torch.zeros(num_sum)
    count = 1
    for i in range(1,protein_nodes):
        zj1 = x_num_index[:i]
        zj11 = zj1.sum()
        zj11 = zj11.int()
        zj22 = zj11 + x_num_index[i]
        zj22 = zj22.int()
        size1 = x_num_index[i]
        size1 = size1.int()
        tc = count * torch.ones(size1)
        batch[zj11:zj22] = tc
        test = batch[zj11:zj22]
        count = count + 1
    batch = batch.int()
    return batch

def multi2big_edge(edge_ori, num_index):
    edge_cat = torch.zeros(2, 1)
    edge_num_index = torch.zeros(protein_nodes)
    for i in range(protein_nodes):
        edge_index_p = edge_ori[i]
        edge_index_p = np.asarray(edge_index_p)
        edge_index_p = torch.tensor(edge_index_p.T)
        edge_num_index[i] = torch.tensor(edge_index_p.size(1))
        if i == 0:
            offset = 0
        else:
            zj = torch.tensor(num_index[:i])
            offset = zj.sum()
        edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
    return edge_cat[:, 1:], edge_num_index

model = ppi_model()
model.to('cuda')
p_x_all = torch.load('./HCT116_x_list.pt')
p_edge_all = np.load('./HCT116_edge_list_amino.npy', allow_pickle=True)
p_x_all, x_num_index = multi2big_x(p_x_all)
p_edge_all, edge_num_index = multi2big_edge(p_edge_all, x_num_index)
batch = multi2big_batch(x_num_index) + 1
ppi = np.load('./HCT116_ppi.npy')
protein_edge = torch.tensor(np.array(ppi).T)

Loss = Unsupvise_weight_loss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
def train(epochs, batch, model, p_x_all, p_edge_all, protein_edge):
    for epoch in range(1, epochs+1):
        model.train()
        F = model(batch, p_x_all, p_edge_all, protein_edge)
        optimizer.zero_grad() 
        loss = Loss(protein_edge, F, S)
        loss.backward()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        # print(f'Epoch {epoch}  learning rate: {current_lr}')
        # for name, param in model.named_parameters():
        #     print(f'Parameter: {name}, Gradient: {param.grad}')
        optimizer.step()
        if epoch % 25 == 0:
            print(f'{epoch} loss is {loss}')
        
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('lr', current_lr, epoch)
        torch.save({'epoch': epoch,
                'state_dict': model.state_dict()},

                os.path.join('./save_sage_gat/', 'HCT116_sage_gat_model_train_v1.ckpt'))
train(2000, batch, model, p_x_all, p_edge_all, protein_edge)
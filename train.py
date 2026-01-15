import os
import torch
import json
import time
start_time = time.time()
import numpy as np
import random
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from unsupvise_loss import Unsupvise_loss, Unsupvise_weight_loss
from sage_gat import ppi_model
import argparse

parser = argparse.ArgumentParser(description="Dataprocess")
parser.add_argument("--log", type=str, help="model train log")
parser.add_argument("--Protein_name", type=str,  help="Output protein corresponding number, json is the suffix")
parser.add_argument("--Amino_conact_matrix",type=str, help="Amino acid residue contact matrix, npy is the suffix")
parser.add_argument("--x_list_feature", type=str, help="All amino acid residue characteristics, pt is the suffix")
parser.add_argument("--ppi_npy", type=str, help="Output protein interaction matrix, npy is the suffix")
parser.add_argument("--model_save", type=str, help="Output model weight, ckpt is the suffix")

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
args = parser.parse_args()


writer = SummaryWriter(args.log)
with open(args.Protein_name, 'r') as f:
    data = json.load(f)
protein_nodes = len(data)
# S = torch.zeros(protein_nodes, protein_nodes)
# with open('../PPI_Dataset/HI_union_weight.txt', 'r') as f:
#     lines = f.readlines()
#     for line in lines[1:]:
#         line = line.strip().split()
#         if line[4] in data and line[5] in data:
#             S[data[line[4]], data[line[5]]] = S[data[line[5]], data[line[4]]]=float(line[-1])
feature_num = 18
def multi2big_x(x_ori):
    x_cat = torch.zeros(1, feature_num)
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
p_x_all = torch.load(args.x_list_feature)
p_edge_all = np.load(args.Amino_conact_matrix, allow_pickle=True)
p_x_all, x_num_index = multi2big_x(p_x_all)
p_edge_all, edge_num_index = multi2big_edge(p_edge_all, x_num_index)
batch = multi2big_batch(x_num_index) + 1
ppi = np.load(args.ppi_npy)
protein_edge = torch.tensor(np.array(ppi).T)

# Loss = Unsupvise_weight_loss()
Loss = Unsupvise_loss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)
scaler = torch.amp.GradScaler()
def train(epochs, batch, model, p_x_all, p_edge_all, protein_edge):
    for epoch in range(1, epochs+1):
        model.train()
        with torch.amp.autocast('cuda'):
            F = model(batch, p_x_all, p_edge_all, protein_edge)
            optimizer.zero_grad() 
            # loss = Loss(protein_edge, F, S)
            loss = Loss(protein_edge, F)
        # loss.backward()
        scheduler.step()
        scaler.scale(loss).backward()  # 使用梯度缩放器缩放损失
        scaler.step(optimizer)  # 更新模型参数
        scaler.update() 
        current_lr = optimizer.param_groups[0]['lr']
        # print(f'Epoch {epoch}  learning rate: {current_lr}')
        # for name, param in model.named_parameters():
        #     print(f'Parameter: {name}, Gradient: {param.grad}')
        # optimizer.step()
        if epoch % 25 == 0:
            print(f'{epoch} loss is {loss}')
        
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('lr', current_lr, epoch)
        torch.save({'epoch': epoch,
                'state_dict': model.state_dict()},
                args.model_save)
train(2000, batch, model, p_x_all, p_edge_all, protein_edge)
end_time = time.time()
print(f"运行时间: {end_time - start_time} 秒")
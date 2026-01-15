import os
import torch
import numpy as np
import json
import pandas as pd
import networkx as nx
from sage_gat import ppi_model
import argparse

parser = argparse.ArgumentParser(description="Dataprocess")
parser.add_argument("--Protein_name", type=str,  help="Output protein corresponding number, json is the suffix")
parser.add_argument("--Amino_conact_matrix",type=str, help="Amino acid residue contact matrix, npy is the suffix")
parser.add_argument("--x_list_feature", type=str, help="All amino acid residue characteristics, pt is the suffix")
parser.add_argument("--ppi_npy", type=str, help="Output protein interaction matrix, npy is the suffix")
parser.add_argument("--model_save", type=str, help="Output model weight, ckpt is the suffix")
parser.add_argument("--PPI_in_Ground_truth", type=str,  help="Some interactions in the gold standard, txt is the suffix")
parser.add_argument("--save_top_k", type=str, help="save topk protein complex folder")


args = parser.parse_args()

with open(args.Protein_name, 'r') as f:
    protein_name = json.load(f)

protein_nodes = len(protein_name)
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

p_x_all = torch.load(args.x_list_feature)
p_edge_all = np.load(args.Amino_conact_matrix, allow_pickle=True)
p_x_all, x_num_index = multi2big_x(p_x_all)
p_edge_all, edge_num_index = multi2big_edge(p_edge_all, x_num_index)
batch = multi2big_batch(x_num_index) + 1
ppi = np.load(args.ppi_npy)
protein_edge = torch.tensor(np.array(ppi).T)


model = ppi_model()
model.to('cuda')
model_path = args.model_save
model.load_state_dict(torch.load(model_path)['state_dict'])
model.eval()


F = model(batch, p_x_all, p_edge_all, protein_edge)
idx_2_protein = {v: k for k, v in protein_name.items()}

def save_topk(k):
    _, indices = torch.topk(F, k, dim=0, largest=True)
    protein = []
    protein_idx = []
    for col in indices.t():
        if set(col.tolist()) not in protein_idx:
            protein_idx.append(set(col.tolist()))
            temp = col.tolist()
            temp_protein = [idx_2_protein[i] for i in temp]
            protein.append(temp_protein)            
    df = pd.DataFrame(protein)
    file_name = os.path.join(args.save_top_k, 'top_'+str(k)+'.txt')
    df.to_csv(file_name, sep='\t', header=False, index=False)

# def merge_file(k):
#     directory = args.save_top_k
#     file_pattern = 'top_{}.txt'
#     filenames_to_merge = [file_pattern.format(i) for i in range(3, k+1)]  
#     output_filename = 'result_sage_gat.txt'

#     with open(os.path.join(directory, output_filename), 'w') as outfile:
#         for filename in filenames_to_merge:
#             with open(os.path.join(directory, filename), 'r') as infile:
#                 outfile.write(infile.read())
                # outfile.write('\n')
def merge_file2(k):
    directory = args.save_top_k
    file_pattern = 'top_{}.txt'
    filenames_to_merge = [file_pattern.format(i) for i in range(5, k+1)] 
    output_filename = 'result_sage_gat.txt'
    PPI_in_gt = pd.read_csv(args.PPI_in_Ground_truth,  sep="\t")

    # 将PPI_in_gt的DataFrame转换为列表形式的2元组
    edges = list(PPI_in_gt[['Protein1', 'Protein2']].itertuples(index=False, name=None))
    # edges = list(PPI.itertuples(index=False, name=None))
    G = nx.Graph()
    G.add_edges_from(edges)
    PPI_hyperedge_dup = list(nx.find_cliques(G))
    # 找到所有团后，过滤掉大小不在3到10之间的团
    PPI_hyperedge_dup = [clique for clique in nx.find_cliques(G) if 3 <= len(clique) <= 4]
    with open(os.path.join(directory, output_filename), 'w') as outfile:
        for clique in PPI_hyperedge_dup:
            outfile.write("\t".join(clique) + "\n")
    with open(os.path.join(directory, output_filename), 'a') as outfile:
        for filename in filenames_to_merge:
            with open(os.path.join(directory, filename), 'r') as infile:
                outfile.write(infile.read())


for i in range(3, 35):        
    save_topk(i)

# merge_file(34)
merge_file2(34)

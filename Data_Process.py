import os
import json
import torch
import math
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import shutil
import argparse

parser = argparse.ArgumentParser(description="Dataprocess")
parser.add_argument("--PPI_path", type=str, help="Specify protein interactions")
parser.add_argument("--Protein_path", type=str, help="Specify the residue characteristics of each protein in the interaction network")
parser.add_argument("--Ground_truth", type=str,  help="The gold standard corresponding to the dataset")
parser.add_argument("--PPI_in_Ground_truth", type=str,  help="Some interactions in the gold standard, txt is the suffix")
parser.add_argument("--contact_map_path", type=str,  help="Protein contact map")
parser.add_argument("--Protein_name", type=str,  help="Output protein corresponding number, json is the suffix")
parser.add_argument("--Amino_conact_matrix",type=str, help="Amino acid residue contact matrix, npy is the suffix")
parser.add_argument("--x_list_feature", type=str, help="All amino acid residue characteristics, pt is the suffix")
parser.add_argument("--ppi_npy", type=str, help="Output protein interaction matrix, npy is the suffix")

def Load_txt_list(file_name, display_flag=True):
    #if display_flag:
        #print(f'Loading {path}{file_name}')
    list = []
    with open(f'{file_name}', 'r') as f:
        lines = f.readlines()
        for line in lines:
            node_list = line.strip('\n').strip(' ').split(' ')
            list.append(node_list)
    return list

def Get_protein_list(folder_path):
    filenames = []
    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        # 确保是文件而不是子文件夹
        if os.path.isfile(os.path.join(folder_path, file)):
            # 使用 splitext 去掉文件后缀
            filename_without_extension = os.path.splitext(file)[0]
            filenames.append(filename_without_extension)
    return filenames

def read_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(int, line.strip().split())) for line in lines]
    return matrix

def matrix_to_undirected_edges(matrix):
    edges = []
    num_nodes = len(matrix)
    for i in range(num_nodes):
        for j in range(i, num_nodes):  # 只需要遍历上三角部分
            if matrix[i][j] != 0:
                edges.append((i, j))
    return edges


if __name__ == '__main__':
    args = parser.parse_args()
    current_path = os.getcwd()
    PPI_path = args.PPI_path
    Protein_path = args.Protein_path
    
    # 读取PPI网络和金标准复合物
    PPI = pd.read_csv(PPI_path, sep='\t', header=None)
    gt = Load_txt_list(args.Ground_truth)

    # 获取PPI网络中下载到的蛋白质
    protein_nodes = set(Get_protein_list(Protein_path))

    # 计算PPI网络蛋白质中可用的金标准复合物
    gt_in_PPI = []
    for complex in gt:
        protein_in_PPI = []
        for protein in complex:
            if protein in protein_nodes:
                protein_in_PPI.append(protein)
        if protein_in_PPI == complex:
            gt_in_PPI.append(complex)

    # 计算可用复合物包含的蛋白质
    protein_in_gt = set()
    for complex in gt_in_PPI:
        for protein in complex:
            protein_in_gt.add(protein)

    # 将PPI网络转化为集合以方便后面计算
    PPI_set = set()
    for _, row in PPI.iterrows():
        proteins = frozenset(row)
        PPI_set.add(proteins)

    # 计算可用蛋白质张成的PPI网络
    PPI_in_gt = set()
    for index, row in PPI.iterrows():
        if (row[0] in protein_in_gt) and (row[1] in protein_in_gt):
            PPI_in_gt.add((row[0],row[1]))
    PPI_in_gt = pd.DataFrame(list(PPI_in_gt), columns=['Protein1', 'Protein2'])
    PPI_in_gt.to_csv(args.PPI_in_Ground_truth, sep='\t', index=False)#相互作用
    # print(gt_in_PPI)
    # print(protein_in_gt)
    # print(PPI_in_gt)
    print(f'Processed dataset {args.PPI_path} contains {len(gt_in_PPI)} complexes, {len(protein_in_gt)} proteins, {len(PPI_in_gt)} PPI edges.')
    
    # 保存protein_in_gt中的氨基酸接触矩阵到npy文件中
    # 筛选出出现在复合物里面的蛋白
    # list_all保存氨基酸连接性
    # fearture_list保存氨基酸特征
    # protein_edge保存蛋白质的连接性
    protein_name = {}
    list_all = [] 
    fearture_list = [] 
    contact_map_path = args.contact_map_path
    fearture_path = Protein_path
    all_sequence = dict()
    num = 0
    for name in protein_in_gt:
        # print(name)
        if name not in protein_name:
            protein_name[name] = num
            num += 1
        # all_sequence[name] = sequences[name]
        path = os.path.join(contact_map_path, name + '.txt')
        fearture_file = os.path.join(fearture_path, name + '.txt')
        # with open(fearture_file, 'r') as f:
        #     lines = f.readlines()
        #     fearture = [list(map(float, line.strip().split()[2:])) for line in lines[1:]]
        # fearture_list.append(np.array(fearture))
        with open(fearture_file, 'r') as f:
            temp = []
            lines = f.readlines()
            # feature = [list(map(float, line.strip().split()[2:20])) for line in lines[1:]]
            for line in lines[1:]:
                l1 = list(map(float, line.strip().split()[2:6]))
                l2 = list(map(float, line.strip().split()[10:]))
                features = l1 + l2 
                temp.append(features)
        fearture_list.append(np.array(temp))
        matrix = read_matrix_from_file(path)
        edges = matrix_to_undirected_edges(matrix)
        list_all.append(edges)
    # print(protein_name)
    filename = args.Protein_name
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(protein_name, f, ensure_ascii=False, indent=4)
    list_all = np.array(list_all, dtype=object)
    np.save(args.Amino_conact_matrix, list_all)
    torch.save(fearture_list, args.x_list_feature)
    # 根据protein_name把ppi转成无向图的边
    protein_edge = []
    n = len(protein_name)
    with open(args.PPI_in_Ground_truth, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            protein1, protein2 = line.strip().split()
            ppi = [protein_name[protein1], protein_name[protein2]]
            protein_edge.append(ppi)
    protein_edge.sort(key=lambda x: x[0])
    np.save(args.ppi_npy, protein_edge)



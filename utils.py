import os
import math
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from evaluation import *
from tqdm import tqdm

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

def replace_extension(file_path, new_extension):
    # 去掉原始文件的后缀
    filename_without_extension = os.path.splitext(file_path)[0]
    # 拼接新的后缀
    new_file_path = f"{filename_without_extension}.{new_extension}"
    return new_file_path

def jaccard_similarity_of_lists(list1, list2):
    sets1 = [frozenset(sublist) for sublist in list1]
    sets2 = [frozenset(sublist) for sublist in list2]
    union_of_sets = set().union(sets1, sets2)
    intersection_of_sets = set().union(sets1)
    intersection_of_sets.intersection_update(sets2)
    jaccard_index = len(intersection_of_sets) / len(union_of_sets)
    return jaccard_index

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

def update_dict_value(key, my_dict):
    if key not in my_dict:
        my_dict[key] = 1
    else:
        my_dict[key] += 1

def cal_prop_of_link(gt_in_PPI_data, PPI_set_data):
    edges_in_complex_data = pd.DataFrame(columns=['all', 'true'])
    for id, complex_data in tqdm(enumerate(gt_in_PPI_data)):
        num_proteins = len(complex_data)
        edges_in_complex_data.loc[id, 'all'] = math.comb(num_proteins, 2)
        protein_pairs = itertools.combinations(complex_data, 2)  # 使用组合生成所有蛋白质对
        edges = 0
        for protein1, protein2 in protein_pairs:
            pair_set = frozenset([protein1, protein2])
            if pair_set in PPI_set_data:
                edges += 1
        edges_in_complex_data.loc[id, 'true'] = edges

    edges_in_complex_data['prop'] = edges_in_complex_data['true'] / edges_in_complex_data['all']
    return edges_in_complex_data

def cal_prob_epsilon(edges_in_complex_data, gt_in_PPI_data):
    prop_larger_than_epsilon = {}
    for integer in range(100):
        epsilon = integer/100
        tight = edges_in_complex_data['prop'] > epsilon
        prop_larger_than_epsilon[epsilon]= tight.sum()/len(gt_in_PPI_data)
    epsilons_data = list(prop_larger_than_epsilon.keys())
    proportions_data = list(prop_larger_than_epsilon.values())
    return epsilons_data, proportions_data
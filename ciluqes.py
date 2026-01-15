import pandas as pd
import networkx as nx
import os
from utils import *

if __name__ == '__main__':
    # dataname = 'krogan_core'
    # current_path = os.getcwd()
    # PPI_path = "../PPI_Dataset/biogrid.txt" 
    # Protein_path = "../Residue_feature_22"

    # # 读取PPI网络和金标准复合物
    # PPI = pd.read_csv(PPI_path, sep='\t', header=None)
    # gt = Load_txt_list("../PPI_Dataset/golden_standard.txt")

    # # 获取PPI网络中下载到的蛋白质
    # protein_nodes = set(Get_protein_list(Protein_path))

    # # 计算PPI网络蛋白质中可用的金标准复合物
    # gt_in_PPI = []
    # for complex in gt:
    #     protein_in_PPI = []
    #     for protein in complex:
    #         if protein in protein_nodes:
    #             protein_in_PPI.append(protein)
    #     if protein_in_PPI == complex:
    #         gt_in_PPI.append(complex)

    # # 计算可用复合物包含的蛋白质
    # protein_in_gt = set()
    # for complex in gt_in_PPI:
    #     for protein in complex:
    #         protein_in_gt.add(protein)

    # # 计算可用蛋白质张成的PPI网络
    # PPI_in_gt = set()
    # for index, row in PPI.iterrows():
    #     if (row[0] in protein_in_gt) and (row[1] in protein_in_gt):
    #         PPI_in_gt.add((row[0],row[1]))
    # PPI_in_gt = pd.DataFrame(list(PPI_in_gt), columns=['Protein1', 'Protein2'])
    # print(PPI_in_gt)
    # #print(gt_in_PPI)
    PPI_in_gt = pd.read_csv('../fuxian/data/biogrid/biogrid_ppi_in_gt.txt',  sep="\t")
    print(PPI_in_gt)

    # 将PPI_in_gt的DataFrame转换为列表形式的2元组
    edges = list(PPI_in_gt[['Protein1', 'Protein2']].itertuples(index=False, name=None))
    # edges = list(PPI.itertuples(index=False, name=None))
    G = nx.Graph()
    G.add_edges_from(edges)
    PPI_hyperedge_dup = list(nx.find_cliques(G))
    # 找到所有团后，过滤掉大小不在3到10之间的团
    PPI_hyperedge_dup = [clique for clique in nx.find_cliques(G) if 3 <= len(clique) <= 4]
    print(len(PPI_hyperedge_dup))
    with open('./test.txt', 'w') as outfile:
        for clique in PPI_hyperedge_dup:
            outfile.write("\t".join(clique) + "\n")
    # PPI_hyperedge_dup_df = pd.DataFrame(PPI_hyperedge_dup)
    # PPI_hyperedge_dup_df.to_csv('./ppi_test_robustness/result_cliques_' + dataname + '_del_20%.txt',header = None,sep = '\t',index = None)
    # print(PPI_hyperedge_dup)
    # print(len(PPI_hyperedge_dup))
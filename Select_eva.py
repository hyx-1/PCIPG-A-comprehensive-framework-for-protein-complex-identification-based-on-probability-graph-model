from utils import *
from evaluation import *
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Dataprocess")
parser.add_argument("--PPI_path", type=str, help="Specify protein interactions")
parser.add_argument("--Protein_path", type=str, help="Specify the residue characteristics of each protein in the interaction network")
parser.add_argument("--PPI_in_Ground_truth", type=str,  help="Some interactions in the gold standard, txt is the suffix")
parser.add_argument("--result", type=str, help="save topk protein complex result")



if __name__ == '__main__':
    args = parser.parse_args()
    current_path = '../../'
    PPI_path = args.PPI_path
    Protein_path = args.Protein_path
    
    # 读取PPI网络和金标准复合物
    PPI = pd.read_csv(PPI_path, sep='\t', header=None)
    gt = Load_txt_list(args.PPI_in_Ground_truth)

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
    T = 0.2
    T_max = 1
    T_min = 1
    #test_complex 为第一步输出的复合物列表
    # 读取文件并将每行存入列表
    file_path = args.result
    with open(file_path, 'r', encoding='utf-8') as file:
        test_complex = [line.strip().split() for line in file]

    edges_in_complex_test = cal_prop_of_link(test_complex, PPI_set)
    predict_complex_index = edges_in_complex_test[edges_in_complex_test['prop'] > T].index.tolist()
    # predict_complex_index = edges_in_complex_test[(edges_in_complex_test['prop'] >= T_min) & (edges_in_complex_test['prop'] <= T_max)].index.tolist()
    predict_complex = []
    for index in predict_complex_index:
        predict_complex.append(test_complex[index])
    print(len(predict_complex))
    precision_temp, recall_temp, f1_temp, acc_temp, sn_temp, PPV_temp, score_temp, p_true, r_true, true_info = get_score(gt_in_PPI, predict_complex)
    print(score_temp)

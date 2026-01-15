# # conda env create -f environment.yml

mkdir -p data/collins/

python Data_Process.py \
--PPI_path "../PPI_Dataset/collins.txt" \
--Protein_path "../Residue_Fea_Choose/collins/Residue_feature_22/" \
--Ground_truth "../PPI_Dataset/golden_standard.txt" \
--PPI_in_Ground_truth "data/collins/collins_ppi_in_gt.txt" \
--contact_map_path "../Feature_dataset/collins/Contact_map/" \
--Protein_name "data/collins/protein_collins_name.json" \
--Amino_conact_matrix "data/collins/collins_edge_list_amino.npy" \
--x_list_feature "data/collins/collins_x_list.pt" \
--ppi_npy "data/collins/collins_ppi.npy"

mkdir -p logs/train_logs/
mkdir -p logs/output_logs/
mkdir -p logs/result_logs/
mkdir -p models/

python train.py \
--log "logs/train_logs/collins.log" \
--Protein_name "data/collins/protein_collins_name.json" \
--Amino_conact_matrix "data/collins/collins_edge_list_amino.npy" \
--x_list_feature "data/collins/collins_x_list.pt" \
--ppi_npy "data/collins/collins_ppi.npy" \
--model_save "models/collins_sage_gat_tarin.ckpt" > logs/output_logs/collins.log

mkdir -p result/collins/save_top_k

python test.py \
--Protein_name "data/collins/protein_collins_name.json" \
--Amino_conact_matrix "data/collins/collins_edge_list_amino.npy" \
--x_list_feature "data/collins/collins_x_list.pt" \
--ppi_npy "data/collins/collins_ppi.npy" \
--model_save "models/collins_sage_gat_tarin.ckpt" \
--PPI_in_Ground_truth "data/collins/collins_ppi_in_gt.txt" \
--save_top_k "result/collins/save_top_k/"


python Select_eva.py \
--PPI_path "../PPI_Dataset/collins.txt" \
--Protein_path "../Residue_Fea_Choose/collins/Residue_feature_22/" \
--PPI_in_Ground_truth "../PPI_Dataset/golden_standard.txt" \
--result "result/collins/save_top_k/result_sage_gat.txt" > logs/result_logs/collins.log

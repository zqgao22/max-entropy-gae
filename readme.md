This PyTorch implementation contains the code used for the "Handling Missing Data via Max-Entropy Regularized Graph Autoencoder".

Dependencies:\
python 3.7\
numpy 1.21.3\
torch 1.9.0\
scipy 1.7.1\
scikit-learn 1.0.1

Usage:\
To train missing data imputation on PROTEINS_FULL dataset at 10% MCAR.\
python train.py --lr 1e-3 --epochsize 100 --max_num_nodes 1000 --gamma 0.15 --missing_r 0.15 --feature_dim_c 29 --feature_dim_d 29 --feature_dim_e 20



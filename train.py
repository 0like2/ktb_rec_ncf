import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from NeuMF import NeuMFEngine
from data_preprocess import Loader  # Assuming data_preprocess.py contains the necessary Loader class for preprocessing
import torch
import os

# Configurations for the models
gmf_config = {
    'alias': 'gmf_factor8neg4-implict',
    'num_epoch': 200,
    'batch_size': 1024,
    'optimizer': 'adam',
    'adam_lr': 1e-3,
    'latent_dim': 8,
    'num_negative': 4,
    'l2_regularization': 0,
    'weight_init_gaussian': True,
    'use_cpu': True,
    'use_bachify_eval': False,
    'device_id': 0,
}

mlp_config = {
    'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
    'num_epoch': 200,
    'batch_size': 256,
    'optimizer': 'adam',
    'adam_lr': 1e-3,
    'latent_dim': 8,
    'num_negative': 4,
    'layers': [16, 64, 32, 16, 8],
    'l2_regularization': 0.0000001,
    'weight_init_gaussian': True,
    'use_cpu': True,
    'use_bachify_eval': False,
    'device_id': 0,
    'pretrain': False,
}

neumf_config = {
    'alias': 'neumf_factor8neg4',
    'num_epoch': 200,
    'batch_size': 1024,
    'optimizer': 'adam',
    'adam_lr': 1e-3,
    'latent_dim_mf': 8,
    'latent_dim_mlp': 8,
    'num_negative': 4,
    'layers': [16, 64, 32, 16, 8],
    'l2_regularization': 0.0000001,
    'weight_init_gaussian': True,
    'use_cpu': True,
    'use_bachify_eval': True,
    'device_id': 0,
    'pretrain': False,
}

# Paths for data and similarity matrix
file_path = 'input'
similarity_matrix_file = 'similarity_matrix.csv'  # Use the precomputed similarity matrix

# Load the data using the Loader class
loader = Loader(file_path, similarity_matrix_file)
train_dataset, test_dataset = loader.load_dataset()

# Automatically set num_users and num_items based on dataset
num_users = train_dataset.num_users  # Assuming Loader provides this info
num_items = train_dataset.num_items  # Assuming Loader provides this info

common_config_updates = {
    'num_users': num_users,
    'num_items': num_items,
    'num_item_categories': len(item_df['item_category'].unique()),  # 아이템 카테고리 수
    'num_channel_categories': len(creator_df['channel_category'].unique()),  # 채널 카테고리 수
    'meta_latent_dim': 4  # 메타데이터의 임베딩 차원 (필요 시 조정)
}

gmf_config.update(common_config_updates)
mlp_config.update(common_config_updates)
neumf_config.update(common_config_updates)

# Create DataLoader for training and testing
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=neumf_config['batch_size'], shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=neumf_config['batch_size'], shuffle=False)

# Choose the configuration for the model to train
config = neumf_config  # Here we choose NeuMF configuration; can switch to gmf_config or mlp_config as needed
engine = NeuMFEngine(config)

# Output directory for saving models
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Training loop
for epoch in range(config['num_epoch']):
    print(f'Epoch {epoch} starts!')
    print('-' * 80)

    # Train the model for one epoch
    engine.train_an_epoch(train_loader, epoch_id=epoch)

    # Evaluate the model
    hit_ratio, ndcg = engine.evaluate(test_loader, epoch_id=epoch)

    # Save the model with performance metrics
    model_path = os.path.join(output_dir, f"{config['alias']}_Epoch{epoch}_HR{hit_ratio:.4f}_NDCG{ndcg:.4f}.model")
    engine.save(model_path)

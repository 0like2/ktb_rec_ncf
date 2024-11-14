import torch
from gmf import GMFEngine
from mlp import MLPEngine
from NeuMF import NeuMFEngine
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

# CSV 파일 경로 설정
creator_file_path = '/Users/iyeonglag/PycharmProjects/ktb_rec_ncf/input/Creator_random25.csv'
item_file_path = '/Users/iyeonglag/PycharmProjects/ktb_rec_ncf/input/Item_random25.csv'


# 데이터셋 클래스 정의
class CreatorItemDataset(Dataset):
    def __init__(self, creator_df, item_df):
        self.creator_data = creator_df
        self.item_data = item_df
        self.num_users = creator_df['creator_id'].nunique()
        self.num_items = item_df['item_id'].nunique()
        # 임의의 사용자-아이템 상호작용 데이터 생성
        self.data = [(i % self.num_users, j % self.num_items, 1) for i, j in
                     zip(range(len(creator_df)), range(len(item_df)))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id, item_id, target = self.data[idx]
        return {
            'user_id': user_id,
            'item_id': item_id,
            'target': target
        }


# CSV 파일 로드
creator_df = pd.read_csv(creator_file_path)
item_df = pd.read_csv(item_file_path)

# 데이터셋 생성 및 분할
dataset = CreatorItemDataset(creator_df, item_df)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, _ = random_split(dataset, [train_size, test_size])

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 각 모델 테스트용 설정
# 각 모델 테스트용 설정
test_config = {
    'num_users': dataset.num_users,
    'num_items': dataset.num_items,
    'latent_dim': 4,
    'latent_dim_mf': 4,
    'latent_dim_mlp': 4,
    'num_negative': 2,
    'batch_size': 2,
    'num_epoch': 1,
    'optimizer': 'adam',
    'adam_lr': 6e-3,
    'l2_regularization': 0,
    'use_cpu': True,
    'alias': 'test_run',

    # MLP 및 NeuMF 모델에서 사용될 메타데이터 관련 설정
    'num_item_categories': len(item_df['item_category'].unique()),  # 아이템 카테고리 수
    'meta_latent_dim': 4,  # 메타데이터의 임베딩 차원
    'num_channel_categories': len(creator_df['channel_category'].unique())  # 채널 카테고리 수
}

# GMF 모델 테스트
print("Testing GMF model...")
gmf_engine = GMFEngine(test_config)
gmf_engine.train_an_epoch(train_loader, epoch_id=0)

# MLP 모델 테스트
print("\nTesting MLP model...")
mlp_engine = MLPEngine(test_config)
mlp_engine.train_an_epoch(train_loader, epoch_id=0)

# NeuMF 모델 테스트
print("\nTesting NeuMF model...")
neumf_engine = NeuMFEngine(test_config)
neumf_engine.train_an_epoch(train_loader, epoch_id=0)

print("\nAll models tested successfully.")

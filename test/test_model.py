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

        # 필요한 메타데이터 정보 추가
        self.item_data['item_category'] = self.item_data['item_category'].astype("category").cat.codes
        self.item_data['media_type'] = self.item_data['media_type'].map({'short': 0, 'long': 1})
        self.creator_data['channel_category'] = self.creator_data['channel_category'].astype("category").cat.codes
        self.creator_data['subscribers'] = self.creator_data['subscribers'].replace({',': ''}, regex=True).astype(int)

        # 임의의 사용자-아이템 상호작용 데이터 생성
        self.data = [(i % self.num_users, j % self.num_items, 1) for i, j in
                     zip(range(len(creator_df)), range(len(item_df)))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id, item_id, target = self.data[idx]
        item_category = torch.tensor(self.item_data.loc[item_id, 'item_category'], dtype=torch.long)
        media_type = torch.tensor(self.item_data.loc[item_id, 'media_type'], dtype=torch.long)
        channel_category = torch.tensor(self.creator_data.loc[user_id, 'channel_category'], dtype=torch.long)
        subscribers = torch.tensor(self.creator_data.loc[user_id, 'subscribers'], dtype=torch.long)

        return {
            'user_id': user_id,
            'item_id': item_id,
            'target': target,
            'item_category': item_category,
            'media_type': media_type,
            'channel_category': channel_category,
            'subscribers': subscribers
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
    'num_channel_categories': len(creator_df['channel_category'].unique()),
    'max_subscribers': creator_df['subscribers'].max()+100, # 최대 구독자 수
    'layers': [24, 64, 32, 16, 8],  # MLP 레이어 구조 추가
    'weight_init_gaussian': True,
    'use_bachify_eval': False,
    'device_id': 0,
    'pretrain': False,
}

# GMF 모델 테스트
print("Testing GMF model...")
gmf_engine = GMFEngine(test_config)
gmf_engine.train_an_epoch(train_loader, epoch_id=0)

# MLP 모델 테스트
print("\nTesting MLP model...")
mlp_engine = MLPEngine(test_config)
mlp_engine.train_an_epoch_mlp(train_loader, epoch_id=0)

# NeuMF 모델 테스트
print("\nTesting NeuMF model...")
neumf_engine = NeuMFEngine(test_config)
neumf_engine.train_an_epoch_neumf(train_loader, epoch_id=0)

print("\nAll models tested successfully.")

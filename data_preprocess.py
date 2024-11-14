import pickle
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset


# BERT 임베딩을 위한 클래스
class TextEmbedder:
    def __init__(self, model_name='bert-base-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_text_embedding(self, text):
        # 텍스트를 BERT로 임베딩
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=32)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0][0].numpy()


# Dataset 클래스 정의
# 데이터셋 클래스 (UserItemRatingDatasetMLP로 통일)
class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor, item_titles, creator_names,
                 item_category, media_type, channel_category, subscribers, item_category_similarities):
        self.user_tensor = torch.tensor(user_tensor, dtype=torch.long)
        self.item_tensor = torch.tensor(item_tensor, dtype=torch.long)
        self.target_tensor = torch.tensor(target_tensor, dtype=torch.float)
        self.item_titles = item_titles
        self.creator_names = creator_names
        self.item_category = item_category
        self.media_type = media_type
        self.channel_category = channel_category
        self.subscribers = subscribers
        self.item_category_similarities = torch.tensor(item_category_similarities, dtype=torch.float)

        self.text_embedder = TextEmbedder()

        # 임베딩 계산
        self.item_embeddings = [self.text_embedder.get_text_embedding(title) for title in self.item_titles]
        self.creator_embeddings = [self.text_embedder.get_text_embedding(name) for name in self.creator_names]

    def __len__(self):
        return len(self.user_tensor)

    def __getitem__(self, idx):
        item_embedding = self.item_embeddings[idx]
        creator_embedding = self.creator_embeddings[idx]
        item_category_similarity = torch.tensor(self.item_category_similarities[idx], dtype=torch.float).unsqueeze(0)

        # combined_features = torch.cat([
        #     item_embedding,
        #     creator_embedding,
        #     item_category_similarity,  # Use item_category_similarity directly
        #     torch.tensor(self.item_category[idx], dtype=torch.long).unsqueeze(0),
        #     torch.tensor(self.media_type[idx], dtype=torch.long).unsqueeze(0),
        #     torch.tensor(self.channel_category[idx], dtype=torch.long).unsqueeze(0),
        #     torch.tensor(self.subscribers[idx], dtype=torch.float).unsqueeze(0)
        # ], dim=-1)

        return {
            'user_id': self.user_tensor[idx],
            'item_id': self.item_tensor[idx],
            'target': self.target_tensor[idx],
            'item_category': torch.tensor(self.item_category[idx], dtype=torch.long),
            'media_type': torch.tensor(self.media_type[idx], dtype=torch.long),
            'channel_category': torch.tensor(self.channel_category[idx], dtype=torch.long),
            'subscribers': torch.tensor(self.subscribers[idx], dtype=torch.long),
            'item_embedding': torch.tensor(item_embedding, dtype=torch.float),
            'creator_embedding': torch.tensor(creator_embedding, dtype=torch.float),
            'item_category_similarity': item_category_similarity,  # Changed to item_category_similarity
            # 'combined_features': combined_features
        }


class Loader:
    def __init__(self, file_path, similarity_matrix_file):
        self.file_path = file_path
        self.similarity_matrix_file = similarity_matrix_file
        self.similarity_matrix = self.load_similarity_matrix()
        self.text_embedder = TextEmbedder()

        # 변수 초기화 - 모델 설정 시 필요로 하는 각종 메타데이터 변수 추가
        self.num_users = None  # 사용자 수
        self.num_items = None  # 아이템 수
        self.num_item_categories = None  # 아이템 카테고리 수
        self.num_channel_categories = None  # 채널 카테고리 수
        self.max_subscribers = None  # 최대 구독자 수

    def load_similarity_matrix(self):
        return pd.read_csv(self.similarity_matrix_file, index_col=0)

    def load_dataset(self):
        item_df = pd.read_csv(self.file_path + '/Item_random25.csv')
        creator_df = pd.read_csv(self.file_path + '/Creator_random25.csv')

        # 전처리 및 데이터셋 생성
        item_df['item_category'] = item_df['item_category'].astype("category").cat.codes
        item_df['media_type'] = item_df['media_type'].map({'short': 0, 'long': 1})
        item_df['score'] = item_df['score'].astype(float)
        item_df['target'] = item_df['score'].apply(lambda x: 1 if x >= 0.85 else 0)  # target을 0과 1로 설정
        creator_df['channel_category'] = creator_df['channel_category'].astype("category").cat.codes
        creator_df['subscribers'] = creator_df['subscribers'].replace({',': ''}, regex=True).astype(int)

        # 정보 업데이트
        self.num_users = creator_df['creator_id'].nunique() + 10
        self.num_items = item_df['item_id'].nunique() + 10
        self.num_item_categories = item_df['item_category'].nunique()
        self.num_channel_categories = creator_df['channel_category'].nunique()
        self.max_subscribers = creator_df['subscribers'].max() + 1

        item_category_similarities = item_df['item_category'].apply(self.calculate_category_similarity).values
        user_tensor = item_df['item_id'].values
        item_tensor = creator_df['creator_id'].values
        target_tensor = item_df['target'].values  # target_tensor 생성

        return UserItemRatingDataset(
            user_tensor, item_tensor, target_tensor,  # 수정된 부분
            item_titles=item_df['title'].values,
            creator_names=creator_df['channel_name'].values,
            item_category=item_df['item_category'].values,
            media_type=item_df['media_type'].values,
            channel_category=creator_df['channel_category'].values,
            subscribers=creator_df['subscribers'].values,
            item_category_similarities=item_category_similarities
        )

    def calculate_category_similarity(self, category_code):
        # 유사도 매트릭스에서 카테고리 코드에 맞는 유사도 값 반환
        if category_code in self.similarity_matrix.columns:
            return self.similarity_matrix.loc[category_code, category_code]
        return 0.5  # 기본 유사도 값 설정

    def create_bidirectional_data(self, item_data):
        item_user_data = item_data[['item_id', 'item_category_similarity']]
        item_user_data['user_id'] = item_user_data['item_id']
        user_item_data = item_data[['item_id', 'item_category_similarity']]
        user_item_data['user_id'] = user_item_data['item_id']

        bidirectional_data = pd.concat([item_user_data, user_item_data], axis=0)
        return bidirectional_data, bidirectional_data

    def train_test_split(self, item_user_data):
        return train_test_split(item_user_data, test_size=0.2, random_state=42)

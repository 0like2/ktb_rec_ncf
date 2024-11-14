import torch
import pandas as pd
from data_preprocess import Loader
from NeuMF import NeuMFEngine  # 또는 GMFEngine, MLPEngine 사용 가능
from utils import use_cpu


class Recommender:
    def __init__(self, config, model_path):
        self.config = config
        self.device = use_cpu()

        # 학습된 모델 엔진 로드
        self.model_engine = NeuMFEngine(config)
        self.model_engine.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model_engine.model.to(self.device)
        self.model_engine.model.eval()

        # 데이터 로더 초기화
        self.loader = Loader(config['file_path'], config['similarity_matrix_file'])
        self.dataset = self.loader.load_dataset()

    def recommend_for_new_item(self, item_id, top_k=10):
        """새로운 기획서(item)이 들어온 경우 추천할 크리에이터(user) 목록을 생성합니다."""
        item_data = self.dataset[item_id]
        item_id_tensor = torch.tensor([item_id]).to(self.device)
        user_ids_tensor = torch.arange(self.config['num_users']).to(self.device)

        # 아이템 임베딩 데이터를 유지하여 모델에 입력
        with torch.no_grad():
            scores = self.model_engine.model(
                user_ids_tensor, item_id_tensor.repeat(self.config['num_users']),
                item_data['item_category'].repeat(self.config['num_users']),
                item_data['media_type'].repeat(self.config['num_users']),
                item_data['channel_category'].repeat(self.config['num_users']),
                item_data['subscribers'].repeat(self.config['num_users'])
            )
            scores = scores.view(-1).cpu().numpy()

        # 상위 top_k 추천 생성
        top_k_indices = scores.argsort()[-top_k:][::-1]
        recommended_users = user_ids_tensor[top_k_indices].cpu().numpy()
        return recommended_users, scores[top_k_indices]

    def recommend_for_new_user(self, user_id, top_k=10):
        """새로운 크리에이터(user)가 들어온 경우 추천할 기획서(item) 목록을 생성합니다."""
        user_data = self.dataset[user_id]
        user_id_tensor = torch.tensor([user_id]).to(self.device)
        item_ids_tensor = torch.arange(self.config['num_items']).to(self.device)

        # 사용자 임베딩 데이터를 유지하여 모델에 입력
        with torch.no_grad():
            scores = self.model_engine.model(
                user_id_tensor.repeat(self.config['num_items']), item_ids_tensor,
                self.dataset.item_category[:self.config['num_items']],
                self.dataset.media_type[:self.config['num_items']],
                self.dataset.channel_category[:self.config['num_items']],
                self.dataset.subscribers[:self.config['num_items']]
            )
            scores = scores.view(-1).cpu().numpy()

        # 상위 top_k 추천 생성
        top_k_indices = scores.argsort()[-top_k:][::-1]
        recommended_items = item_ids_tensor[top_k_indices].cpu().numpy()
        return recommended_items, scores[top_k_indices]


if __name__ == "__main__":
    # 설정 파일
    config = {
        'num_users': 35,
        'num_items': 34,
        'latent_dim_mf': 4,
        'latent_dim_mlp': 4,
        'meta_latent_dim': 4,
        'layers': [24, 64, 32, 16, 8],
        'file_path': 'input',
        'similarity_matrix_file': 'similarity_matrix.csv'
    }

    # 학습된 모델 경로 설정
    model_path = 'output/neumf_factor8neg4_Epoch4_HR0.7500_NDCG0.7500.model'

    recommender = Recommender(config, model_path)

    # 새롭게 들어온 기획서(item)에 대해 추천할 크리에이터(user)
    new_item_id = 0  # 예시 기획서 ID
    recommended_users, user_scores = recommender.recommend_for_new_item(new_item_id)
    print(f"New Item {new_item_id} 추천 사용자 목록: {recommended_users} (Scores: {user_scores})")

    # 새롭게 들어온 크리에이터(user)에 대해 추천할 기획서(item)
    new_user_id = 1  # 예시 크리에이터 ID
    recommended_items, item_scores = recommender.recommend_for_new_user(new_user_id)
    print(f"New User {new_user_id} 추천 아이템 목록: {recommended_items} (Scores: {item_scores})")

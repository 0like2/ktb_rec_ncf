import torch
from NeuMF import NeuMF
from data_preprocess import Loader, TextEmbedder

# 추천 시스템 클래스
class Recommender:
    def __init__(self, config, model_path):
        self.config = config
        self.device = torch.device("cpu" if config["use_cpu"] else f"cuda:{config['device_id']}")

        # Loader 초기화
        self.loader = Loader(config['file_path'], config['similarity_matrix_file'])

        # 데이터셋 로드
        self.loader.load_dataset()  # 데이터셋을 명시적으로 로드하여 num_users와 num_items 계산

        # 메타 정보 가져오기
        meta_info = self.loader.get_meta_info()
        print("메타 정보 확인:", meta_info)  # 디버깅용 출력
        self.config['num_users'] = meta_info['num_users']
        self.config['num_items'] = meta_info['num_items']

        # NeuMF 모델 초기화
        self.model = NeuMF(self.config)

        # 모델 파라미터 로드
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # 모델 평가 모드로 설정

        # TextEmbedder 초기화
        self.text_embedder = TextEmbedder()

    def preprocess_new_item(self, data):
        """
        새로운 아이템 데이터 전처리
        """
        data['item_id'] = self.config['num_items'] + 1  # 새로운 임시 ID
        data['item_category'] = self.loader.similarity_matrix.columns.tolist().index(data['item_category'])
        data['media_type'] = 0 if 'short' in data['media_type'].lower() else 1
        data['item_embedding'] = torch.tensor(self.text_embedder.get_text_embedding(data['title']), dtype=torch.float)
        return data

    def recommend_for_new_item(self, item_data, top_k=10):
        """
        새로운 아이템 데이터에 대해 사용자 추천
        """
        item_data = self.preprocess_new_item(item_data)  # 아이템 데이터 전처리
        item_id_tensor = torch.tensor([item_data['item_id']], dtype=torch.long).to(self.device)
        user_ids_tensor = torch.arange(self.config['num_users'], dtype=torch.long).to(self.device)

        with torch.no_grad():
            scores = self.model(
                user_ids_tensor,
                item_id_tensor.repeat(self.config['num_users']),
                torch.tensor([item_data['item_category']], dtype=torch.long).repeat(self.config['num_users']).to(self.device),
                torch.tensor([item_data['media_type']], dtype=torch.long).repeat(self.config['num_users']).to(self.device),
                item_data['item_embedding'].repeat(self.config['num_users'], 1).to(self.device)
            )
            scores = scores.view(-1).cpu().numpy()

        top_k_indices = scores.argsort()[-top_k:][::-1]
        recommended_users = user_ids_tensor[top_k_indices].cpu().numpy()
        return recommended_users, scores[top_k_indices]


if __name__ == "__main__":
    file_path = 'input'
    similarity_matrix_file = 'similarity_matrix.csv'

    # Loader 초기화 및 데이터 로드
    loader = Loader(file_path, similarity_matrix_file)
    loader.load_dataset()  # 데이터셋 로드

    # 메타 정보 가져오기
    meta_info = loader.get_meta_info()

    # Config 설정
    config = {
        'num_users': meta_info['num_users'],  # 사용자 수
        'num_items': meta_info['num_items'],  # 아이템 수
        'num_item_categories': meta_info['num_item_categories'],  # 아이템 카테고리 수 추가
        'num_channel_categories': meta_info['num_channel_categories'],  # 채널 카테고리 수 추가
        'max_subscribers': int(meta_info['max_subscribers']),  # 구독자 수 최대값 추가
        'file_path': file_path,
        'similarity_matrix_file': similarity_matrix_file,

        'latent_dim_mf': 4,  # MF 임베딩 크기
        'latent_dim_mlp': 4,  # MLP 임베딩 크기
        'meta_latent_dim': 4,  # 추가 메타데이터 임베딩 크기
        'layers': [64, 32, 16, 8],  # MLP 레이어 크기

        'batch_size': 256,  # 배치 크기
        'num_epoch': 20,  # 학습 에포크 수
        'learning_rate': 0.001,  # 학습률
        'l2_regularization': 1e-6,  # L2 정규화
        'optimizer': 'adam',  # 최적화 알고리즘

        'use_cpu': True,  # CPU 사용 여부
        'device_id': 0,  # GPU 디바이스 ID
        'num_negative': 4,  # Negative sampling 비율
        'weight_init_gaussian': False,  # 가우시안 초기화 여부
    }


    # 학습된 모델 경로
    model_path = "output/neumf_factor8neg4_Epoch4_HR1.0000_NDCG1.0000.model"

    # Recommender 초기화
    recommender = Recommender(config, model_path)

    # 새로운 아이템 데이터 예시
    new_item_data = {
        'title': "바밤바를 뛰어넘는 밤 맛 과자가 있을까?",
        'item_category': '음식 리뷰',
        'media_type': 'short',
        'score': 80,
        'item_content': '다양한 밤 맛 과자를 비교하며 맛과 질감을 리뷰하는 콘텐츠'
    }

    # 추천 수행
    recommended_users, user_scores = recommender.recommend_for_new_item(new_item_data)
    print(f"추천 사용자 목록: {recommended_users} (Scores: {user_scores})")

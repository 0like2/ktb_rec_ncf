from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from data_preprocess import TextEmbedder
import os


categories = [
    'beauty', 'car_auto', 'celeb', 'economy', 'education', 'entertainment',
    'food_cooking', 'game', 'government', 'hobbies', 'kids', 'life_style',
    'movie', 'music', 'news', 'pet', 'sports_health', 'tech', 'travel'
]

load_dotenv()
# OpenAI API Key: Retrieve from environment variable or replace with actual key
API_KEY = os.getenv('OPENAI_API_KEY')

embedder = TextEmbedder(api_key=API_KEY)

def get_category_embedding(category, embedder):
    """
    카테고리를 임베딩 벡터로 변환하는 함수.

    Args:
        category (str): 임베딩할 카테고리 이름.
        embedder (TextEmbedder): TextEmbedder 클래스의 인스턴스.

    Returns:
        list: 카테고리 임베딩 벡터.
    """
    try:
        embedding = embedder.get_text_embedding(category)
        return embedding
    except Exception as e:
        print(f"Error encoding category '{category}': {e}")
        raise e

def get_similarity_matrix(categories, embedder):

    # 각 카테고리 임베딩 계산
    embeddings = np.array([get_category_embedding(category, embedder) for category in categories])

    # 유사도 매트릭스 계산
    similarity_matrix = np.zeros((len(categories), len(categories)))

    for i, embedding1 in enumerate(embeddings):
        for j, embedding2 in enumerate(embeddings):
            if i != j:
                similarity_score = cosine_similarity([embedding1], [embedding2])
                similarity_matrix[i][j] = round(similarity_score[0][0], 2)

    # 유사도 매트릭스를 DataFrame으로 변환
    similarity_df = pd.DataFrame(similarity_matrix, columns=categories, index=categories)
    return similarity_df

def log_scaling(similarity_matrix, factor=10):
    return np.log1p(similarity_matrix / factor)

def sigmoid_scaling(similarity_matrix, factor=10):

    return 1 / (1 + np.exp(-similarity_matrix * factor))

def min_max_scaling(similarity_matrix):
    min_val = np.min(similarity_matrix)
    max_val = np.max(similarity_matrix)
    scaled_matrix = (similarity_matrix - min_val) / (max_val - min_val)
    return scaled_matrix



def plot_similarity_distribution(similarity_values, title):

    plt.hist(similarity_values, bins=50, edgecolor='k')
    plt.title(title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    print("Starting similarity matrix computation...")

    # 유사도 매트릭스 생성
    similarity_df = get_similarity_matrix(categories, embedder)
    print("Similarity matrix:")
    print(similarity_df)

    # 원본 매트릭스 저장
    similarity_df.to_csv('similarity_matrix.csv')
    print("Original similarity matrix saved to 'similarity_matrix.csv'.")

    original_values = similarity_df.values.flatten()

    # Min-Max Scaling
    min_max_scaled_matrix = min_max_scaling(similarity_df.values)
    min_max_values = min_max_scaled_matrix.flatten()


    # 스케일링된 매트릭스 저장
    pd.DataFrame(min_max_scaled_matrix, columns=categories, index=categories).to_csv('similarity_matrix.csv')
    print("min_max_scaled_matrix saved to 'similarity_matrix.csv'.")



    # 분포 시각화
    plot_similarity_distribution(similarity_df.values.flatten(), "Original Cosine Similarity Distribution")
    plot_similarity_distribution(min_max_scaled_matrix.flatten(), "min_max_scaled_matrix Distribution")


    print("All tasks completed successfully.")
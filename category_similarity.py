from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special

MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
categories = [
    'beauty', 'car_auto', 'celeb', 'economy', 'education', 'entertainment',
    'food_cooking', 'game', 'government', 'hobbies', 'kids', 'life_style',
    'movie', 'music', 'news', 'pet', 'sports_health', 'tech', 'travel'
]


def get_category_embedding(category, tokenizer, model):

    inputs = tokenizer(category, return_tensors='pt', padding=True, truncation=True, max_length=32)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state[0][0].numpy()


def get_category_similarity(category1, category2, tokenizer, model):
    embedding1 = get_category_embedding(category1, tokenizer, model)
    embedding2 = get_category_embedding(category2, tokenizer, model)

    similarity = cosine_similarity([embedding1], [embedding2])
    return similarity[0][0]


def get_similarity_matrix(categories, tokenizer, model):
    embeddings = np.array([get_category_embedding(category, tokenizer, model) for category in categories])

    """
    # Apply PCA to reduce the dimensionality of embeddings
    reduced_embeddings = apply_pca_to_embeddings(embeddings, n_components=19)
    """
    similarity_matrix = np.zeros((len(categories), len(categories)))

    for i, category1 in enumerate(categories):
        for j, category2 in enumerate(categories):
            if i != j:
                similarity_score = cosine_similarity([embeddings[i]], [embeddings[j]])
                similarity_matrix[i][j] = round(similarity_score[0][0], 2)

    similarity_df = pd.DataFrame(similarity_matrix, columns=categories, index=categories)
    return similarity_df

def log_scaling(similarity_matrix, factor=10):
    # log 변환을 통해 값 확장 (log(1 + similarity) 형태로)
    scaled_matrix = np.log1p(similarity_matrix / factor)
    return scaled_matrix

# Softmax 함수
def softmax_scaling(similarity_matrix):
    # Softmax를 적용하여 유사도를 0-1 사이로 확장
    softmax_matrix = scipy.special.softmax(similarity_matrix, axis=1)
    return softmax_matrix

# Sigmoid 변환 함수
def sigmoid_scaling(similarity_matrix, factor=10):
    # Sigmoid 함수로 값을 0과 1 사이로 압축
    scaled_matrix = 1 / (1 + np.exp(-similarity_matrix * factor))  # factor는 변환 강도를 조정
    return scaled_matrix

# Log(1 + x) 후 지수 확장 함수 정의
def log_exponentiation_scaling(similarity_matrix, factor=10):
    # log(1+x) 후 지수 확장
    log_scaled = np.log1p(similarity_matrix)  # log(1+x) 변환
    exp_scaled = np.exp(log_scaled / factor)  # 지수 변환
    return exp_scaled


print("Starting similarity matrix computation...")
similarity_df = get_similarity_matrix(categories, tokenizer, model)
print(similarity_df.head())
# 변환된 유사도 매트릭스를 CSV로 저장
similarity_df.to_csv('sigmoid_scaled_similarity_matrix.csv')
print(f"Similarity matrix saved to sigmoid_scaled_similarity_matrix.csv")

# 유사도 분포를 히스토그램으로 시각화
similarity_values = similarity_df.values.flatten()
plt.hist(similarity_values, bins=50, edgecolor='k')
plt.title("Distribution of Log(1+x) and Exponentially Scaled Cosine Similarities (Sigmoid)")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.show()

print("Done!")




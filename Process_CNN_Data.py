import pandas as pd
import re
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from collections import Counter
import joblib

df_final = pd.read_csv('data/dataset_final_treinamento.csv')
df_final = df_final.dropna(subset=['Texto']).drop_duplicates(subset=['Texto'])
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
df_final = df_final.drop('Fonte', axis=1)

def clear_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df_final['Texto'] = df_final['Texto'].apply(clear_text)

all_words = Counter()
for text in df_final['Texto']:
    all_words.update(text.split())

vocab_dataset = {word for word, count in all_words.items() if count >= 1}

path = hf_hub_download(repo_id="nilc-nlp/fasttext-cbow-100d", filename="embeddings.safetensors")
vocab_path = hf_hub_download(repo_id="nilc-nlp/fasttext-cbow-100d", filename="vocab.txt")

full_vectors = load_file(path)["embeddings"]
with open(vocab_path) as f:
    full_vocab_list = [w.strip() for w in f]

filtered_vectors = []
word_to_index = {"<PAD>": 0}

filtered_vectors.append(np.zeros(full_vectors.shape[1])) # Vetor de zeros para o PAD

for i, word in enumerate(full_vocab_list):
    if word in vocab_dataset:
        word_to_index[word] = len(word_to_index)
        filtered_vectors.append(full_vectors[i])

embedding_matrix = np.array(filtered_vectors).astype('float32')
joblib.dump(word_to_index, 'data/word_to_index.pkl')

word_to_vec_small = {word: embedding_matrix[idx] for word, idx in word_to_index.items()}
joblib.dump(word_to_vec_small, 'data/word_to_vec_small.pkl')
del full_vectors
del full_vocab_list

MAX_LEN = 300
num_samples = len(df_final)

X_cnn = np.zeros((num_samples, MAX_LEN), dtype='int32')

for idx, text in enumerate(df_final['Texto']):
    words = text.split()[:MAX_LEN]
    indices = [word_to_index[w] for w in words if w in word_to_index]
    X_cnn[idx, :len(indices)] = indices

y = df_final['Label'].values.astype('float32')

# 5. Salvamento
np.save('data/X_cnn_sequences.npy', X_cnn)
np.save('data/y_labels_cnn.npy', y)
np.save('data/embedding_matrix.npy', embedding_matrix)


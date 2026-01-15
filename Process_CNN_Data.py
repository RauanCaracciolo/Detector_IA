import pandas as pd
import re
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

df_humano = pd.read_csv("data/dataset_humano.csv")
df_ia = pd.read_csv("data/dataset_ia.csv")

df_ia = df_ia.drop(columns=["Prompt"], errors='ignore').rename(columns={'Texto_Fragmentado': 'Texto'})
df_final = pd.concat([df_ia, df_humano[['Texto', 'Label']]], ignore_index=True)
df_final = df_final.dropna(subset=['Texto']).drop_duplicates(subset=['Texto'])
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)


def clear_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


df_final['Texto'] = df_final['Texto'].apply(clear_text)

path = hf_hub_download(repo_id="nilc-nlp/fasttext-cbow-100d", filename="embeddings.safetensors")
vocab_path = hf_hub_download(repo_id="nilc-nlp/fasttext-cbow-100d", filename="vocab.txt")

vectors = load_file(path)["embeddings"]
with open(vocab_path) as f:
    vocab_list = [w.strip() for w in f]

word_to_index = {word: i for i, word in enumerate(vocab_list)}

MAX_LEN = 300  # Número máximo de palavras por texto


def text_to_indices(text, word_to_index, max_len):
    words = text.split()
    indices = [word_to_index[w] for w in words if w in word_to_index]

    indices = indices[:max_len]

    if len(indices) < max_len:
        indices.extend([0] * (max_len - len(indices)))

    return np.array(indices)


X_cnn = np.array([text_to_indices(t, word_to_index, MAX_LEN) for t in df_final['Texto']])
y = df_final['Label'].values

np.save('data/X_cnn_sequences.npy', X_cnn)
np.save('data/y_labels_cnn.npy', y)
np.save('data/embedding_matrix.npy', vectors)

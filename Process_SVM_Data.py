
import pandas as pd
import re
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

# Carregamento e limpeza dos dados
df_humano = pd.read_csv("data/dataset_humano.csv")
df_ia = pd.read_csv("data/dataset_ia.csv")

df_ia = df_ia.drop(columns=["Prompt"], errors='ignore').rename(columns={'Texto_Fragmentado': 'Texto'})
df_final = pd.concat([df_ia, df_humano[['Texto', 'Label']]], ignore_index=True)
df_final = df_final.dropna(subset=['Texto']).sample(frac=1, random_state=42).reset_index(drop=True)


def clear_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


df_final['Texto'] = df_final['Texto'].apply(clear_text)

# Carregar e aplicar os embendings prontos
path = hf_hub_download(repo_id="nilc-nlp/fasttext-cbow-100d", filename="embeddings.safetensors")
vocab_path = hf_hub_download(repo_id="nilc-nlp/fasttext-cbow-100d", filename="vocab.txt")

vectors = load_file(path)["embeddings"]
with open(vocab_path) as f:
    vocab = [w.strip() for w in f]

# Criar dicionario de busca rapida
word_to_vec = {word: vectors[i] for i, word in enumerate(vocab)}
dim = vectors.shape[1]  # 100


# Transformar o texto em vetores de tamanho fixo para o VSM
def get_mean_vector(text):
    words = text.split()
    valid_vecs = [word_to_vec[w] for w in words if w in word_to_vec]

    if not valid_vecs:
        return np.zeros(dim)

    return np.mean(valid_vecs, axis=0)


# Criar a matriz X para o SVM (n_exemplos, 100)
X_svm = np.array([get_mean_vector(t) for t in df_final['Texto']])
y = df_final['Label'].values

# Salvar
# X_svm e y em formato binário NumPy (.npy) para uso rápido depois
np.save('data/X_svm.npy', X_svm)
np.save('data/y_labels.npy', y)



import pandas as pd
import re
import numpy as np
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

print("📥 Carregando vetores filtrados...")
try:
    # Usa o arquivo gerado pelo Process_CNN_Data.py
    word_to_vec = joblib.load('data/word_to_vec_small.pkl')
    dim = 100
except FileNotFoundError:
    print("❌ Erro: Execute o Process_CNN_Data.py primeiro para gerar o word_to_vec_small.pkl")
    exit()
# ----------------------------------------------------------

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


import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import re
import os

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Detector de IA", page_icon="🤖", layout="centered")


# --- DEFINIÇÃO DA ARQUITETURA (Igual ao RN_Model.py) ---
class CNNDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CNNDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Não carregamos os pesos aqui, o load_state_dict cuidará disso
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc2(x))


# --- CARREGAMENTO DE RECURSOS (Com Cache) ---
@st.cache_resource
def load_resources():
    # 1. Carregar mapeamentos (Devem ser os mesmos do treino para evitar mismatch)
    # Se você ainda não salvou esses arquivos, veja a nota abaixo do código.
    word_to_index = joblib.load('data/word_to_index.pkl')
    word_to_vec = joblib.load('data/word_to_vec_small.pkl')

    # 2. Carregar Modelo SVM
    svm_model = joblib.load('modelo_svm_detector.pkl')

    # 3. Inicializar e Carregar CNN
    vocab_size = len(word_to_index)
    embedding_dim = 100
    cnn_model = CNNDetector(vocab_size, embedding_dim)

    # Carregar pesos treinados
    checkpoint = torch.load('detector_ia_cnn.pth', map_location=torch.device('cpu'))
    cnn_model.load_state_dict(checkpoint)
    cnn_model.eval()

    return word_to_vec, word_to_index, svm_model, cnn_model


# Tenta carregar os recursos
try:
    word_to_vec, word_to_index, svm_model, cnn_model = load_resources()
except FileNotFoundError:
    st.error(
        "🚨 Arquivos de modelo ou vocabulário não encontrados! Certifique-se de que 'word_to_index.pkl', 'word_to_vec_small.pkl' e os modelos estão na pasta correta.")
    st.stop()

# --- INTERFACE ---
st.title("🤖 Detector de Textos: Humano vs IA")
st.markdown("Analise se um texto foi gerado por inteligência artificial ou escrito por um humano.")


def clean(text):
    text = str(text).lower()
    return re.sub(r"[^a-zA-Z\s]", "", text)


user_input = st.text_area("Digite ou cole o parágrafo aqui:", height=200,
                          placeholder="Ex: No meio do caminho tinha uma pedra...")

if st.button("🚀 Analisar Texto"):
    if not user_input.strip():
        st.warning("⚠️ Por favor, digite algum texto antes de analisar.")
    else:
        cleaned_text = clean(user_input)
        words = cleaned_text.split()

        # --- LÓGICA SVM ---
        vecs = [word_to_vec[w] for w in words if w in word_to_vec]
        if vecs:
            svm_input = np.mean(vecs, axis=0).reshape(1, -1)
            svm_pred = svm_model.predict(svm_input)[0]
            svm_res = "🤖 IA" if svm_pred == 1 else "👨‍💻 Humano"
        else:
            svm_res = "❓ Indeterminado"

        # --- LÓGICA CNN ---
        all_indices = [word_to_index[w] for w in words if w in word_to_index]
        max_len = 300
        cnn_res = "❓ Indeterminado"
        confianca = 0.0

        if all_indices:
            # Divide o texto em blocos de 300 palavras
            chunks = [all_indices[i:i + max_len] for i in range(0, len(all_indices), max_len)]
            chunk_probs = []

            for chunk in chunks:
                if len(chunk) < 5: continue  # Ignora trechos muito curtos

                # Padding manual para o tamanho esperado pelo modelo
                padded_chunk = chunk + [0] * (max_len - len(chunk))
                chunk_input = torch.LongTensor([padded_chunk])

                with torch.no_grad():
                    prob = cnn_model(chunk_input).item()
                    chunk_probs.append(prob)

            if chunk_probs:
                # Média das probabilidades de todos os chunks
                prob_final = np.mean(chunk_probs)

                if prob_final > 0.5:
                    cnn_res = "🤖 IA"
                    confianca = prob_final
                else:
                    cnn_res = "👨‍💻 Humano"
                    confianca = 1 - prob_final

        # --- EXIBIÇÃO DE RESULTADOS ---
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Análise SVM")
            st.metric("Resultado", svm_res)
            st.caption("Foca no vocabulário e palavras-chave.")

        with col2:
            st.subheader("Análise CNN")
            st.metric("Resultado", cnn_res, f"{confianca:.2%} de confiança")
            st.caption("Foca nos padrões e ritmo do texto.")

        st.divider()
        if svm_res == cnn_res and "Indeterminado" not in svm_res:
            st.success(f"✅ **Consenso:** Ambos os modelos indicam que o texto é de origem **{svm_res}**.")
        elif "Indeterminado" in svm_res:
            st.warning("⚠️ O vocabulário é muito restrito para uma análise conclusiva.")
        else:
            st.info(
                "💡 **Divergência:** Os modelos discordam. Isso é comum em textos curtos ou humanos que usam linguagem muito formal.")
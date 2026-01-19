import streamlit as st
import torch
import joblib
import numpy as np
import re
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

st.set_page_config(page_title="Detector de IA", page_icon="🤖")
st.title("🤖 Detector de Textos: Humano vs IA")
st.markdown("Insira um parágrafo para verificar a origem do texto.")


@st.cache_resource
def load_resources():
    path = hf_hub_download(repo_id="nilc-nlp/fasttext-cbow-100d", filename="embeddings.safetensors")
    vocab_path = hf_hub_download(repo_id="nilc-nlp/fasttext-cbow-100d", filename="vocab.txt")
    vectors = load_file(path)["embeddings"]
    with open(vocab_path) as f:
        vocab = [w.strip() for w in f]
    word_to_index = {word: i for i, word in enumerate(vocab)}
    word_to_vec = {word: vectors[i] for i, word in enumerate(vocab)}

    svm_model = joblib.load('modelo_svm_detector.pkl')

    class CNNDetector(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.conv = torch.nn.Conv1d(embedding_dim, 128, 5)
            self.pool = torch.nn.AdaptiveMaxPool1d(1)
            self.fc1 = torch.nn.Linear(128, 64)
            self.fc2 = torch.nn.Linear(64, 1)

        def forward(self, x):
            x = self.embedding(x).permute(0, 2, 1)
            x = torch.relu(self.conv(x))
            x = self.pool(x).squeeze(-1)
            return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))

    cnn_model = CNNDetector(vectors.shape[0], vectors.shape[1])
    cnn_model.load_state_dict(torch.load('detector_ia_cnn.pth', map_location=torch.device('cpu')))
    cnn_model.eval()

    return word_to_vec, word_to_index, svm_model, cnn_model


word_to_vec, word_to_index, svm_model, cnn_model = load_resources()


def clean(text):
    text = str(text).lower()
    return re.sub(r"[^a-zA-Z\s]", "", text)


user_input = st.text_area("Digite o texto aqui:", height=200)

if st.button("Analisar Texto"):
    if user_input.strip() == "":
        st.warning("Por favor, digite algum texto.")
    else:
        cleaned_text = clean(user_input)
        words = cleaned_text.split()

        vecs = [word_to_vec[w] for w in words if w in word_to_vec]
        if vecs:
            svm_input = np.mean(vecs, axis=0).reshape(1, -1)
            svm_pred = svm_model.predict(svm_input)[0]
            svm_res = "🤖 IA" if svm_pred == 1 else "👨‍💻 Humano"
        else:
            svm_res = "Indeterminado"

        all_indices = [word_to_index[w] for w in words if w in word_to_index]
        max_len = 300
        cnn_res = "Indeterminado"
        confianca = 0.0

        if all_indices:
            chunks = [all_indices[i:i + max_len] for i in range(0, len(all_indices), max_len)]
            chunk_probs = []

            for chunk in chunks:
                temp_chunk = list(chunk)
                if len(temp_chunk) < 5: continue

                if len(temp_chunk) < max_len:
                    temp_chunk.extend([0] * (max_len - len(temp_chunk)))

                chunk_input = torch.LongTensor([temp_chunk])
                with torch.no_grad():
                    p = cnn_model(chunk_input).item()
                    chunk_probs.append(p)

            if chunk_probs:
                prob_final = np.max(chunk_probs)

                if prob_final > 0.5:
                    cnn_res = "🤖 IA"
                    confianca = prob_final
                else:
                    cnn_res = "👨‍💻 Humano"
                    confianca = 1 - prob_final

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Resultado SVM", svm_res)
        with col2:
            st.metric("Resultado CNN", cnn_res, f"{confianca:.2%} de confiança")

        if svm_res == cnn_res:
            st.success(f"Consenso: O texto parece ser {svm_res}.")
        else:
            st.info("Divergência: A CNN analisa o ritmo, enquanto o SVM foca no vocabulário.")
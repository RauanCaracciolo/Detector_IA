# 🤖 Detector de Textos: Humano vs IA

Este projeto é uma aplicação web desenvolvida com **Streamlit** que utiliza inteligência artificial para identificar se um texto foi escrito por um humano ou gerado por modelos de linguagem (IA). A solução combina duas abordagens de Machine Learning para oferecer uma análise mais robusta: Processamento de Linguagem Natural (NLP) clássico e Redes Neurais Convolucionais (CNN).

## 🚀 Demonstração
O projeto utiliza um sistema de consenso entre dois modelos:
1. **SVM (Support Vector Machine):** Focado na análise do vocabulário e frequência de palavras.
2. **CNN (Convolutional Neural Network):** Focado na captura de padrões sequenciais e ritmo do texto.

## 🛠️ Tecnologias Utilizadas
* **Python 3.12+**
* **PyTorch**: Framework para a Rede Neural (CNN).
* **Scikit-Learn**: Para o modelo clássico SVM e métricas de avaliação.
* **FastText (NILC)**: Embeddings pré-treinados para representação vetorial das palavras em português.
* **Streamlit**: Interface web interativa.
* **Git LFS**: Gerenciamento de arquivos de modelos pesados.

## 📊 Arquitetura dos Modelos

### CNN (Rede Neural Convolucional)
A rede processa sequências de até 300 palavras.
* **Camada de Embedding**: Pesos fixos baseados no FastText.
* **Conv1d**: Filtros para capturar n-gramas e padrões locais.
* **AdaptiveAvgPool1d**: Redução de dimensionalidade mantendo as características principais.
* **Dense Layers**: Camadas totalmente conectadas com Dropout para evitar overfitting.



### SVM (Classical ML)
Utiliza a média dos vetores das palavras (Mean Word Embeddings) para classificar o texto com base no espaço vetorial semântico.

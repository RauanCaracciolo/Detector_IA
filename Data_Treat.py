import pandas as pd
import re
df_humano = pd.read_csv("data/dataset_humano.csv")
df_ia = pd.read_csv("data/dataset_ia.csv")

df_ia = df_ia.drop(columns= ["Prompt"], axis = 1)
df_ia = df_ia.rename(columns = {'Texto_Fragmentado': 'Texto'})

colunas_finais = ['Texto', 'Label']
df_humano = df_humano[colunas_finais]
df_ia = df_ia[colunas_finais]

df_final = pd.concat([df_ia, df_humano], ignore_index = True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
df_final = df_final.dropna(subset=['Texto'])
def clear_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]","", text)
    return text
#Aplly the function to all rows.
df_final['Texto'] = df_final['Texto'].apply(clear_text)


df_final.to_csv('dataset_final.csv')
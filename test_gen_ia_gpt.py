import pandas as pd
import requests
import json
import time

from helpers import create_request_payload, send_openai_request

# Definir a função para classificar o movimento processual
def classify_movement(movement_name):
    prompt = f"""
    Analise o movimento do processo fornecido e determine se ele constitui uma sentença judicial. 
    Classifique o movimento como '1' se for uma sentença e '0' se não for. 
    A resposta deve ser um número, '1' para sentença e '0' para não sentença, não traga nenhuma informação além do número de classificação.
    
    Movimento: {movement_name}
    """
    data = create_request_payload("Você é um especialista em análise de movimentação de processos judiciais, focado em identificar se um movimento é uma sentença ou não.", prompt)
    response = send_openai_request(data)
    print(response)
    return response.strip()

# Exemplo de dataset
dataset = pd.read_csv('database\\teste_base_gpt.csv',sep='|')

# Listas para armazenar os resultados
movements = []
classifications = []

# Iterar sobre o dataset e classificar cada movimento
for i,item in dataset.iterrows():
    movement_name = item["Nome"]
    classified_as = classify_movement(movement_name)
    movements.append(movement_name)
    classifications.append(classified_as)

# Criar um DataFrame com os resultados
results_df = pd.DataFrame({
    "Tribunal": [item["Tribunal"] for i,item in dataset.iterrows()],
    "Codigo": [item["Codigo"] for i,item in dataset.iterrows()],
    "Nome": movements,
    "is_sentenca": [item["is_sentenca"] for i,item in dataset.iterrows()],
    "classificacao_gpt": classifications
})

# Salvar o DataFrame em um arquivo CSV
results_df.to_csv("resultados_classificacao_gpt.csv", index=False,sep='|')

print("Resultados salvos em resultados_classificacao_gpt.csv")

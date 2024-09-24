import string
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from unicodedata import normalize
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
import joblib
import nltk
import datetime
import requests
import json
import time
from helpers import create_request_payload, send_openai_request

# Funções auxiliares
def remove_punctuation_and_preprocess(txt):
    if isinstance(txt, float):
        return ''
    txt = ''.join([char for char in txt if char not in string.punctuation])
    txt = txt.replace('ç', 'c')
    txt = txt.replace("\r\n", " ")
    txt = txt.replace("  ", " ")
    txt = txt.lower()
    txt = normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')
    return txt

def load_nltk_resources():
    print(10 * "#", ' - Preparação do NLTK - ', 10 * "#")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('rslp')
    print(10 * "#", ' - NLTK DONE - ', 10 * "#")

def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path, sep='|')
        return df
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV: {e}")
        return None

def classify_movement(movement_name):
    prompt = f"""
    Analise o movimento do processo fornecido e determine se ele constitui uma sentença judicial. 
    Classifique o movimento como '1' se for uma sentença e '0' se não for. 
    A resposta deve ser um número, '1' para sentença e '0' para não sentença, não traga nenhuma informação além do número de classificação.
    
    Movimento: {movement_name}
    """
    data = create_request_payload("Você é um especialista em análise de movimentação de processos judiciais, focado em identificar se um movimento é uma sentença ou não.", prompt)
    response = send_openai_request(data)
    return response.strip()

def main():
    print(10 * "#", ' - Programa iniciado - ', 10 * "#")

    load_nltk_resources()

    print(10 * "#", ' - Leitura iniciada - ', 10 * "#")
    df = read_csv_file('database\\resultado_tageado.csv')
    if df is None:
        return 'CSV vazio!'
    print(10 * "#", ' - Leitura finalizada - ', 10 * "#")

    print(10 * "#", ' - Preprocessamento iniciado - ', 10 * "#")
    
    df['texto_processado'] = df['Nome'].apply(remove_punctuation_and_preprocess)

    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(df['texto_processado'])
    y = df['is_sentenca']

    joblib.dump(tfidf_vectorizer, f'models\\tfidf.pkl')

    # Repartir o dataset 70% treino e 30% teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    print(10 * "#", ' - Preprocessamento finalizado - ', 10 * "#")

    models = [
        LogisticRegression(random_state=42, max_iter=1000),
        SVC(random_state=42),
        GradientBoostingClassifier(random_state=42),
        KNeighborsClassifier()
    ]

    model_names = [
        'Logistic Regression',
        'SVC',
        'Gradient Boosting',
        'KNN'
    ]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Rodar os modelos múltiplas vezes
    num_runs = 5
    results = []

    for run in range(num_runs):
        print(f'\n{10 * "#"} - Rodada {run + 1} - {10 * "#"}')

        for model, model_name in zip(models, model_names):
            print('\n')
            print(10 * "#", f' - {model_name} iniciado - ', 10 * "#")

            # Chunk the data into smaller portions
            chunk_size = 1000
            for i in range(0, X_resampled.shape[0], chunk_size):
                X_chunk = X_resampled[i:i + chunk_size]
                y_chunk = y_resampled[i:i + chunk_size]

                # Verifique se o chunk contém ambas as classes
                if len(set(y_chunk)) < 2:
                    print(f"Chunk {i // chunk_size + 1} contém apenas uma classe. Pulando...")
                    continue

                scores = cross_val_score(model, X_chunk, y_chunk, cv=5, scoring='accuracy')

                print(f'Accuracy para {model_name} (chunk {i // chunk_size + 1}): {np.mean(scores)} (+/- {np.std(scores)})')

            model.fit(X_resampled, y_resampled)

            model_file_name = f'models\\{model_name.lower().replace(" ", "_")}_is_sentenca_run{run + 1}.pkl'
            joblib.dump(model, model_file_name)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred)

            results.append({
                'Model': model_name,
                'Run': run + 1,
                'Accuracy': accuracy,
                'Classification Report': classification_rep
            })

            print(f'Accuracy para {model_name}: {accuracy}')
            print(f'Classification Report:\n{classification_rep}')

    # Classificação usando GPT com o mesmo dataset
    df_gpt = read_csv_file('database\\resultado_tageado.csv') # Usar o mesmo dataset para o GPT
    df_gpt['classificacao_gpt'] = df_gpt['Nome'].apply(classify_movement)

    # Comparar resultados do GPT com os dados de treinamento
    comparison_df = pd.DataFrame({
        "Nome": df_gpt['Nome'],
        "is_sentenca": df_gpt['is_sentenca'],
        "classificacao_gpt": df_gpt['classificacao_gpt']
    })

    comparison_df.to_csv("resultados_classificacao_gpt.csv", index=False, sep='|')
    print("Resultados salvos em resultados_classificacao_gpt.csv")

    # Salvando os resultados dos modelos em um arquivo txt
    with open('resultados_modelos.txt', 'w') as f:
        for result in results:
            f.write(f"\n{10 * '#'} - {result['Model']} - Run {result['Run']} - {10 * '#'}\n")
            f.write(f"Accuracy: {result['Accuracy']}\n")
            f.write(f"Classification Report:\n{result['Classification Report']}\n")
            f.write(f"{10 * '#'} - {result['Model']} finalizado - {10 * '#'}\n")

if __name__ == "__main__":
    main()

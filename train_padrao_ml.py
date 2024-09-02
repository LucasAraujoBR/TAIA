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


def remove_punctuation_and_preprocess(txt):
    if isinstance(txt, float):
        return ''

    # Remoção de pontuações
    txt = ''.join([char for char in txt if char not in string.punctuation])

    # Pré-processamento adicional
    txt = txt.replace('ç', 'c')
    txt = txt.replace("\r\n", " ")
    txt = txt.replace("  ", " ")
    txt = txt.lower()

    # Remoção de acentos
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

def main():
    print(10 * "#", ' - Programa iniciado - ', 10 * "#")

    load_nltk_resources()

    print(10 * "#", ' - Leitura iniciada - ', 10 * "#")
    df = read_csv_file(r'Sentencas/SG_1912.CSV')
    if df is None:
        return 'CSV vazio!'
    print(10 * "#", ' - Leitura finalizada - ', 10 * "#")

    print(10 * "#", ' - Preprocessamento iniciado - ', 10 * "#")
    df['texto_processado'] = df['texto'].apply(remove_punctuation_and_preprocess)

    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(df['texto_processado'])
    y = df['is_sentence']

    joblib.dump(tfidf_vectorizer, f'models/tfidf-{datetime.datetime.now()}.pkl')

    sample_size = int(0.25 * len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=sample_size, stratify=y,
                                                        random_state=42)
    print(10 * "#", ' - Preprocessamento finalizado - ', 10 * "#")

    models = [
        # RandomForestClassifier(random_state=42),
        LogisticRegression(random_state=42, max_iter=1000),
        SVC(random_state=42),
        GradientBoostingClassifier(random_state=42),
        KNeighborsClassifier()
    ]

    model_names = [
        'Random Forest',
        'Logistic Regression',
        'SVC',
        'Gradient Boosting',
        'KNN'
    ]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    for model, model_name in zip(models, model_names):
        print('\n')
        print(10 * "#", f' - {model_name} iniciado - ', 10 * "#")

        # Chunk the data into smaller portions
        chunk_size = 1000
        for i in range(0, len(X_resampled), chunk_size):
            X_chunk = X_resampled[i:i + chunk_size]
            y_chunk = y_resampled[i:i + chunk_size]

            scores = cross_val_score(model, X_chunk, y_chunk, cv=5, scoring='accuracy')

            print(f'Accuracy para {model_name} (chunk {i//chunk_size + 1}): {np.mean(scores)} (+/- {np.std(scores)})')

        model.fit(X_resampled, y_resampled)

        model_file_name = f'models/{model_name.lower().replace(" ", "_")}_is_sentence_{datetime.datetime.now()}.pkl'
        joblib.dump(model, model_file_name)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        with open('resultados.txt', 'a') as f:
            f.write(f'\n{10 * "#"} - {model_name} - {10 * f" {datetime.datetime.now()} #"}\n')
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'Classification Report:\n{classification_rep}\n')
            f.write(f'{10 * "#"} - {model_name} finalizado - {10 * "#"}\n')

if __name__ == "__main__":
    main()
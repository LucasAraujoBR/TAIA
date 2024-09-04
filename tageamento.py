import pandas as pd

# Carregar o CSV
df = pd.read_csv('database\\processos_dados.csv', sep='|')

# Define os termos que indicam uma sentença ou decisão
termos_sentenca = ["Sentença", "Decisão", "Julgamento", "Despacho"]

# Função para marcar se há sentença
def marca_sentenca(nome):
    if isinstance(nome, str):  # Verifica se é uma string
        return 1 if any(term in nome for term in termos_sentenca) else 0
    return 0  # Se não for string, marca como 0

# Aplica a função para criar a coluna 'is_sentenca'
df['is_sentenca'] = df['Nome'].apply(marca_sentenca)

# Salvar o DataFrame atualizado em um novo arquivo CSV (opcional)
df.to_csv('database\\resultado_tageado.csv', index=False, sep="|")

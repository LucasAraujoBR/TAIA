import json
import os
import re


import requests

from datetime import datetime, timedelta


def fetch_processes_from_api(url, api_key):
    # Monta o payload para buscar todos os processos
    payload = json.dumps({
        "query": {
            "match_all": {}
        },
        "size": 10000  # Ajuste o tamanho conforme necessário
    })

    # Configura os headers com a chave da API
    headers = {
        'Authorization': f'ApiKey {api_key}',
        'Content-Type': 'application/json'
    }

    try:
        # Faz a requisição POST para a API
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()  # Levanta um erro para códigos de status HTTP ruins
        return response.json()  # Retorna o JSON da resposta
    except requests.RequestException as e:
        print(f"Erro ao buscar dados de {url}: {e}")
        return None


def load_json(file_path):
    """
    Carrega o conteúdo de um arquivo JSON.

    Args:
        file_path (str): O caminho para o arquivo JSON.

    Returns:
        dict: O conteúdo do arquivo JSON como um dicionário.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
        return content
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Erro ao decodificar JSON em: {file_path}")
        return None
    

def send_openai_request(data):
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
        return None

def create_request_payload(system_message, user_message, model="gpt-4o-mini"):
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    }


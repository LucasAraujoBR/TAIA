import os
import warnings
import requests
import json

from helpers import create_request_payload, load_json, send_openai_request



prompt = """
Analise o movimento do processo fornecido e determine se ela constitui uma sentença judicial. Classifique o movimento como '1' se for uma sentença e '0' se não for. 
A resposta deve ser um número, '1' para sentença e '0' para não sentença, não traga nenhuma informação além do número de classificação."
"""
data = create_request_payload("Você é um especialista em análise de movimentação de processos judiciais, focado em identificar se um movimento é uma sentença ou não.", prompt)
response_content = send_openai_request(data)
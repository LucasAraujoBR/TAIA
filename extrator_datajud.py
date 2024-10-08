import requests
import json
import pandas as pd

from helpers import extract_movements, fetch_processes_from_api


def fetch_all_processes(api_key):
    endpoints = {
        # Justiça Estadual
        "TJAC": "https://api-publica.datajud.cnj.jus.br/api_publica_tjac/_search",
        "TJAL": "https://api-publica.datajud.cnj.jus.br/api_publica_tjal/_search",
        "TJAM": "https://api-publica.datajud.cnj.jus.br/api_publica_tjam/_search",
        "TJAP": "https://api-publica.datajud.cnj.jus.br/api_publica_tjap/_search",
        "TJBA": "https://api-publica.datajud.cnj.jus.br/api_publica_tjba/_search",
        "TJCE": "https://api-publica.datajud.cnj.jus.br/api_publica_tjce/_search",
        "TJDFT": "https://api-publica.datajud.cnj.jus.br/api_publica_tjdft/_search",
        "TJES": "https://api-publica.datajud.cnj.jus.br/api_publica_tjes/_search",
        "TJGO": "https://api-publica.datajud.cnj.jus.br/api_publica_tjgo/_search",
        "TJMA": "https://api-publica.datajud.cnj.jus.br/api_publica_tjma/_search",
        "TJMG": "https://api-publica.datajud.cnj.jus.br/api_publica_tjmg/_search",
        "TJMS": "https://api-publica.datajud.cnj.jus.br/api_publica_tjms/_search",
        "TJMT": "https://api-publica.datajud.cnj.jus.br/api_publica_tjmt/_search",
        "TJPA": "https://api-publica.datajud.cnj.jus.br/api_publica_tjpa/_search",
        "TJPB": "https://api-publica.datajud.cnj.jus.br/api_publica_tjpb/_search",
        "TJPE": "https://api-publica.datajud.cnj.jus.br/api_publica_tjpe/_search",
        "TJPI": "https://api-publica.datajud.cnj.jus.br/api_publica_tjpi/_search",
        "TJPR": "https://api-publica.datajud.cnj.jus.br/api_publica_tjpr/_search",
        "TJRJ": "https://api-publica.datajud.cnj.jus.br/api_publica_tjrj/_search",
        "TJRN": "https://api-publica.datajud.cnj.jus.br/api_publica_tjrn/_search",
        "TJRO": "https://api-publica.datajud.cnj.jus.br/api_publica_tjro/_search",
        "TJRR": "https://api-publica.datajud.cnj.jus.br/api_publica_tjrr/_search",
        "TJRS": "https://api-publica.datajud.cnj.jus.br/api_publica_tjrs/_search",
        "TJSC": "https://api-publica.datajud.cnj.jus.br/api_publica_tjsc/_search",
        "TJSE": "https://api-publica.datajud.cnj.jus.br/api_publica_tjse/_search",
        "TJSP": "https://api-publica.datajud.cnj.jus.br/api_publica_tjsp/_search",
        "TJTO": "https://api-publica.datajud.cnj.jus.br/api_publica_tjto/_search",

        # Justiça Federal
        "TRF1": "https://api-publica.datajud.cnj.jus.br/api_publica_trf1/_search",
        "TRF2": "https://api-publica.datajud.cnj.jus.br/api_publica_trf2/_search",
        "TRF3": "https://api-publica.datajud.cnj.jus.br/api_publica_trf3/_search",
        "TRF4": "https://api-publica.datajud.cnj.jus.br/api_publica_trf4/_search",
        "TRF5": "https://api-publica.datajud.cnj.jus.br/api_publica_trf5/_search",
        "TRF6": "https://api-publica.datajud.cnj.jus.br/api_publica_trf6/_search",

        # Tribunais Superiores
        "TST": "https://api-publica.datajud.cnj.jus.br/api_publica_tst/_search",
        "TSE": "https://api-publica.datajud.cnj.jus.br/api_publica_tse/_search",
        "STJ": "https://api-publica.datajud.cnj.jus.br/api_publica_stj/_search",
        "STM": "https://api-publica.datajud.cnj.jus.br/api_publica_stm/_search",

        # Justiça do Trabalho
        "TRT1": "https://api-publica.datajud.cnj.jus.br/api_publica_trt1/_search",
        "TRT2": "https://api-publica.datajud.cnj.jus.br/api_publica_trt2/_search",
        "TRT3": "https://api-publica.datajud.cnj.jus.br/api_publica_trt3/_search",
        "TRT4": "https://api-publica.datajud.cnj.jus.br/api_publica_trt4/_search",
        "TRT5": "https://api-publica.datajud.cnj.jus.br/api_publica_trt5/_search",
        "TRT6": "https://api-publica.datajud.cnj.jus.br/api_publica_trt6/_search",
        "TRT7": "https://api-publica.datajud.cnj.jus.br/api_publica_trt7/_search",
        "TRT8": "https://api-publica.datajud.cnj.jus.br/api_publica_trt8/_search",
        "TRT9": "https://api-publica.datajud.cnj.jus.br/api_publica_trt9/_search",
        "TRT10": "https://api-publica.datajud.cnj.jus.br/api_publica_trt10/_search",
        "TRT11": "https://api-publica.datajud.cnj.jus.br/api_publica_trt11/_search",
        "TRT12": "https://api-publica.datajud.cnj.jus.br/api_publica_trt12/_search",
        "TRT13": "https://api-publica.datajud.cnj.jus.br/api_publica_trt13/_search",
        "TRT14": "https://api-publica.datajud.cnj.jus.br/api_publica_trt14/_search",
        "TRT15": "https://api-publica.datajud.cnj.jus.br/api_publica_trt15/_search",
        "TRT16": "https://api-publica.datajud.cnj.jus.br/api_publica_trt16/_search",
        "TRT17": "https://api-publica.datajud.cnj.jus.br/api_publica_trt17/_search",
        "TRT18": "https://api-publica.datajud.cnj.jus.br/api_publica_trt18/_search",
        "TRT19": "https://api-publica.datajud.cnj.jus.br/api_publica_trt19/_search",
        "TRT20": "https://api-publica.datajud.cnj.jus.br/api_publica_trt20/_search",
        "TRT21": "https://api-publica.datajud.cnj.jus.br/api_publica_trt21/_search",
        "TRT22": "https://api-publica.datajud.cnj.jus.br/api_publica_trt22/_search",
        "TRT23": "https://api-publica.datajud.cnj.jus.br/api_publica_trt23/_search",
        "TRT24": "https://api-publica.datajud.cnj.jus.br/api_publica_trt24/_search",

        # Justiça Eleitoral
        "TREAC": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-ac/_search",
        "TREAL": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-al/_search",
        "TREAM": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-am/_search",
        "TREAP": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-ap/_search",
        "TREBA": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-ba/_search",
        "TRECE": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-ce/_search",
        "TREDF": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-df/_search",
        "TREES": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-es/_search",
        "TREGO": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-go/_search",
        "TREMA": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-ma/_search",
        "TREMG": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-mg/_search",
        "TREMZ": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-mz/_search",
        "TREMT": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-mt/_search",
        "TREPA": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-pa/_search",
        "TREPB": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-pb/_search",
        "TREPE": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-pe/_search",
        "TREPI": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-pi/_search",
        "TREPR": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-pr/_search",
        "TREP": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-rj/_search",
        "TRERN": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-rn/_search",
        "TRERO": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-ro/_search",
        "TRERR": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-rr/_search",
        "TRERS": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-rs/_search",
        "TRESC": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-sc/_search",
        "TRESE": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-se/_search",
        "TRESP": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-sp/_search",
        "TRETO": "https://api-publica.datajud.cnj.jus.br/api_publica_tre-to/_search",

        # Justiça Militar
        "STM": "https://api-publica.datajud.cnj.jus.br/api_publica_stm/_search"
    }

    all_results = []
    for name, url in endpoints.items():
        print(f"Buscando dados do {name}...")
        results = fetch_processes_from_api(url, api_key)
        if results:
            try:
                for item in results['hits']['hits'][0]['_source']['movimentos']:
                    all_results.append({"Tribunal": name, "Codigo":item['codigo'], "Nome": item['nome']})
            except:
                continue
    return all_results


if __name__ == "__main__":
    # Substitua pelo valor real da sua chave de API
    api_key = "cDZHYzlZa0JadVREZDJCendQbXY6SkJlTzNjLV9TRENyQk1RdnFKZGRQdw=="

    processes_data = fetch_all_processes(api_key)
    
    df = pd.DataFrame(processes_data)
    df.to_csv('database\\processos_dados.csv', index=False,sep='|')

    print("Dados salvos no arquivo processos_dados.csv")
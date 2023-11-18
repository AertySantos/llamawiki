import csv
import json


def csv_to_json(csv_file, json_file):
    data = []

    # Abre o arquivo CSV e lê seu conteúdo
    with open(csv_file, 'r', encoding="latin-1") as file:
        csv_reader = csv.reader(file)

        # Itera sobre as linhas do CSV e cria o formato JSON desejado
        for row in csv_reader:
            # Certifique-se de que há pelo menos duas colunas na linha antes de adicioná-la ao JSON
            if len(row) >= 2:
                entry = {
                    "input": row[0],
                    "output": row[1]
                }
                data.append(entry)

    # Escreve os dados convertidos em um arquivo JSON
    with open(json_file, 'w') as output_file:
        json.dump(data, output_file, indent=4)


# Nome do arquivo CSV de entrada e arquivo JSON de saída
nome_arquivo_csv = 'dados.csv'
nome_arquivo_json = 'saida.json'

# Chamada da função para converter o arquivo CSV para JSON
csv_to_json(nome_arquivo_csv, nome_arquivo_json)

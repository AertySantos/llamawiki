import json
from sklearn.model_selection import train_test_split  # Para a divisão dos dados

# Carrega o arquivo JSON
nome_arquivo_json = 'json/saida.json'

with open(nome_arquivo_json, 'r') as file:
    dados_json = json.load(file)

# Separa os dados em conjuntos de treino e teste
# Suponhamos que vamos dividir os dados em 90% de treino e 20% de teste
dados_treino, dados_teste = train_test_split(
    dados_json, test_size=0.1, random_state=42)

# Salva os conjuntos de treino e teste em arquivos JSON separados
nome_arquivo_treino = 'json/treino.json'
nome_arquivo_teste = 'json/teste.json'

with open(nome_arquivo_treino, 'w') as file:
    json.dump(dados_treino, file, indent=4)

with open(nome_arquivo_teste, 'w') as file:
    json.dump(dados_teste, file, indent=4)

import csv
import sys
from langchain.llms import LlamaCpp

MODEL_PATH = "models/llama-2-70b-chat.Q5_K_M.gguf"
csv_file = "csv/perguntas.csv"
data = {}

# criar a função que carrega a llama
n_gpu_layers = 80
n_batch = 1024
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.1,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    max_tokens=2000,
    top_p=1,
    verbose=True,
    n_ctx=4096
)

# Abre o arquivo para escrever ('w')
with open(csv_file, 'r', encoding="iso-8859-1") as file:
    # Itera sobre as linhas do CSV e cria o formato JSON desejado
    csv_reader = csv.reader(file, delimiter=';')
    for row in csv_reader:
        # Certifique-se de que há pelo menos duas colunas na linha antes de adicioná-la ao JSON
        # Supondo que a pergunta está na primeira coluna
        texto1 = row[0]
        print(texto1)
        query = f"Input Prompt: {texto1}"

        if query.lower() == 'exit':
            print('Exiting')
            break  # Interrompe o loop se "exit" for inserido

            # Supondo que llm() retorna a resposta com base na consulta
        resposta = llm(query)
        data[texto1] = resposta
        print(f"{resposta}")

# Use 'w' para escrever no arquivo
with open("resposta_llama_simples.txt", 'w', encoding="iso-8859-1") as arq:
    for perg, resp in data.items():  # Itera sobre os itens (chave, valor) do dicionário data
        # Escreve a pergunta e a resposta no arquivo
        arq.write(f"Pergunta: {perg}\n")
        arq.write(f"Resposta: {resp}\n")
        # Adiciona uma linha em branco para separar as perguntas e respostas
        arq.write('\n')

import csv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp

DB_FAISS_PATH = "vectorstore/db_faiss"
csv_file = "csv/perguntas.csv"
data = {}

# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')

new_db = FAISS.load_local(DB_FAISS_PATH, embeddings)

n_gpu_layers = 80
n_batch = 1024

llm = LlamaCpp(
    model_path="models/llama-2-13b-chat.Q4_K_S.gguf",
    temperature=0.1,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    top_p=1,
    max_tokens=200,
    verbose=True,  # Verbose is required to pass to the callback manager
    n_ctx=4096,
)

qa = ConversationalRetrievalChain.from_llm(
    llm, retriever=new_db.as_retriever(search_kwargs={'k': 1}))


# Abre o arquivo para escrever ('w')
with open(csv_file, 'r', encoding="iso-8859-1") as file:
    # Itera sobre as linhas do CSV e cria o formato JSON desejado
    csv_reader = csv.reader(file, delimiter=';')
    for row in csv_reader:
        # Certifique-se de que há pelo menos duas colunas na linha antes de adicioná-la ao JSON
        # Supondo que a pergunta está na primeira coluna
        texto1 = row[0]
        print(texto1)
        query = f"Pergunta: {texto1}"

        chat_history = []

        result = qa({"question": query, "chat_history": chat_history})

        resposta = result['answer']
        data[texto1] = resposta
        print(f"{resposta}")

with open("resposta_rga.txt", 'w', encoding="iso-8859-1") as arq:
    for perg, resp in data.items():  # Itera sobre os itens (chave, valor) do dicionário data
        # Escreve a pergunta e a resposta no arquivo
        arq.write(f"Pergunta: {perg}\n")
        arq.write(f"Resposta: {resp}\n")
        # Adiciona uma linha em branco para separar as perguntas e respostas
        arq.write('\n')

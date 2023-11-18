# Importação de módulos necessários para processamento de linguagem natural
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredXMLLoader

# Caminho para onde os dados processados serão salvos
DB_FAISS_PATH = "vectorstore/db_faiss"

# Carregamento dos dados de um arquivo XML não estruturado
loader = UnstructuredXMLLoader(
    "data/ptwiki-20230920-pages-articles.xml")
data = loader.load_and_split()  # minimizar o problema  ctx512
print(data[:1])

# Divisão do texto em partes menores para facilitar o processamento
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

# Impressão do número de partes resultantes após a divisão do texto
print(len(text_chunks))

# Geração de embeddings usando o modelo Hugging Face 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')

# Conversão dos pedaços de texto em embeddings e criação de um índice de pesquisa FAISS
docsearch = FAISS.from_documents(text_chunks, embeddings)

# Salvamento dos dados processados, incluindo o índice FAISS, em um diretório específico
docsearch.save_local(DB_FAISS_PATH)

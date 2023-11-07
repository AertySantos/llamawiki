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

# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')

new_db = FAISS.load_local(DB_FAISS_PATH, embeddings)

#query = "qual a capital do brasil?"
#docs = new_db.as_retriever(search_type="mmr",
#    search_kwargs={'k': 1, 'fetch_k': 50})
#print("Result", docs)

n_gpu_layers = 40
n_batch = 512

llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.Q5_K_S.gguf",
    temperature=0.75,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    top_p=1,
    verbose=True,  # Verbose is required to pass to the callback manager
    n_ctx=4096
)

qa = ConversationalRetrievalChain.from_llm(
    llm, retriever = new_db.as_retriever(search_kwargs={'k': 1}))

while True:
    chat_history = []
    #query = "What is the value of  GDP per capita of Finland provided in the data?"
    query = input(f"Input Prompt: ")
    if query == 'exit':
  
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa({"question": query, "chat_history": chat_history})
    print("Response: ", result['answer'])
#retriever = new_db.as_retriever(search_type="similarity_score_threshold",
    #search_kwargs={'score_threshold': 0.8})

#qa = RetrievalQA.from_chain_type(
    #llm=llm, chain_type="map_reduce", retriever=retriever, return_source_documents=True)
#query = "quantos estados tem no brasil?"
#result = qa({"query": query})
#print(f"Response: ", result['answer'])
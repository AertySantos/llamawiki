from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys

DB_FAISS_PATH = "vectorstore/db_faiss"

# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')

new_db = FAISS.load_local(DB_FAISS_PATH, embeddings)

# query = "What is the value of GDP per capita of Finland provided in the data?"

# docs = docsearch.similarity_search(query, k=3)

# print("Result", docs)

llm = CTransformers(model="models/llama-2-7b-chat.Q5_K_S.gguf",
                    model_type="llama",
                    max_new_tokens=512,
                    temperature=0.1)

qa = ConversationalRetrievalChain.from_llm(
    llm, retriever=new_db.as_retriever())

while True:
    chat_history = []
    # query = "What is the value of  GDP per capita of Finland provided in the data?"
    query = input(f"Input Prompt: ")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa({"question": query, "chat_history": chat_history})
    print("Response: ", result['answer'])

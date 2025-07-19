import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

# Define paths
FAISS_INDEX_PATH = "faiss_index"

# Build FAISS index from text
def create_faiss_index(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

# Ask questions using FAISS
def ask_question(question):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True  
    )

    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4")

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain.run(question)

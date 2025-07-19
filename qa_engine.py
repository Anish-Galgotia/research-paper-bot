import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import pinecone

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

# Init Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# Embed + Index function
def create_pinecone_index(text, index_name="paper-index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    if index_name not in pinecone.list_indexes():
        Pinecone.from_documents(docs, embeddings, index_name=index_name)
    else:
        print("Index already exists. Skipping creation.")

# Query function
def ask_question(question, index_name="paper-index"):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4")

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=docsearch.as_retriever())
    result = qa_chain.run(question)
    return result
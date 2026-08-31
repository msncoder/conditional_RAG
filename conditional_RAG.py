import os
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_coummunity.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchian_huggingface import HuggingFaceEmbeddings
from langchain.coummunity.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


#step 1 building rag retriver

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_retriver(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

academic_retriever = build_retriver("workflows/conditional/academic.pdf")
fee_retriever = build_retriver("workflows/conditional/fee_structure.pdf")


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.4)

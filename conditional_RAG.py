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
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

acedemic_retriever = build_retriver("workflows/conditional/academic.pdf")
fee_retriever = build_retriver("workflows/conditional/fee_structure.pdf")


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.4)


#Step 2 state 

class State(TypedDict):
    program: str
    messsage: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


#step 3 - Nodes generation


def classifier_node(state : State) -> dict:
    """Look at the latest user message and decide which path to take."""

    last_message = state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"
    
    return {"query_type" : category}


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state["messages"][-1].content
    docs = acedemic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}
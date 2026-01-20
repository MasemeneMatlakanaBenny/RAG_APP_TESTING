from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings,ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

doc=Document(
    page_content="Forecasting Metrics Glossary",
    source={
        "source":"Forecast Verification Glossary.pdf",
        "field":"Time Series and Forecasting in Machine Learning"
    }
)

loader=PyPDFLoader("Forecast Verification Glossary.pdf")
document=loader.load()

## create the text splitter:
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ",""]
)

chunks=text_splitter.split_documents(document)


## create the embedding model:
embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model)

retriever=vector_store.as_retriever(
    search_type="similarity",
    search_kwags={"k":2}
)

prompt=ChatPromptTemplate.from_template(
    """
    You are a useful assistant that answers questions with honesty and integrity.
    Answer questions using the below context and if the answer is not within the context then 
    say I don't know

    Context:
    {context}

    Question:
    {question}
    """
)

from typing import List,TypedDict

class RAGState(TypedDict):
    question:str
    context:List[Document]
    answer:str

llm_endpoint=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False)

chat_model=ChatHuggingFace(llm=llm_endpoint)

def retrieve_node(state:RAGState)->RAGState:
    docs=retriever.invoke(state["question"])

    return {
    "question":state["question"],
    "context":docs
    }



def generation_node(state:RAGState)->RAGState:
    response=(
        prompt
        |chat_model
        |StrOutputParser()
    ).invoke({
        "question":state["question"],
        "context":state["context"]
    })

    return {"answer":response}

from langgraph.graph import StateGraph,END

graph=StateGraph(RAGState)


graph.add_node("retrieve",retrieve_node)
graph.add_node("generate",generation_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve","generate")
graph.add_edge("generate",END)

rag_app=graph.compile()

import streamlit as st

# -----------------------------
# Minimal Streamlit RAG App
# -----------------------------

st.set_page_config(page_title="RAG App", layout="wide")

st.title("📚 Simple RAG App")
st.caption("Enter a question, get an answer, chat history is saved")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar (Configuration) ---
st.sidebar.header("Configuration")
st.sidebar.info("Basic RAG settings")

# (Optional placeholders – extend later if needed)

# --- Core RAG Call ---
# This mirrors your notebook logic:
# results = rag_app.invoke({"question": user_query})

def run_rag(query: str):
    results = rag_app.invoke({"question": query})
    return results["answer"]


# --- User Input ---
user_query = st.chat_input("Enter your question")

if user_query:
    with st.spinner("Generating answer..."):
        answer = run_rag(user_query)

    st.session_state.chat_history.append(
        {"question": user_query, "answer": answer}
    )

# --- Display Chat History ---
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])






import os
import gradio as gr

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# =========================
# Load API key
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =========================
# Load embeddings + FAISS
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
)


# =========================
# LLM
# =========================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b"
)


# =========================
# Prompt
# =========================
prompt = ChatPromptTemplate.from_template("""
You are a health insurance advisor.

Answer ONLY from context.

Format:
covered:
how:
why:
citation:

Context:
{context}

Question:
{input}
""")


# =========================
# Chain
# =========================
doc_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, doc_chain)
print("hello")


# =========================
# Function
# =========================
def ask(query):
    response = qa_chain.invoke({"input": query})
    return response["answer"]


# =========================
# UI
# =========================
demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(label="Ask about insurance"),
    outputs=gr.Textbox(label="Answer"),
    title="Health Insurance RAG Assistant"
)
demo.launch(ssr_mode=False)
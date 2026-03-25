import os
import gradio as gr

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 🔥 lightweight embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# 🔥 faster LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192"
)

prompt = ChatPromptTemplate.from_template("""
Answer ONLY from context.

Context:
{context}

Question:
{input}
""")

doc_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, doc_chain)

def ask(query):
    return qa_chain.invoke({"input": query})["answer"]

demo = gr.Interface(fn=ask, inputs="text", outputs="text")

# 🔥 RENDER FIX
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)

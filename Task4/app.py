import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# إعدادات الصفحة
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

# عنوان التطبيق
st.title("🧠 AI RAG Chatbot")
st.write("Ask me anything based on your local Chroma knowledge base!")

# إعداد قاعدة البيانات والموديل
DB_DIR = "chroma_store"
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vs = Chroma(persist_directory=DB_DIR, embedding_function=emb)
retriever = vs.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = ChatPromptTemplate.from_template("""
Use ONLY the following context to answer the question clearly:
{context}
Question: {question}
Answer concisely.
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# واجهة المستخدم
question = st.text_input("💬 Your question:", placeholder="Ask me about RAG, Chroma, or AI...")

if question:
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(question)
        st.success(response.content)

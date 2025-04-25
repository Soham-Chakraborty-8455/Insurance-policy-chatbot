import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@st.cache_data

def loadPDF(book):
    loader = PyPDFLoader(book)
    documents = loader.load()
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    return db

@st.cache_resource

def initialize_chatbot(book):
    db = loadPDF(book)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever=db.as_retriever())
    return qa

st.set_page_config(page_title="TCS AI Insurance Chatbot")
st.title("AI Insurance Policy Chatbot")

book = "./Insurance-policy.pdf"  
chatbot = initialize_chatbot(book)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask me anything about insurance policies:")
if user_input:
    with st.spinner("Thinking..."):
        response = chatbot.run({"question": user_input, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((user_input, response))
        st.write("**You:**", user_input)
        st.write("**Bot:**", response)


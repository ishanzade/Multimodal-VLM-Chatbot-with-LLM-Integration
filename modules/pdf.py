import hashlib
# import faiss
import os,streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from utils.model_loader import get_embedding_model,get_llm_model
from prompt import get_chat_prompt_pdf
from utils.config_loader import config
from langchain.chains.question_answering  import load_qa_chain

def get_pdf_hash(pdf_file):
    hasher = hashlib.sha256()
    hasher.update(pdf_file.getvalue())
    return hasher.hexdigest()



def load_existing_embeddings(pdf_hash):
    faiss_path = os.path.join(config['paths']['embeddings_dir'], f"{pdf_hash}.faiss")
    if os.path.exists(faiss_path):
        embeddings = get_embedding_model()
        return FAISS.load_local(faiss_path, embeddings,allow_dangerous_deserialization=True)  
    return None


def process_pdf(pdf, pdf_hash):
    pdf_reader = PdfReader(pdf)
    text = "".join(page.extract_text() for page in pdf_reader.pages)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    faiss_path = os.path.join(config['paths']['embeddings_dir'], f"{pdf_hash}.faiss")
    vectorstore.save_local(faiss_path)
    return vectorstore

def pdf_QndA():
    st.title("📚 PDF Question & Answering with Persistent Embeddings")
    pdf = st.file_uploader("Upload PDF", type='pdf')
    
    #geth hash value for pdf_file
    if pdf:
        pdf_hash = get_pdf_hash(pdf)

        vectorestore = load_existing_embeddings(pdf_hash)
        
        if vectorestore:
            st.success("Loaded existing embeddings for this PDF.")
        else:
            st.warning("Processing new PDF, generating embeddings...")
            vectorestore = process_pdf(pdf, pdf_hash)

        if vectorestore:
            query = st.text_input("🔍 Ask a question about the PDF")
            if query:
                docs = vectorestore.similarity_search(query, k=3)
                
                llm = get_llm_model()
                prompt = get_chat_prompt_pdf()
                
                chain = load_qa_chain(llm, chain_type='stuff', prompt=prompt)
                
                response = chain.run(input_documents=docs, question=query)
                st.write(response)

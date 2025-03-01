import hashlib
import os
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.model_loader import get_embedding_model, get_llm_model
from prompt import get_chat_prompt_pdf
from utils.config_loader import config
from langchain.chains.question_answering import load_qa_chain

def get_pdf_hash(pdf_file):
    hasher = hashlib.sha256()
    hasher.update(pdf_file.getvalue())
    return hasher.hexdigest()

def load_existing_embeddings(pdf_hash):
    faiss_path = os.path.join(config['paths']['embeddings_dir'], f"{pdf_hash}.faiss")
    if os.path.exists(faiss_path):
        embeddings = get_embedding_model()
        return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)  
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

def pdf_QnA():
    st.title("📚 PDF Question & Answering with Persistent Embeddings")
    
    pdf = st.file_uploader("Upload PDF", type='pdf')
    
    if pdf:
        
        pdf_hash = get_pdf_hash(pdf)
        vectorestore = load_existing_embeddings(pdf_hash)
        
        if vectorestore:
            st.success("Loaded existing embeddings for this PDF.")
        else:
            st.warning("Processing new PDF, generating embeddings...")
            vectorestore = process_pdf(pdf, pdf_hash)

        if vectorestore:
            
            if "pdf_chat_history" not in st.session_state:
                st.session_state["pdf_chat_history"] = []

            
            for entry in st.session_state.get('pdf_chat_history', []):
                st.markdown(
                    f"""
                    <div style='text-align: right;'>
                        <div style='display: inline-block; background-color: #DCF8C6; color: black; padding: 10px; border-radius: 10px; max-width: 70%; margin: 10px;'>
                            You: {entry['question']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )
                st.markdown(f"<div style='text-align: left; color: white; margin: 10px;'>Answer:  {entry['response']}</div>", unsafe_allow_html=True)

                
            query = st.text_input("🔍 Ask a question about the PDF")
            if query:
                # Perform similarity search and get the top 3 matching documents
                docs = vectorestore.similarity_search(query, k=3)
                
                llm = get_llm_model()
                prompt = get_chat_prompt_pdf()
                
                # Load QA chain
                #chain_type: "stuff","map_reduce","map_rerank"
                chain = load_qa_chain(llm, chain_type='stuff', prompt=prompt)

                
                # Get response from the model
                response = chain.run(input_documents=docs, question=query)
                
                # Save to chat history
                st.session_state['pdf_chat_history'].append({"question": query, "response": response})
                st.rerun() 

    



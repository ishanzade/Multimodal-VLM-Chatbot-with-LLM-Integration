import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.config_loader import config
import ollama
from langchain_ollama.llms import OllamaLLM

def get_llm_model():

    return OllamaLLM(model=config["llm"]["model_name"])

# This function loads the embedding model locally (no API key)
def get_embedding_model():
    # Using Hugging Face local model for embeddings
    return HuggingFaceEmbeddings(model_name=config["llm"]["embedding_model"])

# Load the FAISS vector store locally
def load_faiss(vectorstore_path):
    embeddings = get_embedding_model()  # Get local embeddings model
    # Load the FAISS vectorstore from the provided local path
    return FAISS.load_local(vectorstore_path, embeddings)

# Save the FAISS vectorstore locally
def save_faiss(vectorstore, vectorstore_path):
    # Save the FAISS vectorstore locally at the specified path
    vectorstore.save_local(vectorstore_path)

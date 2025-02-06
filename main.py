import streamlit as st
from modules.pdf import process_pdf, load_existing_embeddings,get_pdf_hash
from modules.text_qna import ask_question
from modules.image import process_image
from modules.pdf import pdf_QndA
from modules.text_qna import text_qna
from PIL import Image
import io

def main():
    st.set_page_config(page_title="VLM Chat Bot", layout="wide")
    
    # Sidebar
    st.sidebar.title("VLM Chat Bot")
    option = st.sidebar.selectbox("Choose an option:", [
        "Chat with PDF",
        "Text Question Answering",
        "Image Questioning"
    ])
    
    st.title("Welcome to VLM ChatBot")
    
    if option == "Chat with PDF":
        pdf_QndA()
        
    elif option == "Text Question Answering":
        
        text_qna()

 
    elif option == "Image Questioning":
        image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        

        if image_file:
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            new_image = image.resize((600, 600))
            
        
            st.image(new_image, caption="📸 Uploaded Image")
            question = st.text_input("💬 Ask a question about this image:")
            if question:
                
                if st.button("Ask"):
                    st.write("Processing your image question...")
                    response = process_image(image_bytes, question=question)
                    st.write(response)
                    
            elif st.button("📝 Generate Caption"):
                st.write("Processing your image caption...")
                response = process_image(image_bytes, question=question)
                st.write(response)

if __name__ == "__main__":
    main()

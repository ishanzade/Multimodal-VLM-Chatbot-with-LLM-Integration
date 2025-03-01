import streamlit as st
from modules.pdf import process_pdf, load_existing_embeddings,get_pdf_hash
from modules.text_qna import ask_question
from modules.image import process_image
from modules.pdf import pdf_QnA
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
        pdf_QnA()
        
    elif option == "Text Question Answering":
        
        text_qna()

 
    elif option == "Image Questioning":
        # history
        image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

        if "img_chat_history" not in st.session_state:
            st.session_state["img_chat_history"] = []
        

        if image_file:
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            new_image = image.resize((600, 600))
            
        
            st.image(new_image, caption="📸 Uploaded Image")
            
            for entry in st.session_state["img_chat_history"]:
    
                st.markdown(
                    f"""
                    <div style='text-align: right;'>
                        <div style='display: inline-block; background-color: #DCF8C6; color: black; padding: 10px; border-radius: 10px; max-width: 70%; margin: 10px;'>
                            <b>🧑‍💬 You:</b> {entry['question']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )

            
                st.markdown(
                    f"""
                    <div style='text-align: left;'>
                        <div style='display: inline-block; background-color: #222; color: white; padding: 10px; border-radius: 10px; max-width: 70%; margin: 10px;'>
                            <b>🤖 Answer:</b> {entry['response']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )



            question = st.text_input("💬 Ask a question about this image:")
            if st.button("Ask")and question:
                
                st.write("Processing your image question...")
                response = process_image(image_bytes, question=question)
                
                st.session_state["img_chat_history"].append({"question": question, "response": response})
                st.rerun()
                    
            elif st.button("📝 Generate Caption"):
                st.write("Processing your image caption...")
                response = process_image(image_bytes, question=question)
                st.session_state["img_chat_history"].append({"question": "Generate Caption", "response": response})
                st.rerun()
            
            
      

if __name__ == "__main__":
    main()

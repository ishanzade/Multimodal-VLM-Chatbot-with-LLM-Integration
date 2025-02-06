from prompt import get_chat_prompt, save_context, get_output_parser  # Import functions from prompt.py
from utils.model_loader import get_llm_model
import streamlit as st

def ask_question(input_text):
    
    
    # Get LLAMA2 model
    llm = get_llm_model()

    # Get the chat prompt based on memory context
    prompt_template = get_chat_prompt()

    # Get output parser
    output_parser = get_output_parser()

    # Create LangChain pipeline (prompt -> LLM -> output parser)
    chain = prompt_template | llm | output_parser

    if input_text:
        # Generate response using the chain
        response = chain.invoke({"question": input_text, "context": get_chat_prompt()})

        # Save the interaction (question-response) to memory
        save_context({"question": input_text, "response": response})

        return response
    return "No response generated."

def text_qna():
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display chat history
    for entry in st.session_state.get('chat_history', []):
        st.markdown(
            f"""
            <div style='text-align: right;'>
                <div style='display: inline-block; background-color: #DCF8C6; color: black; padding: 10px; border-radius: 10px; max-width: 70%;'>
                    You: {entry['question']}
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown(f"<div style='text-align: left; color: white;'>Answer:  {entry['response']}</div>", unsafe_allow_html=True)

    # Allow user to input a new question
    question = st.text_input("Ask Query")

    if st.button("Send") and question:
        response = ask_question(question)
        st.session_state['chat_history'].append({"question": question, "response": response})
        st.rerun()  

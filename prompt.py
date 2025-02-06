from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Initialize memory
memory = ConversationBufferMemory()

def save_context(interaction):
    memory.save_context(
        {"question": interaction["question"]}, 
        {"response": interaction["response"]}
    )

def get_memory_context():
    memory_data = memory.load_memory_variables({})
    return memory_data.get("history", "No memory available. Generate the answer from scratch.")


def get_chat_prompt():
    memory_context = get_memory_context()

    return ChatPromptTemplate.from_messages([
        ("system", 
         "You are an intelligent and helpful assistant. Ensure your responses are clear, concise, and grammatically correct. "
         "Analyze the user's query carefully before providing a thoughtful response."),
        ("user", f"{memory_context}"),
        ("user", "Question: {question}")
    ])





def get_chat_prompt_pdf():
   
    return ChatPromptTemplate.from_template(
        """
        You are an AI assistant answering questions based on the provided document.
        Your goal is to extract the most relevant information and provide a specific, accurate response.

        Context:
        {context}

        Question: {question}

        Instructions:
        - Only use information present in the provided context.
        - Provide a concise response, directly answering the question.
        - If the question cannot be answered based on the context, state that clearly.
        """
    )

# Function to get output parser (used for parsing LLM responses)
def get_output_parser():
    return StrOutputParser() 
import tempfile
import os
import ollama
from PIL import Image
import io
from utils.config_loader import config
import streamlit as st

def process_image(image_bytes, question=None):
  
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_bytes)
        image_path = temp_file.name

    if question:
        response = ollama.chat(
            model=config["llm"]["image_model"],
            messages=[{
                "role": "user",
                "content": f"Think on the query first and then give Answer the following question using the information from the image, including objects, context, and background.: {question}",
                "images": [image_path]
            }]
        )
        result = response["message"]["content"]

    else:
        response = ollama.chat(
            model=config["llm"]["image_model"],
            messages=[{
                "role": "user",
                "content": "Think and give detailed description of the image and also including objects, context, and background.",
                "images": [image_path]
            }]
        )
        result = response["message"]["content"]

    os.unlink(image_path)
    return result

    
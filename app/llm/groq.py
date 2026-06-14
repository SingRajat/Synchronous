from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def Get_LLM():
    llm=ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1000,
        timeout=30,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return llm



from langchain_openai import OpenAIEmbeddings
from config.config import OPENAI_API_KEY

def create_openai_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY, dimensions=3072)
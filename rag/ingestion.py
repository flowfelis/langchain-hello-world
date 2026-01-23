import os

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

if __name__ == "__main__":
    print("Ingesting...")
    print(f"PINECONE_API_KEY: {os.getenv('PINECONE_API_KEY')}")

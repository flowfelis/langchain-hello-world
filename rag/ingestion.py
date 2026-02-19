import os

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
import math
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("/home/puma/Projects/langchain-hello-world/rag/mediumblog1.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    print("Ingesting...")
    PineconeVectorStore.from_documents(
        documents=texts, embedding=embeddings, index_name=os.getenv("INDEX_NAME")
    )
    print("finish")

from rich import print
from langchain_community.document_loaders import TextLoader
from init_model import EMBEDDING_LLM
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


if __name__ == "__main__":
    load_dotenv()
    print("ingesting documents...")
    loader = TextLoader(r"medium-blog.txt", encoding="utf-8")
    documents = loader.load()

    print("splitting documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    print(f"creating {len(texts)} text chunks...")

    embeddings = EMBEDDING_LLM

    print("ingesting to Pinecone vector store...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("ingestion complete.")

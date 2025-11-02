from http import client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..utility.qdrant_manger import QdrantManager
from langchain_community.document_loaders import TextLoader
import os
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv


os.environ["LANGSMITH_PROJECT"] = "retrieval"
load_dotenv()


# 檢查當前工作目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "facts", "fact1.txt")

loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=0,
)

EMBEDDING_LLM = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
)
COLLECTION_NAME = "fast-rag"
DIMENSION_SIZE = 1536  # 替換為您的嵌入向量維度

docs_split = text_splitter.split_documents(documents)

with QdrantManager() as manager:

    client = manager.get_qdrant_client()

    if client is None:
        raise ValueError("無法獲取 Qdrant 客戶端實例。請檢查配置。")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=DIMENSION_SIZE, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=EMBEDDING_LLM,
    )

    # 加入文件至向量資料庫
    # vector_store.add_documents(documents=docs_split)

    retriever = vector_store.as_retriever()

from langchain_qdrant import QdrantVectorStore
from utility.qdrant_manger import QdrantManager
from graph.state import GraphState
from utility.init_model import EMBEDDING_LLM
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
import os

load_dotenv()

# 模組級別的 retriever 緩存（避免重複創建）
_retriever = None


def _get_retriever():
    """獲取或創建 retriever 實例（單例模式）"""
    global _retriever

    if _retriever is None:
        manager = QdrantManager()
        client = manager.get_qdrant_client()

        if client is None:
            raise ValueError(
                "Unable to get Qdrant client instance. Please check configuration."
            )

        collection_name = os.environ.get("COLLECTION_NAME")
        if not collection_name:
            raise ValueError("COLLECTION_NAME environment variable is not set.")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=EMBEDDING_LLM,
        )

        # 可配置的 retriever 參數
        _retriever = vector_store.as_retriever(
            search_type="similarity",  # 或 "mmr" 以獲得更多樣化的結果
            search_kwargs={"k": 4},  # 檢索的文檔數量
        )

    return _retriever


def retrieve(state: GraphState):
    """
    從向量存儲中檢索相關文檔

    Args:
        state (dict): 當前圖狀態，必須包含 'question' 鍵

    Returns:
        dict: 包含 'documents' 鍵的字典，值為檢索到的文檔列表

    Raises:
        ValueError: 當無法獲取 Qdrant 客戶端或配置錯誤時
        KeyError: 當 state 中缺少 'question' 鍵時
    """
    print("---RETRIEVE---")

    question = state.get("question")
    if not question:
        raise KeyError("State must contain 'question' key with a non-empty value.")

    try:
        retriever = _get_retriever()
        documents: List[Document] = retriever.invoke(question)

        print(f"Retrieved {len(documents)} documents")

        return {"documents": documents}

    except Exception as e:
        print(f"Error during retrieval: {e}")
        # 根據需求決定是否重新拋出異常或返回空結果
        raise

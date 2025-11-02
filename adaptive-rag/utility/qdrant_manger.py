from qdrant_client import QdrantClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()


class QdrantConfig(BaseModel):
    """Qdrant 連線配置"""

    url: str = Field(default="http://localhost", description="Qdrant 伺服器 URL")
    port: int = Field(default=6333, ge=1, le=65535, description="Qdrant 伺服器端口")
    timeout: Optional[int] = Field(default=None, description="連線逾時時間(秒)")

    class Config:
        frozen = True  # 配置不可變


class QdrantManager:
    """管理 Qdrant 客戶端和相關資源的單例類"""

    _instance = None
    _client: Optional[QdrantClient] = None
    _config: Optional[QdrantConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        url: str = "http://localhost",
        port: int = 6333,
        timeout: Optional[int] = None,
    ):
        if self._client is not None:
            return

        # 使用 BaseModel 驗證配置
        self._config = QdrantConfig(url=url, port=port, timeout=timeout)
        self._client = QdrantClient(
            url=self._config.url, port=self._config.port, timeout=self._config.timeout
        )

    def get_qdrant_client(self) -> Optional[QdrantClient]:
        """獲取 Qdrant 客戶端實例"""
        return self._client

    def get_config(self) -> Optional[QdrantConfig]:
        """獲取當前配置"""
        return self._config

    def close(self):
        """關閉客戶端連線"""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

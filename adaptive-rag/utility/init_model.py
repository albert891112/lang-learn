from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureOpenAIEmbeddings
import os

os.environ["LANGSMITH_PROJECT"] = "adaptive-rag"
load_dotenv()

LLM = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    temperature=0,
    timeout=None,
    max_retries=2,
)

OLLAMA_LLM = ChatOllama(
    model=os.environ["OLLAMA_MODEL_NAME"],
    temperature=0,
)

EMBEDDING_LLM = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
)

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

import os


load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

from langchain_core.prompts import PromptTemplate
from rich import print
from init_model import EMBEDDING_LLM, LLM
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


template = """Use the following piece of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible.

context: {context}

question: {question}

answer:
"""


custom_rag_prompt = PromptTemplate.from_template(template=template)


vector_store = PineconeVectorStore(
    embedding=EMBEDDING_LLM, index_name=os.environ["INDEX_NAME"]
)

rag_chain = (
    (
        {
            "context": vector_store.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
    )
    | custom_rag_prompt
    | LLM
)


query = "what is pinecone in machine learning?"

res = rag_chain.invoke(query)

print("Custom RAG Result:", res.content)

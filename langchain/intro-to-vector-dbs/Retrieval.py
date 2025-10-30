from rich import print
from langchain_core.prompts import PromptTemplate
from init_model import EMBEDDING_LLM, LLM
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()


if __name__ == "__main__":
    embeddings = EMBEDDING_LLM
    llm = LLM

    query = "what is pinecone in machine learning?"

    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print("Result:", result.content)

    vector_store = PineconeVectorStore(
        embedding=embeddings, index_name=os.environ["INDEX_NAME"]
    )

    retrieval_qa_chat_prompt = hub.pull(
        "langchain-ai/retrieval-qa-chat",
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm, prompt=retrieval_qa_chat_prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )

    result = retrieval_chain.invoke(input={"input": query})
    print("Retrieval Result:", result)

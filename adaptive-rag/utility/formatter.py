from langchain_core.documents import Document


def format_docs(documents: list[Document]):
    """
    Format documents into a single string

    Args:
        documents (List[Document]): List of documents to format

    Returns:
        str: Formatted documents as a single string
    """
    return "\n\n".join([doc.page_content for doc in documents])

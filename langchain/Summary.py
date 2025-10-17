from init_model import ollama_llm

from langchain_core.prompts import ChatPromptTemplate

prompt_str = """
Summarize the following text in a concise manner:
{text}
"""

prompt = ChatPromptTemplate.from_template(prompt_str)

model_with_prompt = prompt | ollama_llm


def summarize(text: str) -> str:
    """Summarize the given text."""
    return model_with_prompt.invoke({"text": text}).content


if __name__ == "__main__":
    user_input = input("Enter text to summarize: ")
    summary = summarize(user_input)
    print("=============================================================")
    print("Summary:", summary)

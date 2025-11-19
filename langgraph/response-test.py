from typing import AsyncIterator, TypedDict
from click import prompt
from langgraph.graph import StateGraph
from init_model import llm
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate


class AgentState(TypedDict):
    input: str
    response: str


async def agent(state: AgentState):
    """A simple agent that processes the input and returns a response."""
    input_text = state["input"]

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a helpful assistant."),
            ("human", "{input}"),
        ]
    )

    model = prompt_template | llm

    # 收集完整回應
    full_response = ""
    async for chunk in model.astream({"input": input_text}):
        if hasattr(chunk, "content"):
            full_response += chunk.content

    return {"response": full_response}


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.set_entry_point("agent")
workflow.set_finish_point("agent")

graph = workflow.compile()


async def test_streaming():
    """直接使用 model streaming 的版本"""
    user_input = prompt("Enter your input: ")

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a helpful assistant."),
            ("human", "{input}"),
        ]
    )

    model = prompt_template | llm

    async for chunk in model.astream({"input": user_input}):
        if hasattr(chunk, "content"):
            print(chunk.content, end="", flush=True)
    print()


async def test_graph_streaming():
    """使用 LangGraph astream 實現逐字輸出"""
    user_input = prompt("Enter your input: ")

    initial_state: AgentState = {"input": user_input, "response": ""}

    # 使用 astream 來獲取中間狀態
    async for state in graph.astream(initial_state):
        # state 是每個節點執行後的完整狀態
        if "agent" in state and "response" in state["agent"]:
            print(f"Agent response: {state['agent']['response']}")


async def test_graph_streaming_events():
    """使用 astream_events 實現真正的逐字輸出"""
    user_input = prompt("Enter your input: ")

    initial_state: AgentState = {"input": user_input, "response": ""}

    # astream_events 可以捕獲 LLM 的每個 token
    async for event in graph.astream_events(initial_state, version="v2"):
        kind = event["event"]

        # 捕獲 LLM 串流事件
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            print(content, end="", flush=True)
            print("====================================")

    print()


async def test_with_graph():
    """使用 graph 的版本（完整回應）"""
    user_input = prompt("Enter your input: ")

    initial_state: AgentState = {"input": user_input, "response": ""}

    result = await graph.ainvoke(initial_state)
    print("Final Response:", result["response"])


if __name__ == "__main__":
    import asyncio

    async def main():
        # print("=== 1. Direct Model Streaming ===")
        # await test_streaming()

        # print("\n=== 2. Graph astream (完整狀態) ===")
        # await test_graph_streaming()

        print("\n=== 3. Graph astream_events (逐字輸出) ===")
        await test_graph_streaming_events()

        # print("\n=== 4. Graph ainvoke (完整回應) ===")
        # await test_with_graph()

    asyncio.run(main())

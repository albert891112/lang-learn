import asyncio
from typing import Literal

from init_model import AllState, llm
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

chathistory = []


class foodrecommendationInput(BaseModel):
    preference: str = Field(description="食物偏好選項，選項有: 鹹食 、 甜食")


class priceInput(BaseModel):
    item: str = Field(description="食物名稱")


def food_recommendation(preference: str) -> str:
    """根據用戶的偏好推薦食物。"""
    recommendations = {
        "鹹食": "推薦您嘗試鹹蛋炒飯，配料豐富，口感絕佳。",
        "甜食": "推薦您嘗試芒果糯米飯，鮮美的水果讓人回味無窮。",
    }

    return recommendations.get(preference, "抱歉，我們沒有該食物的推薦信息。")


def query_price(
    item: str,
) -> str:
    """查詢食物的價格。"""
    prices = {
        "鹹蛋炒飯": 80,
        "芒果糯米飯": 100,
        "珍珠奶茶": 50,
        "炸雞排": 70,
    }
    if item not in prices:
        return f"抱歉，我們沒有 {item} 的價格信息。"
    total = prices[item]
    return f"{item} 的價格是 {total} 元新台幣。"


food_recommendation_tool = StructuredTool.from_function(
    func=food_recommendation,
    name="food_recommendation_tool",
    description="根據用戶的偏好推薦食物",
    args_schema=foodrecommendationInput,
)

price_query_tool = StructuredTool.from_function(
    func=query_price,
    name="price_query_tool",
    description="計算食物的總價",
    args_schema=priceInput,
)


tool = [food_recommendation_tool, price_query_tool]
tool_node = ToolNode(tools=tool)


model = llm
model_with_tools = model.bind_tools(tool)


# agent
def semantic_agent(state: AllState):
    response_prompt_str = """
    You are a helpful assistant that helps people find information.
    user will tell you what food they prefer (salty or sweet) and you need to response a todo list to ask tool agent to
    1. get the food recommendation and price for them.
    2. get the price of the food you recommended.

    if user's query is not about food recommendation, respond with "抱歉，我無法提供該資訊。" and tag [UNRELATED].

    Here is the user query:
    ---
    {user_query}
    ---
    """
    response_prompt = ChatPromptTemplate.from_template(response_prompt_str)

    response_chain = response_prompt | model
    messages = state["messages"]
    response = response_chain.invoke({"user_query": messages[-1].content})
    return {"messages": [response], "todo": [response]}


# agent
def tool_agent(state: AllState):
    response_chain = model_with_tools
    messages = state["messages"]
    response = response_chain.invoke(messages)
    return {"messages": [response]}


# 步驟二、定義好流程控制函數


# 是否繼續執行語意代理後續工作
def semantic_agent_should_continue(
    state: AllState,
) -> Literal["pretty_print", "tool_agent"]:
    messages = state["messages"]
    last_message = messages[-1]
    if "[UNRELATED]" in last_message.content:
        return "pretty_print"
    return "tool_agent"


# 是否繼續執行工具代理後續工作
def tool_agent_should_continue(state: AllState) -> Literal["tools", "responder"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "responder"


# 是否繼續執行回應代理後續工作
def responder_should_continue(state: AllState) -> Literal["pretty_print", "tool_agent"]:
    if not state["IsComplete"]:
        return "tool_agent"
    return "pretty_print"


def responder(state: AllState):
    response_prompt_str = """
    You have given a information , If you think your to-do list is complete, you have to respond to user's query based on the information and tag [COMPLETE] , 
    or you should response tag [UNCOMPLETE] , and tell agent what information is missing.

    Here is the user query:
    ---
    {user_query}
    ---

    Here is the information:
    ---
    {information}
    ---
    """
    response_prompt = ChatPromptTemplate.from_template(response_prompt_str)

    response_chain = response_prompt | model
    messages = state["messages"]
    response = response_chain.invoke(
        {"user_query": messages[0], "information": messages[-1]}
    )
    # 检查响应中是否包含完成标识
    is_complete = "[COMPLETE]" in response.content

    return {"messages": [response], "IsComplete": is_complete}


def response(state: AllState):
    messages = state["messages"]

    print("All Messages:")
    for msg in messages:
        print(f"{msg} - {msg}\n")

    # 提取初始的使用者查詢
    user_query = messages[0].content

    # 提取所有工具回傳的結果 (ToolMessage)
    tool_results = []
    # 遍歷所有訊息，從後面開始檢查，找到所有的 ToolMessage
    for msg in reversed(messages):
        # 注意: 這裡我們假設 ToolMessage 和 AIMessage 之間是清晰的
        if isinstance(msg, ToolMessage):
            tool_results.append(f"Tool {msg.name} Result: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            # 找到 AIMessage (即工具呼叫請求) 後停止，因為它標誌著工具結果的開始
            break

    # 將所有結果合併成一個字串
    # 反轉列表以保持工具結果的邏輯順序 (雖然在並行呼叫中順序不重要，但保持一致性較好)
    information_str = "\n".join(reversed(tool_results))
    prompt_str = """
    You have given a user information and you have to respond to user based on the information provided.

    Here is the user query:
    ---
    {user_query}
    ---
    Here is the information:
    ---
    {information}
    """
    prompt = ChatPromptTemplate.from_template(prompt_str)
    response_chain = prompt | model

    response = response_chain.invoke(
        {"user_query": user_query, "information": information_str}
    )
    return {"messages": [response]}


def pretty_print(state: AllState):
    response_prompt_str = """
    You are given a response, you have to reformat the response to be more friendly to user.

    Here is the response:
    ---
    {response}
    ---
    """
    response_prompt = ChatPromptTemplate.from_template(response_prompt_str)

    response_chain = response_prompt | model
    messages = state["messages"]
    response = response_chain.invoke({"response": messages[-1].content})
    return {"messages": [response]}


graph_builder = StateGraph(AllState)

graph_builder.add_node("semantic_agent", semantic_agent)
graph_builder.add_node("tool_agent", tool_agent)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("responder", responder)
graph_builder.add_node("pretty_print", pretty_print)

graph_builder.set_entry_point("semantic_agent")
graph_builder.add_conditional_edges("semantic_agent", semantic_agent_should_continue)
graph_builder.add_conditional_edges("tool_agent", tool_agent_should_continue)
graph_builder.add_edge("tools", "responder")
graph_builder.add_conditional_edges("responder", responder_should_continue)
graph_builder.set_finish_point("pretty_print")


graph = graph_builder.compile()


async def process_chunk():
    while True:
        user_input = input()
        inputs = {
            "messages": [HumanMessage(content=user_input)],
            "IsComplete": False,
        }
        chathistory.append(user_input)
        if user_input.lower() in ["quit", "exit", "q"]:
            print("掰啦!")
            print("-----")
            break
        async for chunk in graph.astream(inputs, stream_mode="updates"):
            # 顯示最新訊息的內容，並且漂亮顯示出
            for node, value in chunk.items():
                if "messages" in value:
                    print(f"Node: {node}, New Message: {value['messages'][-1].content}")
            print("-----")


if __name__ == "__main__":
    asyncio.run(process_chunk())

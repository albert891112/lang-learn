import asyncio
from dotenv import load_dotenv

from init_model import AllState, llm
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

load_dotenv()


class GET_NAME_BY_USERID(BaseModel):
    user_id: str = Field(description="使用者ID")


class GET_AGE_BY_USERID(BaseModel):
    user_id: str = Field(description="使用者ID")


def get_name_by_userid(user_id: str) -> str:
    """根據使用者ID獲取使用者名稱。"""
    users = {
        "user1": "Alice",
        "user2": "Bob",
        "user3": "Charlie",
    }
    return users.get(user_id, "Unknown User")


def get_age_by_userid(user_id: str) -> int:
    """根據使用者ID獲取使用者年齡。"""
    users = {
        "user1": 25,
        "user2": 30,
        "user3": 22,
    }
    return users.get(user_id, -1)


get_name_tool = StructuredTool.from_function(
    func=get_name_by_userid,
    name="get_name_by_userid",
    description="根據使用者ID獲取使用者名稱",
    args_schema=GET_NAME_BY_USERID,
)

get_age_tool = StructuredTool.from_function(
    func=get_age_by_userid,
    name="get_age_by_userid",
    description="根據使用者ID獲取使用者年齡",
    args_schema=GET_AGE_BY_USERID,
)

tool = [get_name_tool, get_age_tool]
tool2 = [get_name_tool, get_age_tool]
tool3 = [get_name_tool, get_age_tool]

tool_node = ToolNode(tools=tool)

model_with_tools = llm.bind_tools(tool)
model = llm


# agent
def tool_agent(state: AllState):
    response_chain = model_with_tools
    messages = state["messages"]
    response = response_chain.invoke(messages)
    return {"messages": [response]}


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


graph_builder = StateGraph(AllState)

graph_builder.add_node("tool_agent", tool_agent)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_node("responder", response)

graph_builder.set_entry_point("tool_agent")
graph_builder.add_edge("tool_agent", "tool_node")
graph_builder.add_edge("tool_node", "responder")
graph_builder.set_finish_point("responder")

graph = graph_builder.compile()


async def process_chunk():
    inputs = {
        "messages": [HumanMessage(content="請告訴我userid 為 user1的名字以及年齡")],
        "IsComplete": False,
    }

    async for chunk in graph.astream(inputs, stream_mode="values"):
        print(chunk["messages"][-1])


if __name__ == "__main__":
    asyncio.run(process_chunk())

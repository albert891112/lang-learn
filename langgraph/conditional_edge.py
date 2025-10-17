from init_model import AllState, llm
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph

# Define the model and prompt for extracting city names
model = llm
prompt_str = """
You are given one question and you have to extract city name from it
Don't respond anything except the city name and don't reply anything if you can't find city name
Only reply the city name if it exists or reply 'no_response' if there is no city name in question

  Here is the question:
  {user_query}
"""
prompt = ChatPromptTemplate.from_template(prompt_str)
chain = prompt | model

# Define the response generation chain
response_prompt_str = """
  You have given a weather information and you have to respond to user's query based on the information

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


def responder(state: AllState):
    messages = state["messages"]
    response = response_chain.invoke(
        {"user_query": messages[0], "information": messages[1]}
    )
    return {"messages": [response]}


def get_taiwan_weather(city: str) -> str:
    """查詢台灣特定城市的天氣狀況。"""
    weather_data = {
        "台北": "晴天，溫度28°C",
        "台中": "多雲，溫度26°C",
        "高雄": "陰天，溫度30°C",
    }
    return f"{city}的天氣：{weather_data.get(city, '暫無資料')}"


# Tool to call the model
def call_model(state: AllState):
    messages = state["messages"]
    response = chain.invoke(messages)
    return {"messages": [response]}


# Tool to get weather info
def weather_tool(state):
    context = state["messages"]
    city_name = context[1].content
    data = get_taiwan_weather(city_name)
    return {"messages": [data]}


# Define conditional edge function
def query_classify(state: AllState):
    messages = state["messages"]
    city_name = messages[1].content
    if city_name == "no_response":
        return "end"
    else:
        return "continue"


graph_builder = StateGraph(AllState)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("weather", weather_tool)
graph_builder.add_node("responder", responder)

# Define conditional edge based on model response
graph_builder.add_conditional_edges(
    "agent", query_classify, {"continue": "weather", "end": "responder"}
)

# Define normal edge
graph_builder.add_edge("weather", "responder")

graph_builder.set_entry_point("agent")
graph_builder.set_finish_point("responder")

app = graph_builder.compile()

init_state = {"messages": ["請問我家的天氣如何？"]}


if __name__ == "__main__":
    response = app.invoke(init_state)
    print(response["messages"][-1].content)

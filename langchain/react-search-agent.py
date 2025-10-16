from dotenv import load_dotenv
from init_model import llm

load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_tavily import TavilySearch


tool = [TavilySearch()]

model = llm
react_prompt = hub.pull("hwchase17/react")

react_agent = create_react_agent(
    llm=model,
    tools=tool,
    prompt=react_prompt,
)

chain = AgentExecutor.from_agent_and_tools(
    agent=react_agent,
    tools=tool,
    verbose=True,
)


def main():
    response = chain.invoke(
        input={"input": "What is the latest news about OpenAI and Microsoft?"}
    )
    print(response)


if __name__ == "__main__":
    main()

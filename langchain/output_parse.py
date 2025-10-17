from typing import List
from pydantic import BaseModel, Field
from init_model import llm
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_tavily import TavilySearch
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source document")


class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources"""

    answer: str = Field(description="The answer to the query")
    sources: List[Source] = Field(
        default_factory=list,  # 用來提供一個空的列表作為預設值
        description="List of sources used by the agent to generate the answer",
    )


react_promt_with_format_instruction = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question formatted according to format_instructions:{format_instructions}

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""


tool = [TavilySearch()]

model = llm


react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
output_extractor = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))

react_prompt_with_instructions = PromptTemplate(
    template=react_promt_with_format_instruction,
    input_variables=[
        "tools",
        "tool_names",
        "format_instructions",
        "input",
        "agent_scratchpad",
    ],
).partial(format_instructions=output_parser.get_format_instructions())


react_agent = create_react_agent(
    llm=model,
    tools=tool,
    prompt=react_prompt_with_instructions,
)

chain = (
    AgentExecutor.from_agent_and_tools(
        agent=react_agent,
        tools=tool,
        verbose=True,
    )
    | output_extractor
    | parse_output
)


def main():
    response = chain.invoke(
        input={"input": "What is the latest news about OpenAI and Microsoft?"}
    )
    print(response)


if __name__ == "__main__":
    main()

from dotenv import load_dotenv

load_dotenv()
from langchain_classic.agents.format_scratchpad.log import format_log_to_str
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, render_text_description, tool
from langchain_openai import ChatOpenAI


class ToolNotFoundError(Exception):
    pass


@tool
def get_text_length(txt: str) -> int:
    """Returns the number of characters in text"""
    print(f"Entered text: {txt}")
    txt = txt.strip("'\n").strip('"')
    return len(txt)


def find_tool_by_name(tool_name: str, tools: list[BaseTool]) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    else:
        raise ToolNotFoundError


if __name__ == "__main__":
    print("Starting the script...")
    tools = [get_text_length]
    template = """
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
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad} 
    """
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join(tool.name for tool in tools),
    )
    intermediate_values = []

    # llm = ChatOpenAI(temperature=0, model_kwargs={"stop": ["\nObservation"]})
    # llm = ChatOpenAI(temperature=0, stop=["\nObservation"])  # pyright: ignore[reportCallIssue]
    llm = ChatOpenAI(temperature=0, stop="Observation:")  # pyright: ignore[reportCallIssue]
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )
    agent_step = agent.invoke(
        {
            "input": "What is the text length of 'DOG' in charachters",
            "agent_scratchpad": intermediate_values,
        }
    )
    print(agent_step)

    if agent_step.type == "AgentAction":
        tool_name: str = agent_step.tool
        tool_to_use: BaseTool = find_tool_by_name(tool_name, tools)
        tool_input = agent_step.tool_input
        observation = tool_to_use.func(agent_step.tool_input)
        print(f"{observation=}")
        intermediate_values.append((agent_step, str(observation)))

    agent_step = agent.invoke(
        {
            "input": "What is the text length of 'DOG' in charachters",
            "agent_scratchpad": intermediate_values,
        }
    )

    if agent_step.type == "AgentFinish":
        print(agent_step.return_values)

from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

load_dotenv()

from react import llm, tools

SYSTEM_MESSAGE = """
You are a helpful assistant that can use toold to answer questions.
    """


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    # This is where you would implement the reasoning logic of your agent.
    # For example, you could use the state to determine which tools to call and how to process the results.
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    )
    return {"messages": [response]}


tool_node = ToolNode(tools)

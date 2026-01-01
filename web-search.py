from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

if __name__ == "__main__":
    print("starting the web search")
    input_msg = (
        "Give me 5 job descriptions for Python backend developers that are in EU"
    )
    result = agent.invoke({"messages": HumanMessage(content=input_msg)})
    print(result)

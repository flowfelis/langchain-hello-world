from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for url source"""

    url: str = Field(description="URL string for job application link")


class AgentResponse(BaseModel):
    answer: str = Field(description="Agent's response to query")
    sources: list[Source] = Field(
        default_factory=list,
        description="List of sources that shows urls for job applications",
    )


llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

if __name__ == "__main__":
    print("starting the web search")
    input_msg = (
        "Give me 5 job descriptions for Python backend developers that are in EU"
    )
    result = agent.invoke({"messages": HumanMessage(content=input_msg)})
    print(result)

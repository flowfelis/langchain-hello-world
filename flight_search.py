from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for url source"""

    url: str = Field(description="URL string for flight booking link")


class Flights(BaseModel):
    """Schema for flight search response"""

    flight_provider: str = Field(description="flight provider website name")
    sources: list[Source] = Field(
        default_factory=list,
        description="List of sources that shows urls for flight bookings",
    )


llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=Flights)

if __name__ == "__main__":
    print("starting the web search")
    input_msg = "Give me chaeapest 5 flight tickets with the URL, from Warsaw to Berlin on 25th February 2026 and return on 2nd March 2026"
    result = agent.invoke({"messages": HumanMessage(content=input_msg)})
    print(result)

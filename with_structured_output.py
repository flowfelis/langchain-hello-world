from dotenv import load_dotenv

load_dotenv()


from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from react_prompt_template import react_prompt_template
from schemas import AgentResponse

react_prompt = PromptTemplate(
    input_variables=["agent_scratchpad", "input", "tool_names", "tools"],
    template=react_prompt_template,
).partial(format_instructions="")

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
structured_llm = llm.with_structured_output(AgentResponse)
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
chain = agent_executor | extract_output | structured_llm


def main():
    print("Starting the script...")
    result = chain.invoke(
        input={
            "input": "Give me 5 job descriptions for Python backend developers that are in EU"
        }
    )
    print(result)


if __name__ == "__main__":
    main()

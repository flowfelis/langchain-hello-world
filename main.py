from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from langchain_ollama import ChatOllama

load_dotenv()


def main():
    information = """
        Elon Reeve Musk[b] (born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest person in the world since 2021; as of December 2025, Forbes estimates his net worth to be around $754 billion
        """

    summary_template = """       
        given the information {information} about a person I want you to create:
        1. A short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOllama(
        temperature=0,
        model="gemma3:270m",
    )

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response.content)


if __name__ == "__main__":
    main()

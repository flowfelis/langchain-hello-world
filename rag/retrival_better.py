import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

from langchain_core import embeddings
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings()
# llm = ChatOpenAI(model="gpt-5.2")
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

template = """
Answer the questions based only on the following context:

{context}

Question: {question}

Provide a detailed answer:
"""

prompt_template = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    "Format retrieved documents into a single string."
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_with_lcel():
    """
    Create a retrievla chain using LCEL.
    Returns a chain that can be invoked with {'question': '...'}
    """
    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    print("Retrieving...")

    query = "What is pinecone in machine learning"

    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print(result_raw)
    #
    result_with_lcel = retrieval_chain_with_lcel()
    result = result_with_lcel.invoke(
        {"question": "What is pinecone in machine learning"}
    )
    print(result)

from langchain_classic.agents import AgentExecutor, create_react_agent

from openrouter_llm import OpenRouterLLM
from prompts import REACT_PROMPT
from tools import TOOLS

from dotenv import load_dotenv

load_dotenv()


def main():
    llm = OpenRouterLLM(
        model="openai/gpt-4o-mini",
        temperature=0,
    )

    agent = create_react_agent(
        llm=llm,
        tools=TOOLS,
        prompt=REACT_PROMPT,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        handle_parsing_errors=True,
    )

    #query = "What is the definition of lactate blood?"
    #query = "What is the relationship between initial lactate level and 28-day mortality in septic shock?"
    query = "What are the earliest clinical signs of sepsis in hospitalized patients?"
    result_wiki = agent_executor.invoke(
        {
            "input": f"You are an expert in sepsis. Read the question that I am about to ask you. Use Wikipedia to answer this question: {query}. Relate this question to sepsis. Make sure that you dissect the question correctly. If you can't find any solution, try finding something that is similar to it."
        }
    )

    result_arxiv = agent_executor.invoke(
        {
            "input": f"You are an expert in sepsis. Read the question that I am about to ask you. Use arXiv to find relevant papers for this question: {query}. Relate this question to sepsis. Make sure that you dissect the question correctly. If you can't find any solution, try finding something that is similar to it."
        }
    )

    print("\nWikipedia answer: \n")
    print(result_wiki["output"])

    print("\nArXiv answer: \n")
    print(result_arxiv["output"])


if __name__ == "__main__":
    main()

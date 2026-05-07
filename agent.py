"""Simple ReAct agent using ChatOpenAI."""
from __future__ import annotations

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from index import read_index_entries
from langchain_core.tools import tool
from retrieve import get_paper_markdown, search_literature


@tool
def get_study_index() -> str:
    """Return the full study index listing all indexed papers with their metadata,

    Use this tool for getting a comprehensive overview of all studies currently indexed in the system, including their designs, cohorts, interventions, outcomes,

    before deciding which to search in depth."""
    entries = read_index_entries()
    if not entries:
        return "The study index is empty."
    header = f"Study index — {len(entries)} paper(s):\n\n"
    return header + "\n\n---\n\n".join(entries)

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=4096,
    max_retries=2,
)

agent = create_agent(
    model=llm,
    tools=[search_literature, get_paper_markdown, get_study_index],
    checkpointer=MemorySaver(),
    middleware=[
        ToolCallLimitMiddleware(
            tool_name="search_literature",
            thread_limit=None,
            run_limit=1,
        )
    ],
    system_prompt=(
   "You are a research assistant with access to a scientific literature database. "
    "Use the search_literature tool to look up information before answering. "
    "Forward the answer UNCHANGED to the user from the retrieved literature. "

    "You have the following tools at your disposal:\n"
    "1. search_literature(query: str) -> list[dict]: Search for semantically relevant text chunks from the paper corpus and return a ready-formatted answer, which is passed unchanged immediately to the user. "
    "2. get_study_index() -> str: Return the full study index listing all indexed papers with their metadata, including study design, cohort size, interventions, outcomes, and key findings. Use this tool for getting a comprehensive overview of all studies currently indexed in the system, including their designs, cohorts, interventions, outcomes, and key findings, before deciding which to search in depth. "
    "3. get_paper_markdown(paper_name: str) -> str: Return the entire content of a paper as markdown text, given the paper title or a close approximation. Use this tool if you want to go deep on a specific paper, read its full content, or extract information that may not have been included in the retrieved chunks. Input should be the exact paper title or a close approximation to it. "

    "Use the get_paper_markdown tool after the get_study_index tool to deep dive into specific papers if asked by the user."
    ),
)

if __name__ == "__main__":
    questions = [
        "What were the main findings of the Sepsis-3 study in Brazil?",
        "How does Sepsis-3 compare to Sepsis-2 in predicting ICU mortality?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        print(f"\nA: {result['messages'][-1].content}")
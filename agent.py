"""Simple ReAct agent using ChatOpenRouter."""
from __future__ import annotations

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter

from retrieve import search_literature

load_dotenv()

llm = ChatOpenRouter(
    model="anthropic/claude-sonnet-4.5",
    temperature=0,
    max_tokens=4096,
    max_retries=2,
)

agent = create_agent(
    model=llm,
    tools=[search_literature],
    system_prompt=(
        "You are a research assistant with access to a scientific literature database. "
        "Use the search_literature tool to look up information before answering. "
        "Always cite the sources returned by the tool in your final answer."
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

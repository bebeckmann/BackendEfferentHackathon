import os
import requests
from typing import Optional, List

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.tools import tool
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent

# ----------------------------
# 1. OpenRouter LLM wrapper
# ----------------------------

class OpenRouterLLM(LLM):
    model: str = "openai/gpt-4o-mini"
    url: str = "https://openrouter.ai/api/v1/chat/completions"

    @property
    def _llm_type(self) -> str:
        return "openrouter-chat-completions"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        headers = {
            "Authorization": f"Bearer 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",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        if stop:
            data["stop"] = stop

        response = requests.post(
            self.url,
            headers=headers,
            json=data,
            timeout=60,
        )

        response.raise_for_status()
        payload = response.json()

        return payload["choices"][0]["message"]["content"]


# ----------------------------
# 2. Two very simple tools
# ----------------------------

import re
from langchain_core.tools import tool

@tool
def add_numbers(text: str) -> str:
    """
    Add two numbers.

    Input may look like:
    - 3, 5
    - '3, 5'
    - add 3 and 5
    """
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)

    if len(numbers) < 2:
        return "Please provide two numbers, like: 3, 5"

    a = float(numbers[0])
    b = float(numbers[1])

    result = a + b

    if result.is_integer():
        return str(int(result))

    return str(result)


@tool
def reverse_text(text: str) -> str:
    """
    Reverse the given text.
    """
    return text[::-1]


# ----------------------------
# 3. Create the agent
# ----------------------------

def main():
    llm = OpenRouterLLM(
        model="openai/gpt-4o-mini",
    )

    tools = [
        add_numbers,
        reverse_text,
    ]

    # Standard ReAct prompt from LangChain Hub
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate.from_template("""
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... this Thought/Action/Action Input/Observation can repeat
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """)

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    result = agent_executor.invoke(
        {
            "input": "Add 12 and 30, then reverse the text 'hello'."
        }
    )

    print("\nFinal answer:")
    print(result["output"])


if __name__ == "__main__":
    main()
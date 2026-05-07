import os
from typing import Optional, List

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM


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
        api_key = os.environ["OPENROUTER_API_KEY"]

        headers = {
            "Authorization": f"Bearer {api_key}",
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
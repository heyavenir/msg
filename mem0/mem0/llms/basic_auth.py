import base64
import json
import os
from typing import Dict, List, Optional, Union

import httpx

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.basic_auth import BasicAuthLlmConfig
from mem0.llms.base import LLMBase


class BasicAuthLLM(LLMBase):
    """
    LLM provider for OpenAI-compatible /chat/completions APIs that use Basic Auth.
    API key is Base64-encoded and sent as Authorization: Basic <encoded_key>.
    """

    def __init__(self, config: Optional[Union[BasicAuthLlmConfig, Dict]] = None):
        if config is None:
            config = BasicAuthLlmConfig()
        elif isinstance(config, dict):
            config = BasicAuthLlmConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, BasicAuthLlmConfig):
            config = BasicAuthLlmConfig(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
            )

        super().__init__(config)

        base_url = self.config.base_url or os.getenv("BASIC_AUTH_LLM_BASE_URL")
        api_key = self.config.api_key or os.getenv("BASIC_AUTH_LLM_API_KEY")

        if not base_url:
            raise ValueError("base_url is required. Set it in config or BASIC_AUTH_LLM_BASE_URL env var.")
        if not api_key:
            raise ValueError("api_key is required. Set it in config or BASIC_AUTH_LLM_API_KEY env var.")

        encoded = base64.b64encode(api_key.encode()).decode()
        self.headers = {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
        }
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(headers=self.headers, timeout=60)

    def _parse_response(self, response: dict, tools: Optional[List[Dict]]) -> Union[str, dict]:
        message = response["choices"][0]["message"]
        if tools:
            processed = {"content": message.get("content", ""), "tool_calls": []}
            for tc in message.get("tool_calls") or []:
                processed["tool_calls"].append({
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"]),
                })
            return processed
        return message.get("content", "")

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> Union[str, dict]:
        payload: Dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        payload.update(kwargs)

        resp = self.client.post(f"{self.base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        return self._parse_response(resp.json(), tools)

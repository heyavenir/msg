from typing import Optional, Union, Dict

from mem0.configs.llms.base import BaseLlmConfig


class BasicAuthLlmConfig(BaseLlmConfig):
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        http_client_proxies: Optional[Union[Dict, str]] = None,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            api_key=None,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            enable_vision=enable_vision,
            vision_details=vision_details,
            http_client_proxies=http_client_proxies,
        )
        self.base_url = base_url
        self.api_key = api_key

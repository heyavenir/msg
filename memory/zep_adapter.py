"""
Zep Memory adapter
설치: pip install zep-python
"""
from typing import Dict, List, Optional

from memory.base import BaseMemory, MemoryResult


class ZepMemory(BaseMemory):

    name = "zep"

    def __init__(self, api_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        try:
            from zep_python import ZepClient
        except ImportError:
            raise ImportError("pip install zep-python")

        self._client = ZepClient(base_url=api_url, api_key=api_key)

    def add(self, content: str, user_id: str, **kwargs) -> None:
        from zep_python.memory import Message, Memory

        session_id = kwargs.get("session_id", user_id)
        message = Message(role="human", content=content)
        self._client.memory.add_memory(session_id, Memory(messages=[message]))

    def search(self, query: str, user_id: str, top_k: int = 5, **kwargs) -> List[MemoryResult]:
        from zep_python.memory import MemorySearchPayload

        session_id = kwargs.get("session_id", user_id)
        payload = MemorySearchPayload(text=query, search_scope="summary")
        results = self._client.memory.search_memory(session_id, payload, limit=top_k)

        return [
            MemoryResult(
                content=r.summary.content if r.summary else "",
                score=r.dist or 0.0,
            )
            for r in results
        ]

    def reset(self, user_id: str) -> None:
        session_id = user_id
        self._client.memory.delete_memory(session_id)

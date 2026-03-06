from typing import Dict, List, Optional

from memory.base import BaseMemory, MemoryResult


class Mem0Memory(BaseMemory):

    name = "mem0"

    def __init__(self, config: Dict):
        """
        config: mem0 Memory.from_config()에 넘기는 딕셔너리
        예시는 config_example.py 참고
        """
        from mem0 import Memory
        self._mem = Memory.from_config(config)

    def add(self, content: str, user_id: str, **kwargs) -> None:
        self._mem.add(content, user_id=user_id, **kwargs)

    def search(self, query: str, user_id: str, top_k: int = 5, **kwargs) -> List[MemoryResult]:
        results = self._mem.search(query, user_id=user_id, limit=top_k, **kwargs)
        return [
            MemoryResult(
                content=r["memory"],
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

    def reset(self, user_id: str) -> None:
        self._mem.delete_all(user_id=user_id)

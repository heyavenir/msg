"""
Graphiti dry-run Memory adapter
- KG(그래프 DB) 설정 없이 graphiti LLM extraction pipeline만 실행
- add() 시 entities/facts 추출 결과를 출력하고 로컬에 저장
- search() 시 로컬 저장된 결과에서 검색
설치: pip install graphiti-core openai
"""
import asyncio
import sys
import types
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, "libs/graphiti")

from memory.base import BaseMemory, MemoryResult


class GraphitiDryRunMemory(BaseMemory):
    """
    graphiti extraction pipeline dry-run 어댑터.

    add() 호출마다 LLM으로 entities/facts를 추출해 출력하고,
    그래프 DB에 저장하지 않고 로컬 메모리에만 보관.

    config 예시:
        {
            "base_url": "http://localhost:11434/v1",  # Ollama 등 OpenAI 호환 엔드포인트
            "api_key": "ollama",
            "model": "qwen:32b",
        }
    """

    name = "graphiti_dryrun"

    def __init__(self, config: Dict):
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        llm_cfg = LLMConfig(
            api_key=config.get("api_key", "ollama"),
            model=config.get("model"),
            base_url=config.get("base_url"),
        )
        self._llm = OpenAIGenericClient(config=llm_cfg)
        # extract_nodes/extract_edges는 clients.llm_client만 사용하므로 SimpleNamespace로 충분
        self._clients = types.SimpleNamespace(llm_client=self._llm)
        # user_id → list[dict] (entities + facts)
        self._store: Dict[str, List[Dict]] = {}

    # ------------------------------------------------------------------ #
    #  내부 async 추출 로직                                                #
    # ------------------------------------------------------------------ #

    async def _extract_async(self, content: str, user_id: str):
        from graphiti_core.nodes import EpisodicNode, EpisodeType
        from graphiti_core.utils.maintenance.edge_operations import extract_edges
        from graphiti_core.utils.maintenance.node_operations import extract_nodes

        episode = EpisodicNode(
            name=f"ep_{user_id}_{datetime.now(tz=timezone.utc).isoformat()}",
            group_id=user_id,
            source=EpisodeType.message,
            source_description="conversation",
            content=content,
            valid_at=datetime.now(tz=timezone.utc),
        )

        nodes = await extract_nodes(self._clients, episode, [])
        edges = await extract_edges(self._clients, episode, nodes, [], {})
        return nodes, edges

    def _run(self, coro):
        """동기 컨텍스트에서 async 코루틴 실행."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Jupyter 등 이미 루프가 돌고 있는 경우 별도 스레드에서 실행
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, coro).result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # ------------------------------------------------------------------ #
    #  BaseMemory interface                                                #
    # ------------------------------------------------------------------ #

    def add(self, content: str, user_id: str, **kwargs) -> None:
        nodes, edges = self._run(self._extract_async(content, user_id))

        print(f"\n[graphiti extraction] user={user_id}")
        print(f"  entities ({len(nodes)}):")
        for n in nodes:
            labels = ", ".join(n.labels) if n.labels else "Entity"
            print(f"    - {n.name}  [{labels}]")
        print(f"  facts ({len(edges)}):")
        for e in edges:
            print(f"    - {e.fact}")

        store = self._store.setdefault(user_id, [])
        for n in nodes:
            store.append({"type": "entity", "text": n.name, "labels": n.labels, "summary": n.summary})
        for e in edges:
            store.append({"type": "fact", "text": e.fact, "relation": e.name})

    def search(self, query: str, user_id: str, top_k: int = 5, **kwargs) -> List[MemoryResult]:
        keywords = query.lower().split()
        results = []
        for item in self._store.get(user_id, []):
            text = item["text"]
            if any(kw in text.lower() for kw in keywords):
                results.append(
                    MemoryResult(content=text, score=1.0, metadata=item)
                )
        return results[:top_k]

    def reset(self, user_id: str) -> None:
        self._store.pop(user_id, None)

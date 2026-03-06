from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryResult:
    """메모리 검색 결과 하나"""
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMemory(ABC):
    """
    모든 메모리 시스템이 구현해야 하는 인터페이스.
    mem0, zep, 커스텀 등 어떤 시스템이든 이 클래스를 상속.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """메모리 시스템 이름 (결과 저장 시 사용)"""
        pass

    @abstractmethod
    def add(self, content: str, user_id: str, **kwargs) -> None:
        """대화 내용을 메모리에 저장"""
        pass

    @abstractmethod
    def search(self, query: str, user_id: str, top_k: int = 5, **kwargs) -> List[MemoryResult]:
        """쿼리와 관련된 메모리 검색"""
        pass

    @abstractmethod
    def reset(self, user_id: str) -> None:
        """특정 유저의 메모리 전체 삭제 (세션 간 격리)"""
        pass

    def reset_all(self) -> None:
        """전체 메모리 초기화 (선택 구현)"""
        raise NotImplementedError(f"{self.name} does not support reset_all()")

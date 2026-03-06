from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    """대화 한 턴"""
    role: str       # "user" | "assistant"
    content: str
    timestamp: Optional[str] = None


@dataclass
class QAPair:
    """벤치마크 평가용 질문-정답 쌍"""
    question: str
    answer: str                          # ground truth
    session_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """한 사용자의 대화 세션"""
    session_id: str
    turns: List[Turn]
    qa_pairs: List[QAPair] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseBenchmark(ABC):
    """
    모든 벤치마크가 구현해야 하는 인터페이스.
    새 벤치마크 추가 시 이 클래스를 상속.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """벤치마크 이름 (결과 저장 시 사용)"""
        pass

    @abstractmethod
    def load(self, data_path: Optional[str] = None) -> None:
        """
        데이터셋 로드.
        data_path: 로컬 경로 또는 None (HuggingFace 자동 다운로드)
        """
        pass

    @abstractmethod
    def get_sessions(self) -> List[Session]:
        """평가용 세션 목록 반환"""
        pass

    @abstractmethod
    def evaluate(self, predictions: List[str], references: List[QAPair]) -> Dict[str, float]:
        """
        예측값과 정답 비교해서 점수 계산.

        Args:
            predictions: 모델이 생성한 답변 리스트
            references: 대응하는 QAPair 리스트

        Returns:
            {"f1": 0.xx, "em": 0.xx, ...} 형태의 딕셔너리
        """
        pass

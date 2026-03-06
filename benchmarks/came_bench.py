"""
CAME-Bench (Comprehensive Assessment of Memory Evaluation)
논문/데이터셋 공개 여부에 따라 load() 구현 업데이트 필요.
"""
from typing import Dict, List, Optional

from benchmarks.base import BaseBenchmark, QAPair, Session, Turn
from eval.metrics import f1_score, exact_match


class CAMEBenchmark(BaseBenchmark):

    name = "came_bench"

    def __init__(self, split: str = "test"):
        self.split = split
        self._sessions: List[Session] = []

    def load(self, data_path: Optional[str] = None) -> None:
        """
        CAME-Bench 데이터셋 로드.
        공식 데이터셋 경로 확정 후 _load_from_hf() 구현 업데이트.
        """
        if data_path:
            self._load_from_file(data_path)
        else:
            self._load_from_hf()

    def _load_from_hf(self) -> None:
        # TODO: 공식 HuggingFace dataset ID 확인 후 업데이트
        raise NotImplementedError(
            "CAME-Bench HuggingFace dataset ID를 확인 후 구현하세요. "
            "로컬 파일은 load(data_path='path/to/data.json')으로 사용 가능."
        )

    def _load_from_file(self, path: str) -> None:
        import json
        with open(path) as f:
            data = json.load(f)
        self._sessions = [self._parse_row(row) for row in data]

    def _parse_row(self, row: dict) -> Session:
        turns = []
        for t in row.get("conversation", []):
            turns.append(Turn(
                role=t["role"],
                content=t["content"],
                timestamp=t.get("timestamp"),
            ))

        qa_pairs = []
        for qa in row.get("qa_pairs", []):
            qa_pairs.append(QAPair(
                question=qa["question"],
                answer=qa["answer"],
                session_id=row["session_id"],
                metadata=qa.get("metadata", {}),
            ))

        return Session(
            session_id=row["session_id"],
            turns=turns,
            qa_pairs=qa_pairs,
        )

    def get_sessions(self) -> List[Session]:
        if not self._sessions:
            raise RuntimeError("Call load() first.")
        return self._sessions

    def evaluate(self, predictions: List[str], references: List[QAPair]) -> Dict[str, float]:
        assert len(predictions) == len(references)

        f1_scores, em_scores = [], []
        for pred, ref in zip(predictions, references):
            f1_scores.append(f1_score(pred, ref.answer))
            em_scores.append(exact_match(pred, ref.answer))

        return {
            "f1": sum(f1_scores) / len(f1_scores),
            "em": sum(em_scores) / len(em_scores),
            "num_questions": len(predictions),
        }

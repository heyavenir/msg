"""
LoCoMo Benchmark
HuggingFace: https://huggingface.co/datasets/snap-research/LoCoMo
"""
from typing import Dict, List, Optional

from benchmarks.base import BaseBenchmark, QAPair, Session, Turn
from eval.metrics import f1_score, exact_match


class LoCoMoBenchmark(BaseBenchmark):

    name = "locomo"

    def __init__(self, split: str = "test"):
        self.split = split
        self._sessions: List[Session] = []

    def load(self, data_path: Optional[str] = None) -> None:
        """
        data_path=None 이면 HuggingFace에서 자동 다운로드.
        data_path 지정 시 로컬 JSON 파일 사용.
        """
        if data_path:
            self._load_from_file(data_path)
        else:
            self._load_from_hf()

    def _load_from_hf(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        dataset = load_dataset("snap-research/LoCoMo", split=self.split)
        self._sessions = [self._parse_row(row) for row in dataset]

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
                metadata={"type": qa.get("type", "single-hop")},
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

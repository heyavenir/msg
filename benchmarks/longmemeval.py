"""
LongMemEval Benchmark
HuggingFace: https://huggingface.co/datasets/xiaowu0162/longmemeval
"""
from typing import Dict, List, Optional

from benchmarks.base import BaseBenchmark, QAPair, Session, Turn
from eval.metrics import f1_score, exact_match


class LongMemEvalBenchmark(BaseBenchmark):

    name = "longmemeval"

    # 질문 유형별 분리 평가 지원
    QUESTION_TYPES = [
        "single-session-user",
        "single-session-assistant",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
    ]

    def __init__(self, split: str = "test", subset: Optional[str] = None):
        """
        subset: 특정 question type만 평가할 때 지정
        """
        self.split = split
        self.subset = subset
        self._sessions: List[Session] = []

    def load(self, data_path: Optional[str] = None) -> None:
        if data_path:
            self._load_from_file(data_path)
        else:
            self._load_from_hf()

    def _load_from_hf(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        dataset = load_dataset("xiaowu0162/longmemeval", split=self.split)
        rows = dataset if not self.subset else [r for r in dataset if r.get("type") == self.subset]
        self._sessions = [self._parse_row(row) for row in rows]

    def _load_from_file(self, path: str) -> None:
        import json
        with open(path) as f:
            data = json.load(f)
        rows = data if not self.subset else [r for r in data if r.get("type") == self.subset]
        self._sessions = [self._parse_row(row) for row in rows]

    def _parse_row(self, row: dict) -> Session:
        turns = []
        for t in row.get("history", []):
            turns.append(Turn(
                role=t["role"],
                content=t["content"],
                timestamp=t.get("date"),
            ))

        qa_pairs = [QAPair(
            question=row["question"],
            answer=row["answer"],
            session_id=row["question_id"],
            metadata={"type": row.get("type", "")},
        )]

        return Session(
            session_id=row["question_id"],
            turns=turns,
            qa_pairs=qa_pairs,
            metadata={"type": row.get("type", "")},
        )

    def get_sessions(self) -> List[Session]:
        if not self._sessions:
            raise RuntimeError("Call load() first.")
        return self._sessions

    def evaluate(self, predictions: List[str], references: List[QAPair]) -> Dict[str, float]:
        assert len(predictions) == len(references)

        results: Dict[str, list] = {t: [] for t in self.QUESTION_TYPES}
        results["overall"] = []

        for pred, ref in zip(predictions, references):
            score = f1_score(pred, ref.answer)
            q_type = ref.metadata.get("type", "")
            if q_type in results:
                results[q_type].append(score)
            results["overall"].append(score)

        return {
            k: (sum(v) / len(v) if v else 0.0)
            for k, v in results.items()
        }

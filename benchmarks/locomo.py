"""
LoCoMo Benchmark
데이터: https://github.com/snap-research/locomo (locomo10.json)

실제 데이터 구조:
- sample_id: "conv-XX"
- conversation: {speaker_a, speaker_b, session_1_date_time, session_1: [{dia_id, speaker, text}], ...}
- qa: [{question, answer, category, evidence}]
  category: 1=single-hop, 2=temporal, 3=multi-hop, 4=open-domain, 5=adversarial
"""
from typing import Dict, List, Optional

from benchmarks.base import BaseBenchmark, QAPair, Session, Turn
from eval.metrics import f1_score, exact_match

CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


class LoCoMoBenchmark(BaseBenchmark):

    name = "locomo"

    DEFAULT_DATA_PATH = "data/locomo/locomo10.json"

    def __init__(self):
        self._sessions: List[Session] = []

    def load(self, data_path: Optional[str] = None) -> None:
        """
        data_path 기본값: data/locomo/locomo10.json
        GitHub: https://github.com/snap-research/locomo
        """
        path = data_path or self.DEFAULT_DATA_PATH
        self._load_from_file(path)

    def _load_from_file(self, path: str) -> None:
        import json
        with open(path) as f:
            data = json.load(f)
        self._sessions = [self._parse_row(row) for row in data]
        print(f"LoCoMo 로드 완료: {len(self._sessions)}개 대화, "
              f"{sum(len(s.qa_pairs) for s in self._sessions)}개 QA")

    def _parse_row(self, row: dict) -> Session:
        conv = row["conversation"]
        speaker_a = conv.get("speaker_a", "A")
        speaker_b = conv.get("speaker_b", "B")

        # session_1, session_2, ... 순서대로 turns 수집
        turns = []
        session_keys = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda k: int(k.split("_")[1])
        )
        for sess_key in session_keys:
            date = conv.get(f"{sess_key}_date_time", "")
            for t in conv[sess_key]:
                if "text" not in t:
                    continue
                speaker = t["speaker"]
                role = "user" if speaker == speaker_a else "assistant"
                turns.append(Turn(
                    role=role,
                    content=f"{speaker}: {t['text']}",
                    timestamp=date,
                ))

        qa_pairs = []
        for qa in row.get("qa", []):
            # category 5 (adversarial)는 answer 키가 없고 adversarial_answer 사용
            answer = qa.get("answer") or qa.get("adversarial_answer", "")
            qa_pairs.append(QAPair(
                question=qa["question"],
                answer=str(answer),
                session_id=row["sample_id"],
                metadata={
                    "category": qa.get("category"),
                    "category_name": CATEGORY_NAMES.get(qa.get("category"), "unknown"),
                    "evidence": qa.get("evidence", []),
                },
            ))

        return Session(
            session_id=row["sample_id"],
            turns=turns,
            qa_pairs=qa_pairs,
            metadata={"speaker_a": speaker_a, "speaker_b": speaker_b},
        )

    def get_sessions(self) -> List[Session]:
        if not self._sessions:
            raise RuntimeError("Call load() first.")
        return self._sessions

    def evaluate(self, predictions: List[str], references: List[QAPair]) -> Dict[str, float]:
        assert len(predictions) == len(references)

        by_category: Dict[str, list] = {name: [] for name in CATEGORY_NAMES.values()}
        overall_f1, overall_em = [], []

        for pred, ref in zip(predictions, references):
            f1 = f1_score(pred, ref.answer)
            em = exact_match(pred, ref.answer)
            cat_name = ref.metadata.get("category_name", "unknown")
            if cat_name in by_category:
                by_category[cat_name].append(f1)
            overall_f1.append(f1)
            overall_em.append(em)

        results = {
            "f1": sum(overall_f1) / len(overall_f1),
            "em": sum(overall_em) / len(overall_em),
            "num_questions": len(predictions),
        }
        for cat_name, scores in by_category.items():
            if scores:
                results[f"f1_{cat_name}"] = sum(scores) / len(scores)

        return results

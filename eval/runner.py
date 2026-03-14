"""
EvalRunner: benchmark × memory × llm 조합 실행 및 결과 저장
"""
import csv
import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from benchmarks.base import BaseBenchmark, QAPair, Session
from memory.base import BaseMemory


class EvalRunner:

    def __init__(
        self,
        benchmark: BaseBenchmark,
        memory: BaseMemory,
        answer_fn: Callable[[str, List[str]], str],
        results_dir: str = "results",
    ):
        """
        Args:
            benchmark: 평가할 벤치마크
            memory: 사용할 메모리 시스템
            answer_fn: (question, retrieved_memories) -> answer 함수
                       LLM 호출 로직을 여기에 넣으면 됨
            results_dir: 결과 저장 디렉토리
        """
        self.benchmark = benchmark
        self.memory = memory
        self.answer_fn = answer_fn
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def run(
        self,
        sessions: Optional[List[Session]] = None,
        top_k: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        전체 평가 실행.

        Returns:
            {"scores": {...}, "predictions": [...], "references": [...]}
        """
        sessions = sessions or self.benchmark.get_sessions()
        all_predictions: List[str] = []
        all_references: List[QAPair] = []

        for i, session in enumerate(sessions):
            if verbose:
                print(f"[{i+1}/{len(sessions)}] session_id={session.session_id}")

            # 세션 시작 전 해당 유저 메모리 초기화
            self.memory.reset(session.session_id)

            # 대화 내용을 메모리에 순서대로 저장
            for turn in session.turns:
                self.memory.add(
                    content=turn.content,
                    user_id=session.session_id,
                )

            # 각 QA 평가
            for qa in session.qa_pairs:
                retrieved = self.memory.search(
                    query=qa.question,
                    user_id=session.session_id,
                    top_k=top_k,
                )
                context = [r.content for r in retrieved]
                prediction = self.answer_fn(qa.question, context)

                all_predictions.append(prediction)
                all_references.append(qa)

        scores = self.benchmark.evaluate(all_predictions, all_references)

        result = {
            "benchmark": self.benchmark.name,
            "memory": self.memory.name,
            "timestamp": datetime.now().isoformat(),
            "scores": scores,
            "predictions": all_predictions,
            "references": [
                {"question": r.question, "answer": r.answer, "session_id": r.session_id}
                for r in all_references
            ],
        }

        self._save(result)
        return result

    def _save(self, result: Dict[str, Any]) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{result['benchmark']}_{result['memory']}_{ts}"

        # JSON (전체)
        json_path = os.path.join(self.results_dir, f"{base}.json")
        with open(json_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # CSV (점수 요약)
        csv_path = os.path.join(self.results_dir, "summary.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "benchmark", "memory"] + list(result["scores"].keys()))
            writer.writerow(
                [result["timestamp"], result["benchmark"], result["memory"]]
                + list(result["scores"].values())
            )

        print(f"결과 저장: {json_path}")
        print(f"점수: {result['scores']}")

"""
실험 1: Gemini 생성 일관성 테스트 (Generation Consistency)

[Pipeline - 실제 서비스 시뮬레이션]
각 호출마다 독립적으로 context를 fetch하고 임베딩을 생성.
동일 키워드라도 매 호출마다 context가 다를 수 있는 현실을 반영.

1. Context Fetching: 각 run마다 google_search로 context를 독립 확보
2. Semantic Enrichment: 각 context를 바탕으로 영문 프로필 생성
3. Local Embedding: Qwen-0.6B로 임베딩
4. Similarity Analysis: 5개 결과 간 코사인 유사도 측정 → 일관성 판정

통과 기준: 평균 코사인 유사도 >= 0.98

실행 방법:
    export GEMINI_ENDPOINT="https://your-endpoint/v1"
    export GEMINI_BEARER_TOKEN="your-bearer-token"
    python embeddings/exp1_consistency.py
"""

import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from typing import List

sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    RESULTS_DIR,
    call_gemini,
    fetch_context,
    get_cosine_similarity,
    get_embedding,
)

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

KEYWORDS: List[str] = ["도마뱀", "말차", "토트넘"]
NUM_RUNS: int = 5
COSINE_THRESHOLD: float = 0.98

PROMPT_TEMPLATE: str = (
    "Based on the following context about '{keyword}':\n"
    "{context}\n\n"
    "Write exactly 2 factual sentences about '{keyword}'. "
    "Start the first sentence with 'A {keyword} is' or '{keyword} is'. "
    "Do not add any preamble, headers, bullet points, or closing remarks. "
    "Output only the 2 sentences, nothing else."
)


# ---------------------------------------------------------------------------
# 데이터클래스
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """단일 run 기록 — 어떤 context가 꽂혔는지 포함"""
    run_index: int
    context: str    # 이 run에서 fetch된 context
    text: str       # Gemini가 생성한 영문 텍스트


@dataclass
class KeywordConsistencyResult:
    """키워드별 NUM_RUNS회 실행 집계 결과"""
    keyword: str
    runs: List[RunRecord]           # run별 (context, text) 전체 기록
    avg_word_overlap: float         # Jaccard 유사도 평균 (C(5,2)=10 쌍)
    avg_cosine_similarity: float    # 코사인 유사도 평균
    min_cosine_similarity: float    # 코사인 유사도 최솟값
    passed: bool                    # avg_cosine_similarity >= COSINE_THRESHOLD


# ---------------------------------------------------------------------------
# 유틸 함수
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def word_overlap_rate(text_a: str, text_b: str) -> float:
    """단어 집합 기준 Jaccard 유사도"""
    words_a = set(_normalize_text(text_a).split())
    words_b = set(_normalize_text(text_b).split())
    if not words_a and not words_b:
        return 1.0
    return len(words_a & words_b) / len(words_a | words_b)


# ---------------------------------------------------------------------------
# 실험 실행
# ---------------------------------------------------------------------------

def run_experiment() -> List[KeywordConsistencyResult]:
    results: List[KeywordConsistencyResult] = []

    for keyword in KEYWORDS:
        print(f"\n{'='*60}")
        print(f"[{keyword}] — {NUM_RUNS}회 독립 실행")
        print(f"{'='*60}")

        runs: List[RunRecord] = []

        for i in range(NUM_RUNS):
            print(f"\n  --- run {i + 1}/{NUM_RUNS} ---")

            # 매 run마다 독립적으로 context fetch (실제 서비스 시뮬레이션)
            context = fetch_context(keyword)

            prompt = PROMPT_TEMPLATE.format(keyword=keyword, context=context)
            text = call_gemini(prompt)

            runs.append(RunRecord(run_index=i, context=context, text=text))
            print(f"  context: {context[:80]}...")
            print(f"  text:    {text[:80]}...")

        # 임베딩 계산
        print(f"\n  임베딩 계산 중...")
        texts = [r.text for r in runs]
        embeddings = [get_embedding(t, pooling="mean") for t in texts]

        # C(5,2)=10 쌍 유사도
        pairs = list(combinations(range(NUM_RUNS), 2))
        overlap_scores = [word_overlap_rate(texts[i], texts[j]) for i, j in pairs]
        cosine_scores  = [get_cosine_similarity(embeddings[i], embeddings[j]) for i, j in pairs]

        avg_overlap = sum(overlap_scores) / len(overlap_scores)
        avg_cosine  = sum(cosine_scores)  / len(cosine_scores)
        min_cosine  = min(cosine_scores)
        passed      = avg_cosine >= COSINE_THRESHOLD

        results.append(KeywordConsistencyResult(
            keyword=keyword,
            runs=runs,
            avg_word_overlap=avg_overlap,
            avg_cosine_similarity=avg_cosine,
            min_cosine_similarity=min_cosine,
            passed=passed,
        ))
        print(f"\n  → avg_cosine={avg_cosine:.4f}  {'PASS ✓' if passed else 'FAIL ✗'}")

    return results


# ---------------------------------------------------------------------------
# 출력 테이블
# ---------------------------------------------------------------------------

def print_table(results: List[KeywordConsistencyResult]) -> None:
    print("\n" + "=" * 62)
    print("=== Experiment 1: Gemini 생성 일관성 (run별 독립 context) ===")
    print("=" * 62)
    print(f"{'Keyword':<12}  {'Word Overlap':>12}  {'Avg Cosine':>10}  {'Min Cosine':>10}  {'Result':>6}")
    print(f"{'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*6}")
    for r in results:
        print(
            f"{r.keyword:<12}  {r.avg_word_overlap:>12.4f}  "
            f"{r.avg_cosine_similarity:>10.4f}  {r.min_cosine_similarity:>10.4f}  "
            f"{'PASS' if r.passed else 'FAIL':>6}"
        )
    print("=" * 62)
    all_passed = all(r.passed for r in results)
    print(f"최종 결과: {'전체 PASS ✓' if all_passed else '일부 FAIL ✗'}\n")

    # run별 context 상세 출력
    print("=== Run별 Context 상세 ===")
    for r in results:
        print(f"\n[{r.keyword}]")
        for rec in r.runs:
            print(f"  run {rec.run_index + 1}")
            print(f"    context: {rec.context[:100]}...")
            print(f"    text:    {rec.text[:100]}...")


# ---------------------------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------------------------

def save_results(results: List[KeywordConsistencyResult]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_iso  = datetime.now().isoformat()

    # JSON: run별 context/text 전체 포함
    json_path = os.path.join(RESULTS_DIR, f"exp1_{ts_file}.json")
    payload = {
        "experiment": "exp1_consistency",
        "timestamp": ts_iso,
        "model_embedding": EMBEDDING_MODEL,
        "model_gemini": GEMINI_MODEL,
        "num_runs": NUM_RUNS,
        "cosine_threshold": COSINE_THRESHOLD,
        "prompt_template": PROMPT_TEMPLATE,
        "pipeline": [
            "fetch_context per run (google_search, 독립 호출)",
            "call_gemini (context 주입)",
            "qwen embedding (mean pooling)",
            "cosine similarity",
        ],
        "results": [
            {
                "keyword": r.keyword,
                "avg_word_overlap": r.avg_word_overlap,
                "avg_cosine_similarity": r.avg_cosine_similarity,
                "min_cosine_similarity": r.min_cosine_similarity,
                "passed": r.passed,
                "runs": [asdict(rec) for rec in r.runs],
            }
            for r in results
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {json_path}")

    # CSV: 행 = (keyword, run_index, context, text) + 집계값
    csv_path = os.path.join(RESULTS_DIR, f"exp1_{ts_file}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "keyword", "run_index",
            "context", "text",
            "avg_word_overlap", "avg_cosine_similarity",
            "min_cosine_similarity", "passed",
        ])
        for r in results:
            for rec in r.runs:
                writer.writerow([
                    ts_iso, r.keyword, rec.run_index,
                    rec.context, rec.text,
                    f"{r.avg_word_overlap:.4f}",
                    f"{r.avg_cosine_similarity:.4f}",
                    f"{r.min_cosine_similarity:.4f}",
                    r.passed,
                ])
    print(f"결과 저장: {csv_path}")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_experiment()
    print_table(results)
    save_results(results)

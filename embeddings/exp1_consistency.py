"""
실험 1: Gemini 생성 일관성 테스트 (Generation Consistency)

[Pipeline]
1. Context Fetching: Gemini google_search 툴로 키워드별 context 1회 확보 (고정)
2. Semantic Enrichment: 고정된 context를 바탕으로 Gemini가 영문 프로필 5회 생성
3. Local Embedding: Qwen-0.6B로 각 텍스트 임베딩
4. Similarity Analysis: 5개 텍스트 간 단어 overlap + 코사인 유사도 측정

통과 기준: 평균 코사인 유사도 >= 0.98

실행 방법:
    export GEMINI_ENDPOINT="https://your-endpoint/v1"
    export GEMINI_BEARER_TOKEN="your-bearer-token"
    python embeddings/exp1_consistency.py
"""

import csv
import dataclasses
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import List

# embeddings/ 디렉토리를 sys.path에 추가 (utils 임포트용)
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

# context를 주입받아 순수 변환만 수행하는 프롬프트
# fetch_context()가 context를 고정하므로 재현성 확보
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
class KeywordConsistencyResult:
    """키워드별 5회 실행 집계 결과"""
    keyword: str
    context: str                  # fetch_context()로 확보한 고정 context
    texts: List[str]              # 5개 생성 텍스트
    avg_word_overlap: float       # Jaccard 유사도 평균 (C(5,2)=10 쌍)
    avg_cosine_similarity: float  # 코사인 유사도 평균
    min_cosine_similarity: float  # 코사인 유사도 최솟값
    passed: bool                  # avg_cosine_similarity >= COSINE_THRESHOLD


# ---------------------------------------------------------------------------
# 유틸 함수
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """소문자 변환 + 구두점 제거"""
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
        print(f"[{keyword}]")
        print(f"{'='*60}")

        # Step 1: context 1회 확보 (고정)
        context = fetch_context(keyword)

        # Step 2: 고정 context 기반으로 5회 생성
        prompt = PROMPT_TEMPLATE.format(keyword=keyword, context=context)
        texts: List[str] = []
        print(f"  영문 프로필 {NUM_RUNS}회 생성 중...")
        for i in range(NUM_RUNS):
            text = call_gemini(prompt)
            texts.append(text)
            print(f"  run {i + 1}/{NUM_RUNS}: {text[:70]}...")

        # Step 3: 임베딩 계산 (mean pooling)
        print(f"  임베딩 계산 중...")
        embeddings = [get_embedding(t, pooling="mean") for t in texts]

        # Step 4: C(5,2)=10 쌍 유사도 계산
        pairs = list(combinations(range(NUM_RUNS), 2))
        overlap_scores = [word_overlap_rate(texts[i], texts[j]) for i, j in pairs]
        cosine_scores = [get_cosine_similarity(embeddings[i], embeddings[j]) for i, j in pairs]

        avg_overlap = sum(overlap_scores) / len(overlap_scores)
        avg_cosine = sum(cosine_scores) / len(cosine_scores)
        min_cosine = min(cosine_scores)
        passed = avg_cosine >= COSINE_THRESHOLD

        results.append(KeywordConsistencyResult(
            keyword=keyword,
            context=context,
            texts=texts,
            avg_word_overlap=avg_overlap,
            avg_cosine_similarity=avg_cosine,
            min_cosine_similarity=min_cosine,
            passed=passed,
        ))
        print(f"  → avg_cosine={avg_cosine:.4f}  {'PASS ✓' if passed else 'FAIL ✗'}")

    return results


# ---------------------------------------------------------------------------
# 출력 테이블
# ---------------------------------------------------------------------------

def print_table(results: List[KeywordConsistencyResult]) -> None:
    print("\n" + "=" * 62)
    print("=== Experiment 1: Gemini 생성 일관성 ===")
    print("=" * 62)
    header = f"{'Keyword':<12}  {'Word Overlap':>12}  {'Avg Cosine':>10}  {'Min Cosine':>10}  {'Result':>6}"
    print(header)
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


# ---------------------------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------------------------

def save_results(results: List[KeywordConsistencyResult]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_iso = datetime.now().isoformat()

    # JSON 저장
    json_path = os.path.join(RESULTS_DIR, f"exp1_{ts_file}.json")
    payload = {
        "experiment": "exp1_consistency",
        "timestamp": ts_iso,
        "model_embedding": EMBEDDING_MODEL,
        "model_gemini": GEMINI_MODEL,
        "num_runs": NUM_RUNS,
        "cosine_threshold": COSINE_THRESHOLD,
        "prompt_template": PROMPT_TEMPLATE,
        "pipeline": ["fetch_context (google_search)", "call_gemini x5 (fixed context)", "qwen embedding", "cosine similarity"],
        "results": [dataclasses.asdict(r) for r in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {json_path}")

    # CSV 저장
    csv_path = os.path.join(RESULTS_DIR, f"exp1_{ts_file}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "keyword",
            "avg_word_overlap", "avg_cosine_similarity", "min_cosine_similarity",
            "passed", "context",
            *[f"text_{i}" for i in range(NUM_RUNS)],
        ])
        for r in results:
            writer.writerow([
                ts_iso, r.keyword,
                f"{r.avg_word_overlap:.4f}",
                f"{r.avg_cosine_similarity:.4f}",
                f"{r.min_cosine_similarity:.4f}",
                r.passed, r.context,
                *r.texts,
            ])
    print(f"결과 저장: {csv_path}")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_experiment()
    print_table(results)
    save_results(results)

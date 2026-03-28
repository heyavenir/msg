"""
실험 1: Gemini 생성 일관성 테스트 (Generation Consistency)

동일 키워드에 대해 Gemini가 매번 일관된 텍스트를 생성하는지 검증.
- 키워드 3개 × 5회 생성
- 메트릭: 단어 Jaccard 유사도, Qwen 임베딩 코사인 유사도
- 통과 기준: 평균 코사인 유사도 >= 0.98

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
from dataclasses import dataclass, field
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
    get_cosine_similarity,
    get_embedding,
)

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

KEYWORDS: List[str] = ["도마뱀", "말차", "토트넘"]
NUM_RUNS: int = 5
COSINE_THRESHOLD: float = 0.98

# Gemini에 보낼 고정 프롬프트 (temperature=0이므로 동일 결과 기대)
PROMPT_TEMPLATE: str = (
    "Provide a concise description of '{keyword}' in 2-3 English sentences. "
    "Do not add any preamble."
)


# ---------------------------------------------------------------------------
# 데이터클래스
# ---------------------------------------------------------------------------

@dataclass
class KeywordConsistencyResult:
    """키워드별 5회 실행 집계 결과"""
    keyword: str
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
        print(f"\n[{keyword}] 생성 시작 ({NUM_RUNS}회)")
        prompt = PROMPT_TEMPLATE.format(keyword=keyword)
        texts: List[str] = []

        for i in range(NUM_RUNS):
            text = call_gemini(prompt)
            texts.append(text)
            print(f"  run {i + 1}/{NUM_RUNS}: {text[:60]}...")

        # 임베딩 계산 (mean pooling)
        print(f"  [{keyword}] 임베딩 계산 중...")
        embeddings = [get_embedding(t, pooling="mean") for t in texts]

        # C(5,2)=10 쌍 계산
        pairs = list(combinations(range(NUM_RUNS), 2))
        overlap_scores = [word_overlap_rate(texts[i], texts[j]) for i, j in pairs]
        cosine_scores = [get_cosine_similarity(embeddings[i], embeddings[j]) for i, j in pairs]

        avg_overlap = sum(overlap_scores) / len(overlap_scores)
        avg_cosine = sum(cosine_scores) / len(cosine_scores)
        min_cosine = min(cosine_scores)
        passed = avg_cosine >= COSINE_THRESHOLD

        result = KeywordConsistencyResult(
            keyword=keyword,
            texts=texts,
            avg_word_overlap=avg_overlap,
            avg_cosine_similarity=avg_cosine,
            min_cosine_similarity=min_cosine,
            passed=passed,
        )
        results.append(result)
        print(f"  [{keyword}] avg_cosine={avg_cosine:.4f}, passed={'PASS' if passed else 'FAIL'}")

    return results


# ---------------------------------------------------------------------------
# 출력 테이블
# ---------------------------------------------------------------------------

def print_table(results: List[KeywordConsistencyResult]) -> None:
    print("\n" + "=" * 62)
    print("=== Experiment 1: Gemini 생성 일관성 ===")
    print("=" * 62)
    header = f"{'Keyword':<12}  {'Word Overlap':>12}  {'Avg Cosine':>10}  {'Min Cosine':>10}  {'Result':>6}"
    sep = f"{'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*6}"
    print(header)
    print(sep)
    for r in results:
        result_str = "PASS" if r.passed else "FAIL"
        print(
            f"{r.keyword:<12}  {r.avg_word_overlap:>12.4f}  "
            f"{r.avg_cosine_similarity:>10.4f}  {r.min_cosine_similarity:>10.4f}  "
            f"{result_str:>6}"
        )
    print("=" * 62)
    all_passed = all(r.passed for r in results)
    print(f"최종 결과: {'전체 PASS' if all_passed else '일부 FAIL — 프롬프트 수정 권장'}")
    if not all_passed:
        print("  팁: 프롬프트에 'Do not add any preamble.' 옵션을 추가해보세요.")
    print()


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
            "passed",
            *[f"text_{i}" for i in range(NUM_RUNS)],
        ])
        for r in results:
            writer.writerow([
                ts_iso,
                r.keyword,
                f"{r.avg_word_overlap:.4f}",
                f"{r.avg_cosine_similarity:.4f}",
                f"{r.min_cosine_similarity:.4f}",
                r.passed,
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

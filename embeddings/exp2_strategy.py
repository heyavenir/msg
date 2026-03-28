"""
실험 2: 전략 조합 비교 (Strategy Combination Test)

[Pipeline]
1. Context Fetching: Gemini google_search 툴로 키워드별 context 1회 확보 (고정)
2. Semantic Enrichment: 고정 context를 바탕으로 전략별 영문 프로필 생성
3. Local Embedding: Qwen-0.6B로 임베딩 (전략별 풀링 방식 적용)
4. Similarity Analysis: 코사인 유사도 측정 → Margin 순위 리포트

Margin = Avg(Synonym_Sim) - Avg(Distinct_Sim) → 클수록 좋음

실행 방법:
    export GEMINI_ENDPOINT="https://your-endpoint/v1"
    export GEMINI_BEARER_TOKEN="your-bearer-token"
    python embeddings/exp2_strategy.py
"""

import csv
import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Tuple

# embeddings/ 디렉토리를 sys.path에 추가 (utils 임포트용)
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    RESULTS_DIR,
    PoolingMode,
    call_gemini,
    fetch_context,
    get_cosine_similarity,
    get_embedding,
)

# ---------------------------------------------------------------------------
# 타입
# ---------------------------------------------------------------------------

PairType = Literal["synonym", "distinct", "irrelevant"]


# ---------------------------------------------------------------------------
# 전략 정의
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """전략 정의 (프롬프트 템플릿 + 풀링 방식)"""
    name: str              # "A", "B", "C", "D"
    prompt_template: str   # [Interest], [Context] placeholder 포함
    pooling: PoolingMode
    description: str


# context를 주입받는 형태로 재설계
# [Context]: fetch_context()로 확보한 고정 텍스트
# [Interest]: 키워드
STRATEGIES: List[Strategy] = [
    Strategy(
        name="A",
        prompt_template=(
            "Context about '[Interest]':\n[Context]\n\n"
            "Provide a concise dictionary-style definition for '[Interest]' "
            "in 2-3 English sentences. Standard facts only."
        ),
        pooling="last_token",
        description="Control: 사전적 정의, Last Token Pooling",
    ),
    Strategy(
        name="B",
        prompt_template=(
            "Context about '[Interest]':\n[Context]\n\n"
            "Create a specific English profile for '[Interest]'. "
            "Emphasize unique identifiers (location, specific ingredients, or technical origins) "
            "to distinguish it from similar items. Use unique domain terms."
        ),
        pooling="mean",
        description="Pooling Focus: 고유 식별자 강조, Mean Pooling",
    ),
    Strategy(
        name="C",
        prompt_template=(
            "Context about '[Interest]':\n[Context]\n\n"
            "Generate an English expansion for '[Interest]'. "
            "List and repeat all synonyms and related names "
            "(e.g., Dog, Puppy, Canine) at the start and throughout the text "
            "to maximize keyword overlap."
        ),
        pooling="last_token",
        description="Synonym Focus: 동의어 반복, Last Token Pooling",
    ),
    Strategy(
        name="D",
        prompt_template=(
            "Context about '[Interest]':\n[Context]\n\n"
            "Create a structured English profile for '[Interest]'. "
            "Section 1: List all synonyms. "
            "Section 2: Describe unique identity and specific discriminators "
            "to ensure both clustering and distinction."
        ),
        pooling="mean",
        description="Hybrid: 동의어 + 고유 구분자, Mean Pooling",
    ),
]


# ---------------------------------------------------------------------------
# 테스트 쌍 정의
# ---------------------------------------------------------------------------

@dataclass
class TestPair:
    keyword_a: str
    keyword_b: str
    pair_type: PairType


TEST_PAIRS: List[TestPair] = [
    # 동의어 쌍 — 높은 유사도 기대
    TestPair("개",    "강아지",  "synonym"),
    TestPair("고양이", "야옹이",  "synonym"),
    TestPair("말차",  "마차",    "synonym"),   # 주의: 말차(matcha)↔마차(carriage), 의미적 차이 있는 실험용 쌍
    # 구분 쌍 — 낮은 유사도 기대
    TestPair("토트넘", "리버풀", "distinct"),
    TestPair("말차",  "우롱차",  "distinct"),
    # 무관 쌍 — 매우 낮은 유사도 기대
    TestPair("말차",  "토트넘",  "irrelevant"),
]


# ---------------------------------------------------------------------------
# 데이터클래스
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    """키워드 쌍 하나에 대한 실험 결과"""
    strategy_name: str
    keyword_a: str
    keyword_b: str
    pair_type: PairType
    context_a: str
    context_b: str
    text_a: str
    text_b: str
    similarity: float


@dataclass
class StrategyAnalysis:
    """전략별 집계 분석"""
    strategy_name: str
    description: str
    avg_synonym_sim: float
    avg_distinct_sim: float
    avg_irrelevant_sim: float
    margin: float                             # avg_synonym_sim - avg_distinct_sim
    pair_results: List[PairResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 실험 실행
# ---------------------------------------------------------------------------

def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_prompt(template: str, keyword: str, context: str) -> str:
    return template.replace("[Interest]", keyword).replace("[Context]", context)


def run_experiment() -> Tuple[List[PairResult], List[StrategyAnalysis]]:
    # Step 1: 모든 키워드의 context를 미리 확보 (키워드당 1회, 전략 간 재사용)
    all_keywords = list({kw for pair in TEST_PAIRS for kw in (pair.keyword_a, pair.keyword_b)})
    print("=" * 60)
    print("Step 1: 키워드별 Context 확보 (google_search)")
    print("=" * 60)
    context_cache: Dict[str, str] = {}
    for keyword in all_keywords:
        context_cache[keyword] = fetch_context(keyword)

    # Step 2: 전략별 실험
    all_pair_results: List[PairResult] = []

    for strategy in STRATEGIES:
        print(f"\n{'='*60}")
        print(f"Strategy {strategy.name}: {strategy.description}")
        print(f"{'='*60}")

        for pair in TEST_PAIRS:
            context_a = context_cache[pair.keyword_a]
            context_b = context_cache[pair.keyword_b]

            prompt_a = _build_prompt(strategy.prompt_template, pair.keyword_a, context_a)
            prompt_b = _build_prompt(strategy.prompt_template, pair.keyword_b, context_b)

            text_a = call_gemini(prompt_a)
            text_b = call_gemini(prompt_b)

            emb_a = get_embedding(text_a, pooling=strategy.pooling)
            emb_b = get_embedding(text_b, pooling=strategy.pooling)
            sim = get_cosine_similarity(emb_a, emb_b)

            all_pair_results.append(PairResult(
                strategy_name=strategy.name,
                keyword_a=pair.keyword_a,
                keyword_b=pair.keyword_b,
                pair_type=pair.pair_type,
                context_a=context_a,
                context_b=context_b,
                text_a=text_a,
                text_b=text_b,
                similarity=sim,
            ))
            print(f"  ({pair.keyword_a}, {pair.keyword_b}) [{pair.pair_type:>10}]: {sim:.4f}")

    analyses = _compute_analysis(all_pair_results)
    return all_pair_results, analyses


def _compute_analysis(pair_results: List[PairResult]) -> List[StrategyAnalysis]:
    from collections import defaultdict

    grouped: dict = defaultdict(lambda: defaultdict(list))
    for r in pair_results:
        grouped[r.strategy_name][r.pair_type].append(r.similarity)

    strategy_map = {s.name: s for s in STRATEGIES}
    analyses: List[StrategyAnalysis] = []
    for name, strategy in strategy_map.items():
        sims = grouped[name]
        avg_syn  = _safe_mean(sims.get("synonym", []))
        avg_dist = _safe_mean(sims.get("distinct", []))
        avg_irr  = _safe_mean(sims.get("irrelevant", []))
        analyses.append(StrategyAnalysis(
            strategy_name=name,
            description=strategy.description,
            avg_synonym_sim=avg_syn,
            avg_distinct_sim=avg_dist,
            avg_irrelevant_sim=avg_irr,
            margin=avg_syn - avg_dist,
            pair_results=[r for r in pair_results if r.strategy_name == name],
        ))

    analyses.sort(key=lambda a: a.margin, reverse=True)
    return analyses


# ---------------------------------------------------------------------------
# 출력 테이블
# ---------------------------------------------------------------------------

def print_tables(
    pair_results: List[PairResult],
    analyses: List[StrategyAnalysis],
) -> None:
    # 테이블 1: 전략×쌍별 raw 유사도
    print("\n" + "=" * 70)
    print("=== Experiment 2: 전략별 쌍 유사도 ===")
    print("=" * 70)
    print(f"{'Strategy':<10}  {'Pair':<18}  {'Type':<11}  {'Similarity':>10}")
    print(f"{'-'*10}  {'-'*18}  {'-'*11}  {'-'*10}")
    for r in pair_results:
        pair_label = f"{r.keyword_a} / {r.keyword_b}"
        print(f"{r.strategy_name:<10}  {pair_label:<18}  {r.pair_type:<11}  {r.similarity:>10.4f}")

    # 테이블 2: margin 순위
    print("\n" + "=" * 75)
    print("=== 전략 분석 (Margin = Avg_Synonym - Avg_Distinct) ===")
    print("=" * 75)
    print(
        f"{'Rank':<5}  {'Strategy':<10}  "
        f"{'Avg_Synonym':>11}  {'Avg_Distinct':>12}  "
        f"{'Avg_Irrelevant':>14}  {'Margin':>8}"
    )
    print(f"{'-'*5}  {'-'*10}  {'-'*11}  {'-'*12}  {'-'*14}  {'-'*8}")
    for rank, a in enumerate(analyses, start=1):
        best = "  ← Best" if rank == 1 else ""
        print(
            f"{rank:<5}  {a.strategy_name:<10}  "
            f"{a.avg_synonym_sim:>11.4f}  {a.avg_distinct_sim:>12.4f}  "
            f"{a.avg_irrelevant_sim:>14.4f}  {a.margin:>8.4f}{best}"
        )
    print("=" * 75)
    best = analyses[0]
    print(f"\n최적 전략: Strategy {best.strategy_name} ({best.description})")
    print(f"  Margin = {best.margin:.4f}  (Synonym={best.avg_synonym_sim:.4f}, Distinct={best.avg_distinct_sim:.4f})\n")


# ---------------------------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------------------------

def save_results(
    pair_results: List[PairResult],
    analyses: List[StrategyAnalysis],
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_iso = datetime.now().isoformat()

    analyses_dicts = [
        {
            "strategy_name": a.strategy_name,
            "description": a.description,
            "avg_synonym_sim": a.avg_synonym_sim,
            "avg_distinct_sim": a.avg_distinct_sim,
            "avg_irrelevant_sim": a.avg_irrelevant_sim,
            "margin": a.margin,
        }
        for a in analyses
    ]

    # JSON 저장
    json_path = os.path.join(RESULTS_DIR, f"exp2_{ts_file}.json")
    payload = {
        "experiment": "exp2_strategy",
        "timestamp": ts_iso,
        "model_embedding": EMBEDDING_MODEL,
        "model_gemini": GEMINI_MODEL,
        "pipeline": ["fetch_context (google_search, per keyword)", "call_gemini (fixed context)", "qwen embedding", "cosine similarity"],
        "strategies": [
            {"name": s.name, "pooling": s.pooling, "description": s.description}
            for s in STRATEGIES
        ],
        "analysis": analyses_dicts,
        "pair_results": [dataclasses.asdict(r) for r in pair_results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {json_path}")

    # CSV 저장
    csv_path = os.path.join(RESULTS_DIR, f"exp2_{ts_file}.csv")
    strategy_desc = {s.name: s.description for s in STRATEGIES}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "strategy_name", "strategy_description",
            "keyword_a", "keyword_b", "pair_type",
            "similarity", "context_a", "context_b", "text_a", "text_b",
        ])
        for r in pair_results:
            writer.writerow([
                ts_iso,
                r.strategy_name,
                strategy_desc.get(r.strategy_name, ""),
                r.keyword_a, r.keyword_b, r.pair_type,
                f"{r.similarity:.4f}",
                r.context_a, r.context_b, r.text_a, r.text_b,
            ])
    print(f"결과 저장: {csv_path}")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pair_results, analyses = run_experiment()
    print_tables(pair_results, analyses)
    save_results(pair_results, analyses)

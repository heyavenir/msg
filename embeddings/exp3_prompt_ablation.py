"""
실험 3: 프롬프트 변형 비교 (Prompt Ablation Study)

[목적]
Strategy A (사전적 정의, Last Token Pooling) 기반으로 프롬프트 변형을 통해
distinct/irrelevant 쌍의 유사도를 낮추는 최적 프롬프트 찾기.

현재 문제: distinct와 irrelevant 쌍 모두 유사도가 높게 나옴
→ Gemini 출력에 포함된 generic 카테고리 단어들이 모든 아이템을 클러스터링시킴

[7가지 변형]
- A-original:          기존 Strategy A (baseline)
- A-unique-first:      카테고리 문맥 제거, 고유 특성 우선 기술
- A-category-last:     컨텍스트/카테고리를 프롬프트 끝으로 이동
- A-negative:          generic 카테고리 용어 사용 금지 지시어 추가
- A-unique-identifiers: 고유명사/위치/특정 식별자만 추출
- A-combined:          A-category-last + A-negative 결합
- A-tag-extraction:    산문 없이 고유 키워드 태그만 쉼표 구분 출력

[3가지 지표]
- Margin          = Avg(synonym_sim) - Avg(distinct_sim)       ← 클수록 좋음
- Gap             = Avg(distinct_sim) - Avg(irrelevant_sim)    ← 클수록 좋음
- Total Separation = Avg(synonym_sim) - Avg(irrelevant_sim)    ← 클수록 좋음

[Pipeline]
1. Context Fetching: 키워드별 google_search context 1회 확보 (변형 간 재사용)
2. Semantic Enrichment: 각 변형 프롬프트로 영문 프로필 생성
3. Local Embedding: Qwen-0.6B Last Token Pooling
4. Similarity Analysis: 코사인 유사도 측정 → 3지표 순위 리포트

실행 방법:
    export GEMINI_ENDPOINT="https://your-endpoint/v1"
    export GEMINI_BEARER_TOKEN="your-bearer-token"
    python embeddings/exp3_prompt_ablation.py
"""

import csv
import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

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
from dataset import INTERESTS, InterestItem, PairType, build_test_pairs


# ---------------------------------------------------------------------------
# 프롬프트 변형 정의
# ---------------------------------------------------------------------------

@dataclass
class PromptVariant:
    """프롬프트 변형 정의 — 모두 Last Token Pooling 사용"""
    name: str
    prompt_template: str   # [Interest], [Context] placeholder 포함
    description: str


VARIANTS: List[PromptVariant] = [
    PromptVariant(
        name="A-original",
        prompt_template=(
            "Context about '[Interest]':\n[Context]\n\n"
            "Provide a concise dictionary-style definition for '[Interest]' "
            "in 2-3 English sentences. Standard facts only."
        ),
        description="Baseline: 컨텍스트 선행, 사전적 정의",
    ),
    PromptVariant(
        name="A-unique-first",
        prompt_template=(
            "Describe '[Interest]' focusing exclusively on its unique, "
            "distinguishing characteristics in 2-3 English sentences. "
            "Emphasize what makes it different from similar items — "
            "specific origin, method, ingredient, or technical property. "
            "Avoid generic category names."
        ),
        description="카테고리 문맥 제거, 고유 특성 우선 기술",
    ),
    PromptVariant(
        name="A-category-last",
        prompt_template=(
            "Provide a concise dictionary-style definition for '[Interest]' "
            "in 2-3 English sentences. Standard facts only.\n\n"
            "Additional context:\n[Context]"
        ),
        description="컨텍스트/카테고리를 프롬프트 끝으로 이동",
    ),
    PromptVariant(
        name="A-negative",
        prompt_template=(
            "Context about '[Interest]':\n[Context]\n\n"
            "Provide a concise dictionary-style definition for '[Interest]' "
            "in 2-3 English sentences. Standard facts only. "
            "Do NOT use generic category names or broad domain terms. "
            "Focus on specific, unique attributes only."
        ),
        description="Generic 카테고리 용어 사용 금지 지시어 추가",
    ),
    PromptVariant(
        name="A-unique-identifiers",
        prompt_template=(
            "Context about '[Interest]':\n[Context]\n\n"
            "Extract only the most specific identifiers for '[Interest]': "
            "proper names, geographic origins, specific technical terms, "
            "unique ingredients, or precise model/version numbers. "
            "Write 2-3 sentences using only these precise identifiers. "
            "No generic terms."
        ),
        description="고유명사/위치/특정 식별자만 추출",
    ),
    PromptVariant(
        name="A-combined",
        prompt_template=(
            "Provide a concise dictionary-style definition for '[Interest]' "
            "in 2-3 English sentences. Standard facts only. "
            "Do NOT use generic category names or broad domain terms. "
            "Focus on specific, unique attributes only.\n\n"
            "Additional context:\n[Context]"
        ),
        description="A-category-last + A-negative 결합",
    ),
    PromptVariant(
        name="A-tag-extraction",
        prompt_template=(
            "Context about '[Interest]':\n[Context]\n\n"
            "Extract the most unique and specific keyword tags for '[Interest]'. "
            "Output ONLY a comma-separated list of specific tags "
            "(proper nouns, locations, technical terms, unique identifiers). "
            "No sentences, no generic words, no explanation. "
            "Example format: tag1, tag2, tag3, tag4, tag5"
        ),
        description="산문 없이 고유 키워드 태그만 쉼표 구분 출력",
    ),
]

POOLING_MODE = "last_token"  # 모든 변형에 동일 적용


# ---------------------------------------------------------------------------
# 테스트 쌍 (exp2와 동일 구조)
# ---------------------------------------------------------------------------

@dataclass
class TestPair:
    item_a: InterestItem
    item_b: InterestItem
    pair_type: PairType


USE_DATASET: bool = True

_MANUAL_PAIRS: List[TestPair] = [
    TestPair(InterestItem("Pets",  "개"),    InterestItem("Pets",  "강아지"),  "synonym"),
    TestPair(InterestItem("Pets",  "고양이"), InterestItem("Pets",  "야옹이"),  "synonym"),
    TestPair(InterestItem("Food",  "말차"),   InterestItem("Food",  "마차"),    "synonym"),
    TestPair(InterestItem("Sport", "토트넘"), InterestItem("Sport", "리버풀"), "distinct"),
    TestPair(InterestItem("Food",  "말차"),   InterestItem("Food",  "우롱차"),  "distinct"),
    TestPair(InterestItem("Food",  "말차"),   InterestItem("Sport", "토트넘"), "irrelevant"),
]

def _build_dataset_pairs() -> List[TestPair]:
    raw = build_test_pairs(INTERESTS, n_intra=5, n_inter=5, n_irrelevant=3)
    return [TestPair(a, b, t) for a, b, t in raw]

TEST_PAIRS: List[TestPair] = _build_dataset_pairs() if USE_DATASET else _MANUAL_PAIRS


# ---------------------------------------------------------------------------
# 데이터클래스
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    """키워드 쌍 하나에 대한 실험 결과"""
    variant_name: str
    keyword_a: str
    keyword_b: str
    pair_type: PairType
    context_a: str
    context_b: str
    prompt_a: str
    prompt_b: str
    text_a: str
    text_b: str
    similarity: float


@dataclass
class VariantAnalysis:
    """변형별 집계 분석"""
    variant_name: str
    description: str
    avg_synonym_sim: float
    avg_distinct_sim: float
    avg_irrelevant_sim: float
    margin: float            # synonym - distinct
    gap: float               # distinct - irrelevant
    total_separation: float  # synonym - irrelevant
    pair_results: List[PairResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 실험 실행
# ---------------------------------------------------------------------------

def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_prompt(template: str, keyword: str, context: str) -> str:
    return template.replace("[Interest]", keyword).replace("[Context]", context)


def run_experiment() -> Tuple[List[PairResult], List[VariantAnalysis]]:
    # Step 1: 모든 아이템의 context를 미리 확보 (아이템당 1회, 변형 간 재사용)
    all_items = list({
        item.search_query: item
        for pair in TEST_PAIRS
        for item in (pair.item_a, pair.item_b)
    }.values())

    print("=" * 60)
    print("Step 1: 키워드별 Context 확보 (google_search)")
    print("=" * 60)
    context_cache: Dict[str, str] = {}
    for item in all_items:
        context_cache[item.search_query] = fetch_context(item.search_query)

    # Step 2: 변형별 실험
    all_pair_results: List[PairResult] = []

    for variant in VARIANTS:
        print(f"\n{'='*60}")
        print(f"Variant {variant.name}: {variant.description}")
        print(f"{'='*60}")

        for pair in TEST_PAIRS:
            context_a = context_cache[pair.item_a.search_query]
            context_b = context_cache[pair.item_b.search_query]

            prompt_a = _build_prompt(variant.prompt_template, pair.item_a.item, context_a)
            prompt_b = _build_prompt(variant.prompt_template, pair.item_b.item, context_b)

            text_a = call_gemini(prompt_a)
            text_b = call_gemini(prompt_b)

            emb_a = get_embedding(text_a, pooling=POOLING_MODE)
            emb_b = get_embedding(text_b, pooling=POOLING_MODE)
            sim = get_cosine_similarity(emb_a, emb_b)

            all_pair_results.append(PairResult(
                variant_name=variant.name,
                keyword_a=pair.item_a.item,
                keyword_b=pair.item_b.item,
                pair_type=pair.pair_type,
                context_a=context_a,
                context_b=context_b,
                prompt_a=prompt_a,
                prompt_b=prompt_b,
                text_a=text_a,
                text_b=text_b,
                similarity=sim,
            ))
            print(f"  ({pair.item_a.item}, {pair.item_b.item}) [{pair.pair_type:>10}]: {sim:.4f}")

    analyses = _compute_analysis(all_pair_results)
    return all_pair_results, analyses


def _compute_analysis(pair_results: List[PairResult]) -> List[VariantAnalysis]:
    from collections import defaultdict

    grouped: dict = defaultdict(lambda: defaultdict(list))
    for r in pair_results:
        grouped[r.variant_name][r.pair_type].append(r.similarity)

    variant_map = {v.name: v for v in VARIANTS}
    analyses: List[VariantAnalysis] = []
    for name, variant in variant_map.items():
        sims = grouped[name]
        avg_syn  = _safe_mean(sims.get("synonym", []))
        avg_dist = _safe_mean(sims.get("distinct", []))
        avg_irr  = _safe_mean(sims.get("irrelevant", []))
        analyses.append(VariantAnalysis(
            variant_name=name,
            description=variant.description,
            avg_synonym_sim=avg_syn,
            avg_distinct_sim=avg_dist,
            avg_irrelevant_sim=avg_irr,
            margin=avg_syn - avg_dist,
            gap=avg_dist - avg_irr,
            total_separation=avg_syn - avg_irr,
            pair_results=[r for r in pair_results if r.variant_name == name],
        ))

    # Margin 기준 정렬 (1차), Total Separation 기준 정렬 (2차)
    analyses.sort(key=lambda a: (a.margin, a.total_separation), reverse=True)
    return analyses


# ---------------------------------------------------------------------------
# 출력 테이블
# ---------------------------------------------------------------------------

def print_tables(
    pair_results: List[PairResult],
    analyses: List[VariantAnalysis],
) -> None:
    # 테이블 1: 변형×쌍별 raw 유사도
    print("\n" + "=" * 75)
    print("=== Experiment 3: 변형별 쌍 유사도 ===")
    print("=" * 75)
    print(f"{'Variant':<22}  {'Pair':<30}  {'Type':<11}  {'Similarity':>10}")
    print(f"{'-'*22}  {'-'*30}  {'-'*11}  {'-'*10}")
    for r in pair_results:
        pair_label = f"{r.keyword_a} / {r.keyword_b}"
        print(f"{r.variant_name:<22}  {pair_label:<30}  {r.pair_type:<11}  {r.similarity:>10.4f}")

    # 테이블 2: 3지표 순위
    print("\n" + "=" * 95)
    print("=== 변형 분석 (Margin, Gap, Total Separation) ===")
    print("=" * 95)
    print(
        f"{'Rank':<5}  {'Variant':<22}  "
        f"{'Avg_Syn':>7}  {'Avg_Dist':>8}  {'Avg_Irr':>7}  "
        f"{'Margin':>8}  {'Gap':>8}  {'TotalSep':>9}"
    )
    print(f"{'-'*5}  {'-'*22}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*9}")
    for rank, a in enumerate(analyses, start=1):
        best = "  ← Best" if rank == 1 else ""
        print(
            f"{rank:<5}  {a.variant_name:<22}  "
            f"{a.avg_synonym_sim:>7.4f}  {a.avg_distinct_sim:>8.4f}  {a.avg_irrelevant_sim:>7.4f}  "
            f"{a.margin:>8.4f}  {a.gap:>8.4f}  {a.total_separation:>9.4f}{best}"
        )
    print("=" * 95)

    best = analyses[0]
    print(f"\n최적 변형: {best.variant_name} ({best.description})")
    print(f"  Margin={best.margin:.4f}  Gap={best.gap:.4f}  TotalSep={best.total_separation:.4f}")
    print(f"  (Synonym={best.avg_synonym_sim:.4f}, Distinct={best.avg_distinct_sim:.4f}, Irrelevant={best.avg_irrelevant_sim:.4f})\n")

    # 지표별 Top-1 요약
    print("=== 지표별 Top-1 ===")
    best_margin = max(analyses, key=lambda a: a.margin)
    best_gap    = max(analyses, key=lambda a: a.gap)
    best_total  = max(analyses, key=lambda a: a.total_separation)
    print(f"  Best Margin:           {best_margin.variant_name:<22} ({best_margin.margin:.4f})")
    print(f"  Best Gap:              {best_gap.variant_name:<22} ({best_gap.gap:.4f})")
    print(f"  Best Total Separation: {best_total.variant_name:<22} ({best_total.total_separation:.4f})\n")


# ---------------------------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------------------------

def save_results(
    pair_results: List[PairResult],
    analyses: List[VariantAnalysis],
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_iso = datetime.now().isoformat()

    analyses_dicts = [
        {
            "variant_name": a.variant_name,
            "description": a.description,
            "avg_synonym_sim": a.avg_synonym_sim,
            "avg_distinct_sim": a.avg_distinct_sim,
            "avg_irrelevant_sim": a.avg_irrelevant_sim,
            "margin": a.margin,
            "gap": a.gap,
            "total_separation": a.total_separation,
        }
        for a in analyses
    ]

    # JSON 저장
    json_path = os.path.join(RESULTS_DIR, f"exp3_{ts_file}.json")
    payload = {
        "experiment": "exp3_prompt_ablation",
        "timestamp": ts_iso,
        "model_embedding": EMBEDDING_MODEL,
        "model_gemini": GEMINI_MODEL,
        "pooling": POOLING_MODE,
        "pipeline": [
            "fetch_context (google_search, per keyword)",
            "call_gemini (variant prompt + fixed context)",
            "qwen embedding (last_token pooling)",
            "cosine similarity",
        ],
        "variants": [
            {"name": v.name, "description": v.description}
            for v in VARIANTS
        ],
        "metrics": {
            "margin": "Avg(synonym_sim) - Avg(distinct_sim)",
            "gap": "Avg(distinct_sim) - Avg(irrelevant_sim)",
            "total_separation": "Avg(synonym_sim) - Avg(irrelevant_sim)",
        },
        "analysis": analyses_dicts,
        "pair_results": [dataclasses.asdict(r) for r in pair_results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {json_path}")

    # CSV 저장
    csv_path = os.path.join(RESULTS_DIR, f"exp3_{ts_file}.csv")
    variant_desc = {v.name: v.description for v in VARIANTS}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "variant_name", "variant_description",
            "keyword_a", "keyword_b", "pair_type",
            "similarity", "context_a", "context_b",
            "prompt_a", "prompt_b", "text_a", "text_b",
        ])
        for r in pair_results:
            writer.writerow([
                ts_iso,
                r.variant_name,
                variant_desc.get(r.variant_name, ""),
                r.keyword_a, r.keyword_b, r.pair_type,
                f"{r.similarity:.4f}",
                r.context_a, r.context_b,
                r.prompt_a, r.prompt_b, r.text_a, r.text_b,
            ])
    print(f"결과 저장: {csv_path}")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pair_results, analyses = run_experiment()
    print_tables(pair_results, analyses)
    save_results(pair_results, analyses)

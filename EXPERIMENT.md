# 실험 설계 문서

## 현재 실험 조합

| 항목 | 설정 |
|------|------|
| **Benchmark** | LoCoMo (locomo10.json, 10 conversations, 1986 QA) |
| **Memory System** | mem0 (editable, 로컬 소스 수정 가능) |
| **LLM (memory 추출)** | qwen:32b via Ollama |
| **LLM (answer 생성)** | qwen:32b via Ollama |
| **Embedder** | Qwen3-Embedding-0.6B (sentence-transformers, local) |
| **Vector Store** | Chroma (embedded, 로컬 파일 `./chroma_db`) |
| **Metric** | F1, Exact Match (카테고리별 분리) |

---

## 전체 실행 흐름

```
locomo10.json
    │
    ▼
[1] LoCoMoBenchmark.load()
    → 10개 Session 파싱
    → 각 Session: turns (대화) + qa_pairs (질문/정답)

    │
    ▼
[2] EvalRunner.run() — session 루프
    │
    ├─ memory.reset(session_id)        # 세션 시작 전 이전 메모리 삭제
    │
    ├─ [turn 루프] memory.add(turn, user_id=session_id)
    │       └─ mem0 내부: qwen:32b로 turn에서 핵심 facts 추출 → Chroma에 저장
    │
    └─ [QA 루프]
            ├─ memory.search(question, user_id=session_id, top_k=5)
            │       └─ Qwen3-Embedding으로 question 임베딩 → Chroma 유사도 검색
            │
            └─ answer_fn(question, retrieved_memories)
                    └─ qwen:32b에 context + question 전달 → 답변 생성

    │
    ▼
[3] benchmark.evaluate(predictions, references)
    → F1, EM 계산 (카테고리별)

    │
    ▼
[4] results/ 저장
    → locomo_mem0_YYYYMMDD_HHMMSS.json (전체)
    → summary.csv (점수 요약, 실험 간 비교용)
```

---

## user_id와 mem0 연결 방식

LoCoMo의 `session_id` (예: `"conv-26"`)를 mem0의 `user_id`로 그대로 사용.

```python
memory.add(content=turn, user_id="conv-26")
memory.search(query=question, user_id="conv-26")
memory.reset(user_id="conv-26")   # 세션 시작 전 초기화
```

mem0는 `user_id`로 메모리를 격리하기 때문에 각 대화(session)가 서로 간섭하지 않음.
세션 순서대로 실행되며, 매 세션 시작 시 `reset()`으로 이전 세션 메모리를 완전히 삭제.

---

## mem0가 turn을 처리하는 방식

일반 RAG와 달리 mem0는 turn 텍스트를 **그대로 저장하지 않음**.

```
turn 입력: "Caroline: I went to a LGBTQ support group yesterday"
    │
    ▼
mem0 내부 (qwen:32b 호출)
    → "Caroline visited a LGBTQ support group" 같은 핵심 fact 추출
    → 기존 메모리와 중복/충돌 여부 확인 후 업데이트 또는 신규 저장
    │
    ▼
Chroma에 fact 벡터로 저장
```

즉, **LLM이 2번** 사용됨:
1. `add()` 시: turn → fact 추출 (qwen:32b)
2. `answer_fn()` 시: retrieved facts + question → 답변 생성 (qwen:32b)

---

## turns 구조 (LoCoMo 기준)

LoCoMo 한 session은 여러 sub-session으로 구성됨 (session_1, session_2, ...):

```
conv-26
├── session_1 (1:56 pm on 8 May, 2023)
│   ├── Turn: "Caroline: Hey Mel! ..."
│   ├── Turn: "Melanie: ..."
│   └── ...
├── session_2 (...)
│   └── ...
└── ... (최대 ~35 sub-sessions)

QA pairs (199개)
├── Q: "When did Caroline go to the LGBTQ support group?" → A: "7 May 2023"  [category: temporal]
├── Q: "..."  [category: single-hop]
└── ...
```

현재 구현에서는 **모든 sub-session의 turns를 순서대로 합쳐서** 메모리에 한꺼번에 저장.
(시간 순서 정보는 `timestamp` 필드로 보존)

---

## LoCoMo QA 카테고리별 평가

| Category | 설명 | 특징 |
|----------|------|------|
| 1 single-hop | 단일 발화에서 답 가능 | 가장 쉬움 |
| 2 temporal | 날짜/시간 추론 필요 | 시간 정보 정확성 중요 |
| 3 multi-hop | 여러 발화 조합 필요 | retrieval 품질이 핵심 |
| 4 open-domain | 외부 지식 필요할 수 있음 | LLM 자체 지식도 활용 |
| 5 adversarial | 정답이 없거나 함정 질문 | `adversarial_answer` 키 사용 |

`evaluate()` 결과 예시:
```json
{
  "f1": 0.42,
  "em": 0.18,
  "f1_single-hop": 0.55,
  "f1_temporal": 0.38,
  "f1_multi-hop": 0.31,
  "f1_open-domain": 0.44,
  "f1_adversarial": 0.29,
  "num_questions": 1986
}
```

---

## 향후 실험 확장 계획

### 메모리 시스템 비교
```
LoCoMo × {mem0, graphiti_dryrun, custom} × qwen:32b
```

### 벤치마크 확장
```
{LoCoMo, LongMemEval, CAME-Bench} × mem0 × qwen:32b
```

### 변수 실험
- `top_k`: 3, 5, 10 변화에 따른 성능 차이
- mem0 내부 memory extraction prompt 수정 후 비교
- Embedder 교체 (Qwen3-Embedding-0.6B vs 다른 모델)

### 결과 비교
모든 실험은 `results/summary.csv`에 누적 저장되어 실험 간 비교 가능.

---

## 프로젝트 구조

```
msg/
├── benchmarks/          # LoCoMo, LongMemEval, CAME-Bench
├── memory/
│   ├── mem0_adapter.py        # mem0 로컬 (libs/mem0)
│   ├── zep_adapter.py         # GraphitiDryRunMemory (graphiti extraction dry-run)
│   └── custom/                # 커스텀 메모리 추가용
├── eval/                # EvalRunner, F1/EM metrics
├── data/locomo/         # locomo10.json
├── results/             # JSON + summary.csv
├── libs/
│   ├── mem0/            # mem0 소스 (editable 설치)
│   │   └── evaluation/  # 공식 LoCoMo eval 코드 (MemoryClient 기반)
│   └── graphiti/        # graphiti-core 소스
└── run_eval.py          # 실행 진입점
```

## 실행 방법

```bash
cd /Users/future/Public/memory/msg
source venv/bin/activate

# Ollama 실행 확인
ollama serve &

# 평가 실행
python run_eval.py
```

## 참고: 공식 mem0 eval 코드 (`libs/mem0/evaluation/`)

mem0 공식 LoCoMo 평가 코드가 `libs/mem0/evaluation/`에 있음.
우리 구현과 주요 차이점:

| | 우리 구현 | 공식 mem0 eval |
|---|---|---|
| user_id | session_id 하나 | speaker별 2개 (`Caroline_0`, `Melanie_0`) |
| add/search | 한 번에 실행 | 단계 분리 (`--method add` → `--method search`) |
| LLM | 로컬 Ollama | mem0 cloud API + OpenAI |
| 검색 | 단일 user 검색 | 두 speaker 메모리 모두 검색 후 합산 |

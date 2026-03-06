# Memory Research Setup

mem0 오픈소스 기반으로 로컬 LLM + 커스텀 임베딩을 연결한 연구용 환경.
LoCoMo, LongMemEval 벤치마크 테스트 및 코드 수정 후 비교 실험 목적.

---

## 환경 구성

```
/Users/future/Public/memory/
├── venv/                  # Python 가상환경
├── mem0/                  # mem0ai 소스코드 (editable 설치)
├── chroma_db/             # Chroma 벡터 DB 저장 경로 (자동 생성)
└── config_example.py      # 설정 예시
```

### venv 활성화

```bash
source /Users/future/Public/memory/venv/bin/activate
```

### 설치된 패키지

- `mem0ai` — editable 모드 (`mem0/` 소스 직접 수정 가능)
- `chromadb` — 로컬 벡터 스토어
- `sentence-transformers` — Qwen3 임베딩용

---

## LLM 설정

### 1. Ollama (로컬)

설치된 모델:
- `qwen:32b` — 영어 벤치마크용 (추천)
- `exaone3.5:32b` — 한국어 특화 (영어 벤치마크엔 불리)

```python
"llm": {
    "provider": "ollama",
    "config": {
        "model": "qwen:32b",
        "ollama_base_url": "http://localhost:11434",
    }
}
```

### 2. 외부 API (Basic Auth, OpenAI-compatible)

커스텀 provider `basic_auth` 직접 구현.
API key 하나를 Base64 인코딩해서 `Authorization: Basic <key>` 헤더로 전송.

```python
"llm": {
    "provider": "basic_auth",
    "config": {
        "model": "model-name",
        "base_url": "https://your-api.com/v1",
        "api_key": "your_api_key",
    }
}
```

환경변수로도 설정 가능:
```bash
BASIC_AUTH_LLM_BASE_URL=https://your-api.com/v1
BASIC_AUTH_LLM_API_KEY=your_api_key
```

---

## 임베딩 설정

### Qwen3-Embedding-0.6B (커스텀 provider)

커스텀 provider `qwen_embedding` 직접 구현.

```python
"embedder": {
    "provider": "qwen_embedding",
    "config": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_dims": 1024,
    }
}
```

#### query / document 임베딩 구분이 중요한 이유

Qwen3-Embedding은 **검색 질문(query)**과 **저장 내용(document)**을 다르게 처리해야 성능이 제대로 나옴.

| 역할 | 처리 방식 |
|------|-----------|
| document (`add`, `update`) | 프롬프트 없이 그대로 임베딩 |
| query (`search`) | `"Instruct: ...\nQuery: "` 프롬프트 추가 후 임베딩 |

mem0의 `memory_action` 파라미터(`add`/`search`/`update`)로 자동 구분.
이 구분 없이 쓰면 retrieval 성능이 떨어져 벤치마크 점수가 실제보다 낮게 나옴.

query_instruction 변경:
```python
"model_kwargs": {"query_instruction": "Retrieve relevant passages for the query"}
```

---

## 벡터 스토어

Chroma (embedded 모드) — 서버 없이 로컬 파일로 저장.

```python
"vector_store": {
    "provider": "chroma",
    "config": {
        "collection_name": "memories",
        "path": "./chroma_db",
    }
}
```

---

## 커스텀 코드 위치

| 파일 | 설명 |
|------|------|
| `mem0/mem0/llms/basic_auth.py` | Basic Auth LLM provider |
| `mem0/mem0/configs/llms/basic_auth.py` | Basic Auth 설정 클래스 |
| `mem0/mem0/embeddings/qwen_embedding.py` | Qwen3 커스텀 embedder |

---

## 사용 예시

```python
from mem0 import Memory

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "qwen:32b",
            "ollama_base_url": "http://localhost:11434",
        }
    },
    "embedder": {
        "provider": "qwen_embedding",
        "config": {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dims": 1024,
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "memories",
            "path": "./chroma_db",
        }
    },
}

m = Memory.from_config(config)
m.add("John went to the store yesterday", user_id="user1")
results = m.search("Where did John go?", user_id="user1")
```

---

## 향후 작업

- [ ] LoCoMo 벤치마크 연결
- [ ] LongMemEval 벤치마크 연결
- [ ] 외부 API (non-OpenAI style) 커스텀 provider 추가

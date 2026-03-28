"""
임베딩 실험 공유 유틸리티 모듈.

- Qwen/Qwen2.5-0.5B-Instruct 모델 로딩 (싱글톤 캐시)
- Mean / Last Token 풀링 전환 가능한 임베딩 함수
- L2 정규화 적용 (코사인 유사도 = 내적)
- Gemini OpenAI 호환 엔드포인트 호출 (GEMINI_ENDPOINT, GEMINI_BEARER_TOKEN)
"""

import os
import time
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import requests

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"
GEMINI_MODEL: str = "gemini-1.5-flash"
RESULTS_DIR: str = os.path.join(os.path.dirname(__file__), "results")

PoolingMode = Literal["mean", "last_token"]

# ---------------------------------------------------------------------------
# 모델 싱글톤 캐시
# ---------------------------------------------------------------------------

_model_cache: Dict[str, Tuple] = {}  # {"model_name": (tokenizer, model, device)}


def load_qwen_model(model_name: str = EMBEDDING_MODEL) -> Tuple:
    """
    HuggingFace 모델과 토크나이저를 로드하고 캐싱.
    device 우선순위: MPS (iMac M4) → CUDA → CPU

    Returns:
        (tokenizer, model, device) 튜플
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            f"필수 라이브러리 미설치: {e}\n"
            "pip install -r embeddings/requirements.txt 실행 후 재시도"
        ) from e

    # device 선택
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"모델 로딩 중: {model_name} (device: {device})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    _model_cache[model_name] = (tokenizer, model, device)
    print(f"모델 로딩 완료: {model_name}")
    return tokenizer, model, device


# ---------------------------------------------------------------------------
# 임베딩
# ---------------------------------------------------------------------------

def get_embedding(
    text: str,
    model_name: str = EMBEDDING_MODEL,
    pooling: PoolingMode = "mean",
) -> np.ndarray:
    """
    텍스트 → L2 정규화된 임베딩 벡터 반환.

    Args:
        text: 임베딩할 텍스트
        model_name: HuggingFace 모델 이름
        pooling: "mean" | "last_token"
            - mean: attention_mask 기반 패딩 제외 평균 (의미적 유사도에 적합)
            - last_token: 마지막 유효 토큰 사용 (CLM 방식)

    Returns:
        shape (hidden_dim,) float32 ndarray, L2 norm = 1
    """
    if not text.strip():
        raise ValueError("빈 텍스트는 임베딩할 수 없습니다.")

    import torch

    tokenizer, model, device = load_qwen_model(model_name)

    # 토크나이징
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

    if pooling == "mean":
        # attention_mask 기반 평균 (패딩 토큰 제외)
        mask = attention_mask.unsqueeze(-1).float()  # (1, seq_len, 1)
        sum_hidden = (hidden_states * mask).sum(dim=1)  # (1, hidden_dim)
        count = mask.sum(dim=1).clamp(min=1e-9)        # (1, 1)
        pooled = (sum_hidden / count).squeeze(0)        # (hidden_dim,)
    elif pooling == "last_token":
        # 마지막 유효 토큰 인덱스
        last_idx = attention_mask.sum(dim=1) - 1  # (1,)
        pooled = hidden_states[0, last_idx[0], :]  # (hidden_dim,)
    else:
        raise ValueError(f"알 수 없는 pooling 방식: {pooling}")

    vec = pooled.float().cpu().numpy()

    # L2 정규화
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        raise ValueError("영벡터 임베딩 결과 — 입력 텍스트를 확인하세요.")
    return vec / norm


# ---------------------------------------------------------------------------
# 유사도
# ---------------------------------------------------------------------------

def get_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    L2 정규화된 벡터 간 코사인 유사도 = 내적.

    Returns:
        float in [-1.0, 1.0]
    """
    return float(np.dot(vec1, vec2))


# ---------------------------------------------------------------------------
# Gemini API (OpenAI 호환 엔드포인트)
# ---------------------------------------------------------------------------

def _check_gemini_env() -> Tuple[str, str]:
    """환경변수 존재 여부 확인. 미설정 시 즉시 ValueError."""
    endpoint = os.environ.get("GEMINI_ENDPOINT", "").strip().rstrip("/")
    token = os.environ.get("GEMINI_BEARER_TOKEN", "").strip()
    if not endpoint:
        raise ValueError(
            "GEMINI_ENDPOINT 환경변수가 설정되지 않았습니다.\n"
            'export GEMINI_ENDPOINT="https://your-endpoint/v1"'
        )
    if not token:
        raise ValueError(
            "GEMINI_BEARER_TOKEN 환경변수가 설정되지 않았습니다.\n"
            'export GEMINI_BEARER_TOKEN="your-bearer-token"'
        )
    # GEMINI_ENDPOINT가 /chat/completions 포함 여부 상관없이 올바른 URL 구성
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"
    return endpoint, token


def call_gemini(
    prompt: str,
    model_name: str = GEMINI_MODEL,
    max_retries: int = 3,
) -> str:
    """
    Gemini OpenAI 호환 엔드포인트 호출.

    POST {GEMINI_ENDPOINT}/chat/completions  (또는 GEMINI_ENDPOINT 자체가 전체 URL인 경우)
    Authorization: Bearer {GEMINI_BEARER_TOKEN}

    Args:
        prompt: 사용자 메시지
        model_name: 모델명 (기본값: gemini-1.5-flash)
        max_retries: 지수 백오프 최대 재시도 횟수

    Returns:
        생성된 텍스트 (stripped)

    Raises:
        ValueError: 환경변수 미설정 또는 빈 응답
        requests.exceptions.HTTPError: 400번대 클라이언트 에러 (재시도 없음)
        RuntimeError: max_retries 초과
    """
    url, token = _check_gemini_env()

    # curl과 동일한 payload — temperature 미포함 (일부 엔드포인트에서 400 유발)
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # DEBUG: 실제 요청 내용 출력 (문제 해결 후 제거 예정)
    print(f"  [DEBUG] URL: {url}")
    print(f"  [DEBUG] payload: {payload}")

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60,
            )
            if not resp.ok:
                print(f"  HTTP {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()  # 4xx/5xx → HTTPError

            result = resp.json()["choices"][0]["message"]["content"].strip()
            if not result:
                raise ValueError(f"Gemini 빈 응답 (prompt 앞부분: {prompt[:60]!r})")
            return result

        except requests.exceptions.HTTPError as e:
            # 400번대 클라이언트 에러 → 재시도 무의미
            if e.response is not None and 400 <= e.response.status_code < 500:
                raise
            last_exc = e
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_exc = e

        wait = 2 ** attempt  # 1s, 2s, 4s
        print(f"  Gemini API 재시도 {attempt + 1}/{max_retries} (대기 {wait}s): {last_exc}")
        time.sleep(wait)

    raise RuntimeError(
        f"Gemini API {max_retries}회 재시도 실패: {last_exc}"
    )

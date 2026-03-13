import sys
sys.path.insert(0, "libs/mem0")
from mem0 import Memory

# --- Ollama (로컬) ---
ollama_config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "exaone3.5:32b",        # 또는 "qwen:32b"
            "ollama_base_url": "http://localhost:11434",
        }
    },
    "embedder": {
        "provider": "qwen_embedding",
        "config": {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dims": 1024,
            # query_instruction을 바꾸고 싶으면 model_kwargs에 추가
            # "model_kwargs": {"query_instruction": "Retrieve relevant passages for the query"},
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "memories",
            "path": "./chroma_db",           # 로컬 파일로 저장
        }
    },
}

# --- External API (Basic Auth + /chat/completions) ---
external_config = {
    "llm": {
        "provider": "basic_auth",
        "config": {
            "model": "your-model-name",
            "base_url": "https://your-api-endpoint.com/v1",
            "api_key": "your_api_key",   # Base64 인코딩 후 Basic <key> 형식으로 전송
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

# 사용 예시
if __name__ == "__main__":
    # 원하는 config 선택
    m = Memory.from_config(ollama_config)
    # m = Memory.from_config(external_config)

    m.add("내 이름은 홍길동이야", user_id="user1")
    results = m.search("내 이름이 뭐야?", user_id="user1")
    for r in results:
        print(r)

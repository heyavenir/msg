"""
실행 예시: LoCoMo × mem0 × qwen:32b
"""
import sys
sys.path.insert(0, ".")

from benchmarks.locomo import LoCoMoBenchmark
from memory.mem0_adapter import Mem0Memory
from eval.runner import EvalRunner
from ollama import Client as OllamaClient

# --- LLM 설정 ---
ollama = OllamaClient(host="http://localhost:11434")
LLM_MODEL = "qwen:32b"

def answer_fn(question: str, memories: list[str]) -> str:
    """검색된 메모리를 컨텍스트로 LLM에 답변 생성 요청"""
    context = "\n".join(f"- {m}" for m in memories) if memories else "No relevant memory found."
    prompt = f"""You are a helpful assistant. Use the following memory context to answer the question.

Memory context:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    return response.message.content.strip()


# --- mem0 설정 ---
mem0_config = {
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
            "collection_name": "locomo_eval",
            "path": "./chroma_db",
        }
    },
}

if __name__ == "__main__":
    benchmark = LoCoMoBenchmark()
    benchmark.load()  # 기본값: data/locomo/locomo10.json

    memory = Mem0Memory(config=mem0_config)

    runner = EvalRunner(
        benchmark=benchmark,
        memory=memory,
        answer_fn=answer_fn,
        results_dir="results",
    )

    result = runner.run(top_k=5, verbose=True)
    print("\n=== 최종 점수 ===")
    for k, v in result["scores"].items():
        print(f"  {k}: {v:.4f}")

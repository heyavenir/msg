from typing import Literal, Optional

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers is required. Run: pip install sentence-transformers")

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_DIMS = 1024
# Qwen3-Embedding 권장 query instruction
DEFAULT_QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class QwenEmbedding(EmbeddingBase):
    """
    Qwen3-Embedding을 위한 커스텀 embedder.
    search(query)와 add/update(document)에 서로 다른 prompt를 적용해
    벤치마크 정확도를 높입니다.
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or DEFAULT_MODEL
        self.query_instruction = (
            self.config.model_kwargs.pop("query_instruction", DEFAULT_QUERY_INSTRUCTION)
        )

        self.model = SentenceTransformer(
            self.config.model,
            trust_remote_code=True,
            **self.config.model_kwargs,
        )
        self.config.embedding_dims = (
            self.config.embedding_dims or self.model.get_sentence_embedding_dimension() or DEFAULT_DIMS
        )

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        memory_action == "search" → query 임베딩 (instruction prompt 적용)
        그 외 (add / update / None) → document 임베딩 (prompt 없음)
        """
        if memory_action == "search":
            return self.model.encode(
                text,
                prompt=f"Instruct: {self.query_instruction}\nQuery: ",
                convert_to_numpy=True,
            ).tolist()
        else:
            return self.model.encode(text, convert_to_numpy=True).tolist()

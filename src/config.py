from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_model_fast: str = "llama-3.1-8b-instant"  # for router + grader (speed)

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "documents"

    # Retrieval
    embed_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    hybrid_alpha: float = 0.6  # weight for vector vs BM25 (1.0 = pure vector)

    # Semantic cache
    cache_similarity_threshold: float = 0.92
    cache_max_size: int = 500

    # Agent
    max_retries: int = 2
    complexity_threshold: float = 0.6  # above = multi-agent
    hitl_confidence_threshold: float = 0.4  # below = escalate to human

    # Langfuse (optional — set to empty to disable)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)


settings = Settings()

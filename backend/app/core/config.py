from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "MIRA Backend"
    debug: bool = True

    database_url: str = "postgresql://postgres:postgres@localhost:5432/mira"
    cors_origins: str = "http://localhost:5173"
    seed_demo_users: bool = True

    # Security & JWT configuration
    secret_key: str = "mira-development-secret-key-change-in-production-min32chars"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 12  # 12 hours

    # Seed Admin Defaults
    seed_admin_email: str = "admin@mira.gov.in"
    seed_admin_password: str = "Admin@123"

    # Milvus vector store
    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    milvus_collection: str = "material_embeddings"
    milvus_enabled: bool = True
    milvus_top_k: int = 50

    # Embedding model provisioning.
    # When the local MIRA model is absent, optionally fetch it from Hugging Face
    # instead of failing. The published repo holds the fine-tuned Epoch-2 INT8
    # checkpoint -- the same one the scoring thresholds were calibrated on.
    model_auto_download: bool = True
    model_hub_id: str = "AshIndian/Mira.ai"
    # Empty -> <project_root>/models  (already a searched path and gitignored)
    model_auto_download_dir: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

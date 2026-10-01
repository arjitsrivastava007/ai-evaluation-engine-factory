"""Runtime configuration loaded from the environment."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paths, upload limits, and browser origins for a local factory."""

    data_dir: Path = Path("data")
    max_upload_bytes: int = 10 * 1024 * 1024
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(env_prefix="EVAL_", extra="ignore")

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

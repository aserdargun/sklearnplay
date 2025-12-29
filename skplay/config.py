"""Centralized configuration management for sklearn-playground.

Uses pydantic-settings for environment variable loading with Azure-specific settings.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application settings
    app_name: str = "sklearn-playground"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # Azure PostgreSQL Flexible Server
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="skplayground", alias="POSTGRES_DB")
    postgres_user: str = Field(default="", alias="POSTGRES_USER")
    postgres_password: SecretStr = Field(default=SecretStr(""), alias="POSTGRES_PASSWORD")
    postgres_ssl_mode: str = Field(default="require", alias="POSTGRES_SSL_MODE")

    # Azure Blob Storage
    azure_storage_account: str = Field(default="", alias="AZURE_STORAGE_ACCOUNT")
    azure_storage_key: SecretStr = Field(default=SecretStr(""), alias="AZURE_STORAGE_KEY")
    azure_storage_connection_string: SecretStr = Field(
        default=SecretStr(""), alias="AZURE_STORAGE_CONNECTION_STRING"
    )
    azure_storage_container_models: str = Field(
        default="models", alias="AZURE_STORAGE_CONTAINER_MODELS"
    )
    azure_storage_container_datasets: str = Field(
        default="datasets", alias="AZURE_STORAGE_CONTAINER_DATASETS"
    )

    # Azure Entra ID (formerly Azure AD)
    azure_tenant_id: str = Field(default="", alias="AZURE_TENANT_ID")
    azure_client_id: str = Field(default="", alias="AZURE_CLIENT_ID")
    azure_client_secret: SecretStr = Field(default=SecretStr(""), alias="AZURE_CLIENT_SECRET")
    azure_redirect_uri: str = Field(
        default="http://localhost:8501/auth/callback", alias="AZURE_REDIRECT_URI"
    )

    # Session settings
    session_secret_key: SecretStr = Field(
        default=SecretStr("change-me-in-production"), alias="SESSION_SECRET_KEY"
    )
    session_expiry_hours: int = Field(default=24, alias="SESSION_EXPIRY_HOURS")

    # Feature flags
    enable_auth: bool = Field(default=False, alias="ENABLE_AUTH")
    enable_experiments: bool = Field(default=True, alias="ENABLE_EXPERIMENTS")
    enable_model_storage: bool = Field(default=True, alias="ENABLE_MODEL_STORAGE")

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Normalize environment name."""
        return v.lower()

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        password = self.postgres_password.get_secret_value()
        if not password:
            # Use SQLite for local development without PostgreSQL
            return "sqlite:///./skplayground.db"

        ssl_param = f"?sslmode={self.postgres_ssl_mode}" if self.postgres_ssl_mode else ""
        return (
            f"postgresql://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_db}{ssl_param}"
        )

    @property
    def async_database_url(self) -> str:
        """Construct async PostgreSQL connection URL."""
        password = self.postgres_password.get_secret_value()
        if not password:
            return "sqlite+aiosqlite:///./skplayground.db"

        ssl_param = f"?sslmode={self.postgres_ssl_mode}" if self.postgres_ssl_mode else ""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_db}{ssl_param}"
        )

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def is_azure_storage_configured(self) -> bool:
        """Check if Azure Blob Storage is configured."""
        return bool(
            self.azure_storage_connection_string.get_secret_value()
            or (self.azure_storage_account and self.azure_storage_key.get_secret_value())
        )

    @property
    def is_azure_auth_configured(self) -> bool:
        """Check if Azure Entra ID is configured."""
        return bool(
            self.azure_tenant_id
            and self.azure_client_id
            and self.azure_client_secret.get_secret_value()
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function for direct access
settings = get_settings()

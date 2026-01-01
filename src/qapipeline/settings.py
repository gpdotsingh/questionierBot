from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

# Optional dotenv import so we can read .env once for the entire project.
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - fallback when dotenv is missing
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:  # type: ignore[override]
        return False

# Optional provider SDKs.
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - library might be unavailable
    OpenAI = None  # type: ignore[assignment]

try:
    from ollama import Client as OllamaClient  # type: ignore
except Exception:  # pragma: no cover - library might be unavailable
    OllamaClient = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIClientType
    from ollama import Client as OllamaClientType
else:  # pragma: no cover - runtime aliases only
    OpenAIClientType = Any
    OllamaClientType = Any

_ENV_READY = False


def ensure_env_loaded() -> None:
    """
    Load the project's .env file only once. All modules call this helper so we
    do not scatter dotenv logic or pay repeated disk reads.
    """
    global _ENV_READY
    if _ENV_READY:
        return
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
    _ENV_READY = True


@dataclass(frozen=True)
class ProviderConfig:
    component: str
    provider: str
    openai_key: Optional[str]
    openai_model: str
    ollama_url: str
    ollama_model: str


@dataclass(frozen=True)
class ProviderRuntime:
    provider: str
    openai_client: Optional[OpenAIClientType]
    openai_model: str
    ollama_client: Optional[OllamaClientType]
    ollama_model: str


def _component_key(component: str) -> str:
    return f"{component.strip().upper()}_PROVIDER"


@lru_cache(maxsize=None)
def get_provider_config(component: str) -> ProviderConfig:
    """
    Read provider-specific values for a pipeline component from the environment.
    Values are cached so repeated lookups do not hit os.environ or disk again.
    """
    ensure_env_loaded()
    provider = (
        os.getenv(_component_key(component))
        or os.getenv("DEFAULT_PROVIDER")
        or ""
    ).lower().strip()
    return ProviderConfig(
        component=component,
        provider=provider,
        openai_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        ollama_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("OLLAMA_GEN_MODEL", "llama3.1"),
    )


@lru_cache(maxsize=None)
def get_provider_runtime(component: str) -> ProviderRuntime:
    """
    Create (and cache) provider SDK clients per component so the expensive
    connection objects are instantiated once per process.
    """
    cfg = get_provider_config(component)
    openai_client = None
    if cfg.provider == "openai" and OpenAI and cfg.openai_key:
        openai_client = OpenAI(api_key=cfg.openai_key)
    ollama_client = None
    if cfg.provider == "ollama" and OllamaClient:
        ollama_client = OllamaClient(host=cfg.ollama_url)
    return ProviderRuntime(
        provider=cfg.provider,
        openai_client=openai_client,
        openai_model=cfg.openai_model,
        ollama_client=ollama_client,
        ollama_model=cfg.ollama_model,
    )


__all__ = [
    "ensure_env_loaded",
    "get_provider_config",
    "get_provider_runtime",
    "ProviderConfig",
    "ProviderRuntime",
]

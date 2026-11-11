import os
import re
from dataclasses import dataclass, field

from .contracts import AgentError


BASE_URL = "https://routerai.ru/api/v1"
MODEL_ALIAS = "~deepseek/deepseek-v4-flash-latest"


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls):
        return cls(
            os.getenv("ROUTERAI_BASE_URL", BASE_URL).rstrip("/"),
            os.getenv("ROUTERAI_MODEL", MODEL_ALIAS),
            os.getenv("ROUTERAI_API_KEY", "").strip(),
        )

    def require_access(self):
        if self.base_url != BASE_URL or not re.fullmatch(r"~?deepseek/[a-z0-9][a-z0-9.-]*", self.model):
            raise AgentError("INVALID_ROUTERAI_CONFIG")
        if not self.api_key:
            raise AgentError("ROUTERAI_KEY_MISSING")

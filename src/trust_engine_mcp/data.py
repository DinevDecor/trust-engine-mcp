"""Public data from agenttrustapi.com: fetched with GET, cached in memory, bundled snapshot as fallback.

The only network calls this package makes are GETs of the two JSON files below, from ``TRUST_ENGINE_BASE_URL``
(default ``https://agenttrustapi.com``). No telemetry, no keys, no uploads.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from typing import Any

import httpx

BASE_URL_ENV = "TRUST_ENGINE_BASE_URL"
OFFLINE_ENV = "TRUST_ENGINE_OFFLINE"  # "1": never fetch, use the bundled snapshot only
DEFAULT_BASE_URL = "https://agenttrustapi.com"
TIMEOUT_SECONDS = 10.0
CACHE_SECONDS = 6 * 3600
FALLBACK_RETRY_SECONDS = 300  # after a failed fetch, try live again after 5 minutes

RETIREMENTS_PATH = "/retirements.json"
BENCHMARK_PATH = "/benchmarks/aider-model-diet.json"
TEMPLATE_PATH = "/pilot/tasks_template.csv"
SNAPSHOTS = {RETIREMENTS_PATH: "retirements.json", BENCHMARK_PATH: "aider-model-diet.json"}

ClientFactory = Callable[[], httpx.Client]


def bundled_text(name: str) -> str:
    return resources.files("trust_engine_mcp").joinpath("data", name).read_text(encoding="utf-8")


@dataclass
class Settings:
    base_url: str = DEFAULT_BASE_URL
    offline: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            base_url=(os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/"),
            offline=(os.environ.get(OFFLINE_ENV) or "").strip() in ("1", "true", "yes"),
        )


class DataSource:
    """GET + in-memory cache + bundled fallback. Each result says where it came from."""

    def __init__(
        self,
        settings: Settings,
        client_factory: ClientFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self._client_factory = client_factory or (
            lambda: httpx.Client(
                base_url=settings.base_url,
                timeout=TIMEOUT_SECONDS,
                follow_redirects=False,
                headers={"User-Agent": "trust-engine-mcp", "Accept": "application/json"},
            )
        )
        self._clock = clock
        self._cache: dict[str, tuple[float, dict[str, Any], dict[str, Any]]] = {}

    def url(self, path: str) -> str:
        return self.settings.base_url + path

    def get(self, path: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (data, provenance). Never raises for network problems: falls back to the bundled snapshot."""
        hit = self._cache.get(path)
        if hit and self._clock() < hit[0]:
            return hit[1], dict(hit[2], cached=True)
        error = None
        if not self.settings.offline:
            try:
                with self._client_factory() as client:
                    r = client.get(path)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, dict):
                        prov = {"source": "live", "url": self.url(path)}
                        self._cache[path] = (self._clock() + CACHE_SECONDS, data, prov)
                        return data, dict(prov, cached=False)
                    error = "unexpected JSON shape"
                else:
                    error = f"HTTP {r.status_code}"
            except (httpx.HTTPError, ValueError) as exc:
                error = type(exc).__name__
        data = json.loads(bundled_text(SNAPSHOTS[path]))
        prov = {
            "source": "bundled snapshot",
            "url": self.url(path),
            "snapshot_updated": data.get("updated") or data.get("date"),
            "why": "offline mode" if self.settings.offline else f"live fetch failed ({error})",
            "note": "Dates may have changed since the snapshot; check the source_url of each event.",
        }
        ttl = CACHE_SECONDS if self.settings.offline else FALLBACK_RETRY_SECONDS
        self._cache[path] = (self._clock() + ttl, data, prov)
        return data, dict(prov, cached=False)

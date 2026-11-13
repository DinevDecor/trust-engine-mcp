"""Find the model ids a codebase uses: local, read-only, text files only.

Every model id on the retirement calendar, its aliases (the id without a dated snapshot or ``-v1:0`` suffix) and its
Bedrock core (without ``anthropic.``) are matched as whole tokens, case-sensitively (API model ids are lower case).
Only candidates that ``check_my_models`` would also match are used, so both tools always agree.

Never read: ``.env`` and ``.env.*`` files (secrets), except ``.env.example``, ``.env.sample`` and ``.env.template``.
They are reported by name only. Skipped: version-control, dependency and build folders, lockfiles, binaries.
Nothing is sent anywhere.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trust_engine_mcp import models as mm

MAX_FILES = 2000
MAX_BYTES = 5 * 1024 * 1024
MAX_LOCATIONS = 50  # per model in the answer; the full count is reported
SKIP_DIRS = frozenset({".git", "node_modules", "venv", ".venv", "dist", "build", "__pycache__"})
LOCKFILES = frozenset({"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock", "Cargo.lock"})
ENV_TEMPLATES = frozenset({".env.example", ".env.sample", ".env.template"})
BINARY_SNIFF = 8192

_SUFFIX = re.compile(r"(-\d{8}|-\d{4}-\d{2}-\d{2}|-\d{2}-\d{4}|-\d{3}|@\d{3})?(-v\d+(:\d+)?)?$")
_SPECIFIC = re.compile(r"[a-z]{3}.*[-.]|[-.].*[a-z]{3}")
_BEFORE = r"(?<![A-Za-z0-9_\-])"  # "." and "/" may precede: us.anthropic.…, models/gemini-…
_AFTER = r"(?![A-Za-z0-9_\-]|\.[A-Za-z0-9])"  # gpt-5 must not match inside gpt-5.1 or gpt-5-mini


class ScanRefused(Exception):
    """The scan cannot run as asked; the message is safe to show."""


def is_secret_env(name: str) -> bool:
    return (name == ".env" or name.startswith(".env.")) and name not in ENV_TEMPLATES


def candidates(events: list[dict[str, Any]]) -> list[str]:
    """Strings to look for: calendar ids plus aliases that check_my_models resolves to the same events."""
    literal: set[str] = set()
    derived: set[str] = set()
    for e in events:
        for mid in e.get("model_ids") or []:
            m = mid.strip()
            if not m:
                continue
            literal.add(m)
            stripped = _SUFFIX.sub("", m) if _SUFFIX.search(m) else m
            for v in mm.variants(m) | mm.variants(stripped) | {stripped, mm.core(stripped)}:
                if v != m:
                    derived.add(v)
    # Derived forms must look like a model id (a word, a digit and a separator: not "r1-v1:0" or "text-embedding")
    # and must resolve to the same calendar events in check_my_models.
    keep = {d for d in derived - literal if _SPECIFIC.search(d) and re.search(r"\d", d) and mm.find(d, events)}
    return sorted(literal | keep, key=lambda s: (-len(s), s))


def compile_matcher(events: list[dict[str, Any]]) -> re.Pattern[str]:
    alts = "|".join(re.escape(c) for c in candidates(events))
    return re.compile(f"{_BEFORE}({alts}){_AFTER}")


@dataclass
class ScanResult:
    root: str
    hits: dict[str, list[str]] = field(default_factory=dict)  # model id as written -> ["file:line", ...]
    files_scanned: int = 0
    bytes_scanned: int = 0
    skipped_env_files: list[str] = field(default_factory=list)
    skipped_binary: int = 0
    skipped_other: int = 0
    limits_hit: list[str] = field(default_factory=list)

    def add_text(self, rx: re.Pattern[str], name: str, text: str) -> None:
        for n, line in enumerate(text.splitlines(), start=1):
            if len(line) > 20000:  # minified bundles: skip the line, not the file
                continue
            for m in rx.finditer(line):
                locs = self.hits.setdefault(m.group(1), [])
                loc = f"{name}:{n}"
                if not locs or locs[-1] != loc:
                    locs.append(loc)

    def summary(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "files_scanned": self.files_scanned,
            "bytes_scanned": self.bytes_scanned,
            "skipped_env_files_not_read": self.skipped_env_files,
            "skipped_binary_files": self.skipped_binary,
            "skipped_other_files": self.skipped_other,
            "limits_hit": self.limits_hit,
            "complete": not self.limits_hit,
        }


def resolve_inside(path: str | None, base: Path) -> Path:
    """Resolve ``path`` (relative to ``base``) and refuse anything outside ``base``."""
    base = base.resolve()
    target = (base / (path or ".")).resolve()
    if target != base and base not in target.parents:
        raise ScanRefused(
            f"path must be inside the current working directory ({base}); got {path!r}. "
            "Pass a relative path inside it, or pass texts instead."
        )
    if not target.exists():
        raise ScanRefused(f"path not found: {path!r}")
    return target


def scan_path(rx: re.Pattern[str], target: Path, base: Path) -> ScanResult:
    """Walk ``target`` (a file or folder inside ``base``); locations are relative to ``base``."""
    base = base.resolve()
    res = ScanResult(root=str(target))

    def rel(p: Path) -> str:
        try:
            return p.relative_to(base).as_posix()
        except ValueError:
            return p.as_posix()

    def files() -> Any:
        if target.is_file():
            yield target
            return
        for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for f in sorted(filenames):
                yield Path(dirpath) / f

    for p in files():
        name = p.name
        if is_secret_env(name):
            res.skipped_env_files.append(rel(p))
            continue
        if name in LOCKFILES or p.is_symlink() or not p.is_file():
            res.skipped_other += 1
            continue
        if res.files_scanned >= MAX_FILES:
            res.limits_hit.append(f"stopped after {MAX_FILES} files; results are partial")
            break
        try:
            size = p.stat().st_size
        except OSError:
            res.skipped_other += 1
            continue
        if res.bytes_scanned + size > MAX_BYTES:
            res.limits_hit.append(f"stopped at the {MAX_BYTES // (1024 * 1024)} MB limit; results are partial")
            break
        try:
            with open(p, "rb") as fh:
                raw = fh.read(size + 1)
        except OSError:
            res.skipped_other += 1
            continue
        if b"\0" in raw[:BINARY_SNIFF]:
            res.skipped_binary += 1
            continue
        res.files_scanned += 1
        res.bytes_scanned += len(raw)
        res.add_text(rx, rel(p), raw.decode("utf-8", errors="replace"))
    return res


def scan_texts(rx: re.Pattern[str], texts: list[dict[str, Any]]) -> ScanResult:
    """Scan texts an agent passes as [{"name": ..., "content": ...}], with the same limits and .env rule."""
    res = ScanResult(root="(texts passed by the agent)")
    for i, t in enumerate(texts):
        if not isinstance(t, dict):
            raise ScanRefused("texts must be a list of objects with name and content")
        name = str(t.get("name") or f"text{i + 1}")
        content = t.get("content")
        if not isinstance(content, str):
            raise ScanRefused(f"texts[{i}].content must be a string")
        if is_secret_env(Path(name).name):
            res.skipped_env_files.append(name)
            continue
        if res.files_scanned >= MAX_FILES:
            res.limits_hit.append(f"stopped after {MAX_FILES} texts; results are partial")
            break
        size = len(content.encode("utf-8"))
        if res.bytes_scanned + size > MAX_BYTES:
            res.limits_hit.append(f"stopped at the {MAX_BYTES // (1024 * 1024)} MB limit; results are partial")
            break
        if "\0" in content[:BINARY_SNIFF]:
            res.skipped_binary += 1
            continue
        res.files_scanned += 1
        res.bytes_scanned += size
        res.add_text(rx, name, content)
    return res

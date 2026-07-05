"""Configuration helpers for local, Colab, and notebook execution."""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_API_KEYS = ("OPENAI_API_KEY", "SERPER_API_KEY")


class MissingEnvironmentVariable(RuntimeError):
    """Raised when a required environment variable is unavailable."""


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by planner runs."""

    project_dir: Path
    results_dir: Path
    log_dir: Path

    def to_json_dict(self) -> dict[str, str]:
        return {
            "project_dir": str(self.project_dir),
            "results_dir": str(self.results_dir),
            "log_dir": str(self.log_dir),
        }


def configure_paths(
    project_dir: str | Path | None = None,
    *,
    create: bool = True,
) -> ProjectPaths:
    """Return output paths, defaulting to a local ``outputs/`` directory."""

    configured_dir = project_dir or os.getenv("STUDY_PLANNER_PROJECT_DIR") or "outputs"
    root = Path(configured_dir).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root

    paths = ProjectPaths(
        project_dir=root,
        results_dir=root / "variation_outputs",
        log_dir=root / "tensorboard_logs",
    )
    if create:
        paths.results_dir.mkdir(parents=True, exist_ok=True)
        paths.log_dir.mkdir(parents=True, exist_ok=True)
    return paths


def configure_colab_paths(
    project_dir: str | Path,
    *,
    create: bool = True,
) -> ProjectPaths:
    """Mount Google Drive in Colab, then configure paths under ``project_dir``."""

    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("Google Colab is not available in this environment.") from exc

    drive.mount("/content/drive")
    return configure_paths(project_dir=project_dir, create=create)


def load_api_keys(
    *,
    env_file: str | Path | None = ".env",
    required_keys: Iterable[str] = REQUIRED_API_KEYS,
    prompt_missing: bool = True,
) -> dict[str, str]:
    """Load required API keys from ``.env``/environment, prompting only if missing."""

    if env_file is not None:
        _load_env_file(Path(env_file))

    keys: dict[str, str] = {}
    missing: list[str] = []
    for key in required_keys:
        value = os.getenv(key)
        if value:
            keys[key] = value
        else:
            missing.append(key)

    if missing and not prompt_missing:
        raise MissingEnvironmentVariable(
            "Missing required environment variables: " + ", ".join(missing)
        )

    for key in missing:
        value = getpass.getpass(f"Enter your {key}: ").strip()
        if not value:
            raise MissingEnvironmentVariable(f"Missing required environment variable: {key}")
        os.environ[key] = value
        keys[key] = value

    return keys


def _load_env_file(path: Path) -> None:
    """Load an env file with python-dotenv when available, with a small fallback."""

    if not path.exists():
        try:
            from dotenv import load_dotenv
        except ImportError:
            return
        load_dotenv()
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_simple_env_file(path)
    else:
        load_dotenv(path)


def _load_simple_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

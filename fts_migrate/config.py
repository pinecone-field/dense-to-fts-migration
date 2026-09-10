"""Configuration loading for the migration toolkit.

One `Settings` object is threaded through every phase so that the CLI, the notebook
and `sync.py` all agree on which indexes, namespaces and directories are in play.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


class ConfigError(RuntimeError):
    """Raised when the config file is missing, malformed, or internally inconsistent."""


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


@dataclass(frozen=True)
class SourceSettings:
    index: str
    namespace: str = "__default__"
    text_metadata_key: str = "text"


@dataclass(frozen=True)
class TargetSettings:
    index: str
    dense_field: str
    text_fields: list[str]
    namespace: str = "__default__"
    full_text_search: dict[str, Any] = field(default_factory=dict)
    deployment: dict[str, Any] = field(default_factory=dict)
    read_capacity: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_text_field(self) -> str:
        return self.text_fields[0]


@dataclass(frozen=True)
class ExportSettings:
    dir: Path = Path("work/export")
    rows_per_file: int = 25_000


@dataclass(frozen=True)
class ConvertSettings:
    dir: Path = Path("work/jsonl")
    gzip: bool = True
    drop_fields: list[str] = field(default_factory=list)
    rename_fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ImportSettings:
    mode: str = "upsert"
    uri: str = ""
    integration_id: str = ""
    error_mode: str = "continue"


@dataclass(frozen=True)
class CdcSettings:
    db: Path = Path("work/cdc.sqlite")
    batch_size: int = 500


@dataclass(frozen=True)
class DemoSettings:
    records: int = 2000
    dimension: int = 384
    write_rate: float = 5.0


@dataclass(frozen=True)
class Settings:
    source: SourceSettings
    target: TargetSettings
    export: ExportSettings
    convert: ConvertSettings
    import_: ImportSettings
    cdc: CdcSettings
    demo: DemoSettings
    path: Path

    @property
    def api_key(self) -> str:
        key = os.environ.get("PINECONE_API_KEY", "").strip()
        if not key:
            raise ConfigError(
                "PINECONE_API_KEY is not set. Copy .env.example to .env, fill it in, "
                "and export it (e.g. `set -a; source .env; set +a`)."
            )
        return key


def load_settings(path: str | Path | None = None) -> Settings:
    """Read a config file into `Settings`, expanding ${ENV_VAR} references."""
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        raise ConfigError(
            f"{config_path} not found. Copy config.example.yaml to {config_path} and edit it."
        )
    raw = _expand(yaml.safe_load(config_path.read_text()) or {})

    try:
        source = SourceSettings(**raw["source"])
        target = TargetSettings(**raw["target"])
    except KeyError as exc:
        raise ConfigError(f"{config_path} is missing the {exc} section.") from exc
    except TypeError as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc

    if not target.text_fields:
        raise ConfigError(
            "target.text_fields is empty. An FTS index with no full-text field cannot "
            "do BM25 — declare at least one."
        )

    export_raw = dict(raw.get("export") or {})
    convert_raw = dict(raw.get("convert") or {})
    cdc_raw = dict(raw.get("cdc") or {})
    if "dir" in export_raw:
        export_raw["dir"] = Path(export_raw["dir"])
    if "dir" in convert_raw:
        convert_raw["dir"] = Path(convert_raw["dir"])
    if "db" in cdc_raw:
        cdc_raw["db"] = Path(cdc_raw["db"])

    return Settings(
        source=source,
        target=target,
        export=ExportSettings(**export_raw),
        convert=ConvertSettings(**convert_raw),
        import_=ImportSettings(**(raw.get("import") or {})),
        cdc=CdcSettings(**cdc_raw),
        demo=DemoSettings(**(raw.get("demo") or {})),
        path=config_path,
    )

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fts_migrate.config import (
    CdcSettings,
    ConvertSettings,
    DemoSettings,
    ExportSettings,
    ImportSettings,
    Settings,
    SourceSettings,
    TargetSettings,
)
from fts_migrate.convert import (
    ConversionError,
    DocumentMapper,
    MissingTextError,
    iter_jsonl_dir,
    parquet_to_jsonl,
)
from fts_migrate.simulate import EXPORT_SCHEMA


def make_settings(**overrides) -> Settings:
    base = dict(
        source=SourceSettings(index="src", namespace="__default__", text_metadata_key="text"),
        target=TargetSettings(
            index="dst",
            dense_field="embedding",
            text_fields=["text"],
            namespace="__default__",
        ),
        export=ExportSettings(),
        convert=ConvertSettings(),
        import_=ImportSettings(),
        cdc=CdcSettings(),
        demo=DemoSettings(),
        path=Path("config.yaml"),
    )
    base.update(overrides)
    return Settings(**base)


def write_export(path: Path, rows: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=EXPORT_SCHEMA)
    pq.write_table(table, path / "0.parquet")


def row(record_id: str, metadata: dict, dimension: int = 3) -> dict:
    return {
        "id": record_id,
        "values": [0.1] * dimension,
        "metadata": json.dumps(metadata),
    }


def test_maps_metadata_to_top_level_fields():
    mapper = DocumentMapper(make_settings(), dimension=3)
    doc = mapper.from_record("a", [0.1, 0.2, 0.3], {"text": "hello world", "year": 2024})
    assert doc == {
        "_id": "a",
        "embedding": [0.1, 0.2, 0.3],
        "text": "hello world",
        "year": 2024,
    }


def test_dimension_mismatch_is_rejected():
    mapper = DocumentMapper(make_settings(), dimension=3)
    with pytest.raises(ConversionError, match="expected 3"):
        mapper.from_record("a", [0.1, 0.2], {"text": "hello"})


def test_missing_dense_values_are_rejected():
    mapper = DocumentMapper(make_settings(), dimension=3)
    with pytest.raises(ConversionError, match="no dense values"):
        mapper.from_record("a", None, {"text": "hello"})


def test_reserved_field_names_are_rejected():
    mapper = DocumentMapper(make_settings(), dimension=3)
    with pytest.raises(ConversionError, match="reserved character"):
        mapper.from_record("a", [0.1, 0.2, 0.3], {"text": "hi", "_internal": "x"})


def test_numeric_list_metadata_is_rejected():
    mapper = DocumentMapper(make_settings(), dimension=3)
    with pytest.raises(ConversionError, match="list of non-strings"):
        mapper.from_record("a", [0.1, 0.2, 0.3], {"text": "hi", "scores": [1, 2]})


def test_rename_and_drop_are_applied():
    settings = make_settings(
        convert=ConvertSettings(rename_fields={"body": "text"}, drop_fields=["junk"])
    )
    mapper = DocumentMapper(settings, dimension=3)
    doc = mapper.from_record("a", [0.1, 0.2, 0.3], {"body": "renamed", "junk": "gone"})
    assert doc["text"] == "renamed"
    assert "junk" not in doc


def test_export_with_text_converts_every_row(tmp_path: Path):
    export_dir = tmp_path / "export"
    write_export(
        export_dir, [row(f"doc-{i}", {"text": f"body {i}", "year": 2024}) for i in range(5)]
    )

    out_dir = tmp_path / "jsonl"
    stats = parquet_to_jsonl(export_dir, out_dir, DocumentMapper(make_settings(), dimension=3))

    assert stats.rows == 5
    assert stats.converted == 5
    docs = list(iter_jsonl_dir(out_dir))
    assert {d["_id"] for d in docs} == {f"doc-{i}" for i in range(5)}
    assert all("embedding" in d and "text" in d for d in docs)


def test_export_without_text_fails_loudly(tmp_path: Path):
    export_dir = tmp_path / "export"
    write_export(export_dir, [row("doc-0", {"year": 2024})])

    with pytest.raises(MissingTextError, match="no text for full-text field"):
        parquet_to_jsonl(
            export_dir, tmp_path / "jsonl", DocumentMapper(make_settings(), dimension=3)
        )


def test_export_without_text_can_skip_rows(tmp_path: Path):
    export_dir = tmp_path / "export"
    write_export(
        export_dir,
        [row("doc-0", {"year": 2024}), row("doc-1", {"text": "has text", "year": 2024})],
    )

    out_dir = tmp_path / "jsonl"
    stats = parquet_to_jsonl(
        export_dir,
        out_dir,
        DocumentMapper(make_settings(), dimension=3, allow_missing_text=True),
    )

    assert stats.skipped_missing_text == 1
    assert stats.converted == 1
    assert [d["_id"] for d in iter_jsonl_dir(out_dir)] == ["doc-1"]


def test_shards_are_split_and_gzipped(tmp_path: Path):
    export_dir = tmp_path / "export"
    write_export(export_dir, [row(f"doc-{i}", {"text": "body"}) for i in range(10)])

    out_dir = tmp_path / "jsonl"
    parquet_to_jsonl(
        export_dir, out_dir, DocumentMapper(make_settings(), dimension=3), rows_per_file=4
    )

    shards = sorted(p.name for p in out_dir.glob("*.jsonl.gz"))
    assert shards == ["0.jsonl.gz", "1.jsonl.gz", "2.jsonl.gz"]
    with gzip.open(out_dir / "0.jsonl.gz", "rt") as handle:
        assert len(handle.readlines()) == 4
    assert len(list(iter_jsonl_dir(out_dir))) == 10


def test_reconvert_does_not_leave_stale_shards_behind(tmp_path: Path):
    """A re-run after deletions produces fewer shards; the old ones must not survive
    to reload documents the source no longer has."""
    export_dir = tmp_path / "export"
    out_dir = tmp_path / "jsonl"
    settings = make_settings()

    write_export(export_dir, [row(f"doc-{i}", {"text": "body"}) for i in range(10)])
    parquet_to_jsonl(export_dir, out_dir, DocumentMapper(settings, dimension=3), rows_per_file=4)
    assert len(list(iter_jsonl_dir(out_dir))) == 10

    write_export(export_dir, [row(f"doc-{i}", {"text": "body"}) for i in range(3)])
    parquet_to_jsonl(export_dir, out_dir, DocumentMapper(settings, dimension=3), rows_per_file=4)

    remaining = [d["_id"] for d in iter_jsonl_dir(out_dir)]
    assert remaining == ["doc-0", "doc-1", "doc-2"]


def test_unset_env_var_is_blanked_and_reported_not_sent_verbatim(tmp_path: Path, monkeypatch):
    """An unset ${VAR} must never reach the API as a literal, but it also must not stop
    the phases that do not need it."""
    import pytest as _pytest
    import yaml as _yaml

    from fts_migrate.config import ConfigError, load_settings

    monkeypatch.delenv("SOME_UNSET_INTEGRATION_ID", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(
        _yaml.safe_dump(
            {
                "source": {"index": "src"},
                "target": {"index": "dst", "dense_field": "embedding", "text_fields": ["text"]},
                "import": {"integration_id": "${SOME_UNSET_INTEGRATION_ID}"},
            }
        )
    )

    settings = load_settings(path)

    assert settings.import_.integration_id == ""
    assert settings.unresolved_vars == ("SOME_UNSET_INTEGRATION_ID",)
    with _pytest.raises(ConfigError, match="SOME_UNSET_INTEGRATION_ID"):
        settings.require_env("SOME_UNSET_INTEGRATION_ID", "the import path needs it")


def test_the_shipped_example_config_loads_with_no_env_vars_set(monkeypatch):
    from fts_migrate.config import load_settings

    monkeypatch.delenv("PINECONE_STORAGE_INTEGRATION_ID", raising=False)
    settings = load_settings("config.example.yaml")
    assert settings.import_.integration_id == ""


def test_over_long_record_ids_are_rejected():
    mapper = DocumentMapper(make_settings(), dimension=3)
    with pytest.raises(ConversionError, match="over the 512 character limit"):
        mapper.from_record("x" * 513, [0.1, 0.2, 0.3], {"text": "hi"})

    assert mapper.from_record("x" * 512, [0.1, 0.2, 0.3], {"text": "hi"})["_id"] == "x" * 512


def test_text_fields_may_carry_per_field_analyzer_options():
    """n-grams can't share a field with stemming, so each field needs its own options."""
    import tempfile

    import yaml as _yaml

    from fts_migrate.config import load_settings
    from fts_migrate.dense_source import SourceSpec
    from fts_migrate.target_index import build_schema

    config = {
        "source": {"index": "src"},
        "target": {
            "index": "dst",
            "dense_field": "embedding",
            "text_fields": {
                "text": {"language": "en", "stemming": True},
                "sku": {"ngram": {"min_gram": 3, "max_gram": 4, "prefix_only": False}},
            },
        },
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(_yaml.safe_dump(config, sort_keys=False))
        path = handle.name

    settings = load_settings(path)
    assert settings.target.text_fields == ["text", "sku"]
    assert settings.target.primary_text_field == "text"

    fields = build_schema(settings, SourceSpec("src", 8, "cosine", "h"))["fields"]
    assert fields["text"]["full_text_search"] == {"language": "en", "stemming": True}
    assert fields["sku"]["full_text_search"]["ngram"]["max_gram"] == 4
    assert "stemming" not in fields["sku"]["full_text_search"]


def _write(tmp_path: Path, config: dict, schema: dict | None = None) -> Path:
    import json as _json

    import yaml as _yaml

    if schema is not None:
        (tmp_path / "schema.json").write_text(_json.dumps(schema))
    path = tmp_path / "config.yaml"
    path.write_text(_yaml.safe_dump(config, sort_keys=False))
    return path


def _base_config() -> dict:
    return {
        "source": {"index": "src"},
        "target": {"index": "dst", "schema_file": "schema.json"},
    }


def test_a_json_schema_is_sent_as_written(tmp_path: Path):
    """The point of declaring it as JSON is that what you reviewed is what is sent."""
    from fts_migrate.config import load_settings
    from fts_migrate.dense_source import SourceSpec
    from fts_migrate.target_index import build_schema

    schema = {
        "fields": {
            "embedding": {"type": "dense_vector", "dimension": 8, "metric": "cosine"},
            "body": {"type": "string", "full_text_search": {"language": "en"}},
            "sku": {"type": "string", "full_text_search": {"ngram": {"min_gram": 3}}},
        }
    }
    settings = load_settings(_write(tmp_path, _base_config(), schema))

    assert settings.target.dense_field == "embedding"
    assert settings.target.text_fields == ["body", "sku"]
    assert build_schema(settings, SourceSpec("src", 8, "cosine", "h")) == schema


def test_dimension_and_metric_are_filled_in_from_the_source(tmp_path: Path):
    from fts_migrate.config import load_settings
    from fts_migrate.dense_source import SourceSpec
    from fts_migrate.target_index import build_schema

    schema = {
        "fields": {
            "embedding": {"type": "dense_vector"},
            "body": {"type": "string", "full_text_search": {}},
        }
    }
    settings = load_settings(_write(tmp_path, _base_config(), schema))
    built = build_schema(settings, SourceSpec("src", 1536, "dotproduct", "h"))

    assert built["fields"]["embedding"]["dimension"] == 1536
    assert built["fields"]["embedding"]["metric"] == "dotproduct"
    assert settings.target.schema["fields"]["embedding"] == {"type": "dense_vector"}


def test_a_schema_that_disagrees_with_the_source_is_refused(tmp_path: Path):
    from fts_migrate.config import load_settings
    from fts_migrate.dense_source import SourceSpec
    from fts_migrate.target_index import TargetError, build_schema

    schema = {
        "fields": {
            "embedding": {"type": "dense_vector", "dimension": 768, "metric": "cosine"},
            "body": {"type": "string", "full_text_search": {}},
        }
    }
    settings = load_settings(_write(tmp_path, _base_config(), schema))

    with pytest.raises(TargetError, match="would not fit"):
        build_schema(settings, SourceSpec("src", 1536, "cosine", "h"))

    schema["fields"]["embedding"]["dimension"] = 1536
    schema["fields"]["embedding"]["metric"] = "euclidean"
    settings = load_settings(_write(tmp_path, _base_config(), schema))
    with pytest.raises(TargetError, match="Ranking would not match"):
        build_schema(settings, SourceSpec("src", 1536, "cosine", "h"))


def test_a_schema_without_a_searchable_field_is_refused(tmp_path: Path):
    from fts_migrate.config import ConfigError, load_settings

    schema = {"fields": {"embedding": {"type": "dense_vector", "dimension": 8, "metric": "cosine"}}}
    with pytest.raises(ConfigError, match="no string field with full_text_search"):
        load_settings(_write(tmp_path, _base_config(), schema))


def test_a_schema_copied_from_a_legacy_index_is_refused(tmp_path: Path):
    """describe_index on a legacy dense index reports its vector as `_values`, so copying
    that schema forward is a natural mistake. The API rejects it at creation; catch it at
    config load, before an export and a conversion have been spent."""
    from fts_migrate.config import ConfigError, load_settings

    schema = {
        "fields": {
            "_values": {"type": "dense_vector", "dimension": 1024, "metric": "cosine"},
            "_sparse_values": {"type": "sparse_vector"},
            "text": {"type": "string", "full_text_search": {"language": "en"}},
        }
    }
    with pytest.raises(ConfigError, match="_sparse_values, _values") as exc:
        load_settings(_write(tmp_path, _base_config(), schema))
    assert "embedding" in str(exc.value)

"""Turn dense-index records into full-text-search documents.

`DocumentMapper` is the single place where a dense record (`id`, `values`, `metadata`)
becomes a document (`_id`, dense field, text fields, metadata fields). Both the bulk
path (Parquet -> JSONL) and the CDC replay path go through it, so a document that
lands via bulk import is byte-identical to the same document arriving via upsert.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .config import Settings

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_TEXT_FIELD_BYTES = 100 * 1024
MAX_TEXT_FIELD_TOKENS = 10_000
MAX_METADATA_BYTES = 40 * 1024
MAX_FIELD_NAME_BYTES = 64
MAX_ERRORS_KEPT = 20


class ConversionError(RuntimeError):
    """A record could not be turned into a valid document."""


class MissingTextError(ConversionError):
    """The record has no text for a declared full-text-search field.

    This is the "export without source text in metadata" case. There is nothing the
    toolkit can invent here: BM25 needs the words. Join the text back in from your
    system of record before converting, or pass --allow-missing-text to skip these
    rows and load them later.
    """


@dataclass
class MappingStats:
    rows: int = 0
    converted: int = 0
    skipped_missing_text: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def record_error(self, message: str) -> None:
        self.failed += 1
        if len(self.errors) < MAX_ERRORS_KEPT:
            self.errors.append(message)

    def summary(self) -> str:
        parts = [f"{self.rows} rows", f"{self.converted} converted"]
        if self.skipped_missing_text:
            parts.append(f"{self.skipped_missing_text} skipped (no text)")
        if self.failed:
            parts.append(f"{self.failed} failed")
        return ", ".join(parts)


def _is_metadata_scalar(value: Any) -> bool:
    return isinstance(value, (str, bool, int, float))


def _validate_field_name(name: str) -> None:
    if not name:
        raise ConversionError("field name is empty")
    if name[0] in "_$":
        raise ConversionError(
            f"field name {name!r} starts with a reserved character. "
            f"Rename it via convert.rename_fields or drop it via convert.drop_fields."
        )
    if len(name.encode()) > MAX_FIELD_NAME_BYTES:
        raise ConversionError(f"field name {name!r} exceeds {MAX_FIELD_NAME_BYTES} bytes")


class DocumentMapper:
    """Maps one dense record to one document, enforcing the document-API limits."""

    def __init__(
        self,
        settings: Settings,
        dimension: int | None = None,
        allow_missing_text: bool = False,
    ) -> None:
        self.dense_field = settings.target.dense_field
        self.text_fields = list(settings.target.text_fields)
        self.rename_fields = dict(settings.convert.rename_fields)
        self.drop_fields = set(settings.convert.drop_fields)
        self.dimension = dimension
        self.allow_missing_text = allow_missing_text

    def from_record(
        self,
        record_id: str,
        values: Sequence[float] | None,
        metadata: Mapping[str, Any] | None,
        sparse_values: Mapping[str, Any] | None = None,
        sparse_field: str | None = None,
    ) -> dict[str, Any] | None:
        """Build a document, or return None when the row is skipped for missing text."""
        if not record_id:
            raise ConversionError("record has an empty id")

        doc: dict[str, Any] = {"_id": record_id}

        if values is None:
            raise ConversionError(
                f"{record_id}: no dense values. The target schema declares "
                f"{self.dense_field!r}, and every document must carry it."
            )
        values = list(values)
        if self.dimension is None:
            self.dimension = len(values)
        if len(values) != self.dimension:
            raise ConversionError(
                f"{record_id}: dense vector has {len(values)} values, expected {self.dimension}"
            )
        doc[self.dense_field] = values

        if sparse_values and sparse_field:
            doc[sparse_field] = {
                "indices": list(sparse_values["indices"]),
                "values": list(sparse_values["values"]),
            }

        for key, value in self._clean_metadata(metadata or {}, record_id):
            doc[key] = value

        missing = [f for f in self.text_fields if not isinstance(doc.get(f), str) or not doc[f]]
        if missing:
            if self.allow_missing_text:
                return None
            raise MissingTextError(
                f"{record_id}: no text for full-text field(s) {', '.join(missing)}. "
                f"The export carries no searchable text for this record — join it in from "
                f"your system of record before converting, or rerun with --allow-missing-text "
                f"to skip these rows."
            )

        self._check_sizes(doc, record_id)
        return doc

    def from_partial(
        self,
        record_id: str,
        values: Sequence[float] | None = None,
        set_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a `documents.update` patch from a partial write to the source index."""
        patch: dict[str, Any] = {"_id": record_id}
        if values is not None:
            patch[self.dense_field] = list(values)
        for key, value in self._clean_metadata(set_metadata or {}, record_id):
            patch[key] = value
        return patch

    def _clean_metadata(
        self, metadata: Mapping[str, Any], record_id: str
    ) -> Iterator[tuple[str, Any]]:
        for raw_key, value in metadata.items():
            key = self.rename_fields.get(raw_key, raw_key)
            if key in self.drop_fields or raw_key in self.drop_fields:
                continue
            if value is None:
                continue
            _validate_field_name(key)
            if key in (self.dense_field, "_id"):
                raise ConversionError(
                    f"{record_id}: metadata key {key!r} collides with a schema field. "
                    f"Rename it via convert.rename_fields."
                )
            if isinstance(value, list):
                if not all(isinstance(v, str) for v in value):
                    raise ConversionError(
                        f"{record_id}: field {key!r} is a list of non-strings. Only lists of "
                        f"strings are storable as metadata."
                    )
            elif not _is_metadata_scalar(value):
                raise ConversionError(
                    f"{record_id}: field {key!r} has unsupported type {type(value).__name__}"
                )
            yield key, value

    def _check_sizes(self, doc: Mapping[str, Any], record_id: str) -> None:
        for name in self.text_fields:
            text = doc[name]
            if len(text.encode()) > MAX_TEXT_FIELD_BYTES:
                raise ConversionError(
                    f"{record_id}: text field {name!r} is "
                    f"{len(text.encode())} bytes, over the {MAX_TEXT_FIELD_BYTES} byte limit"
                )
            if len(text.split()) > MAX_TEXT_FIELD_TOKENS:
                raise ConversionError(
                    f"{record_id}: text field {name!r} has more than "
                    f"{MAX_TEXT_FIELD_TOKENS} tokens"
                )

        metadata_only = {
            k: v
            for k, v in doc.items()
            if k not in self.text_fields and k != self.dense_field and k != "_id"
        }
        metadata_bytes = len(json.dumps(metadata_only).encode())
        if metadata_bytes > MAX_METADATA_BYTES:
            raise ConversionError(
                f"{record_id}: metadata is {metadata_bytes} bytes, over the "
                f"{MAX_METADATA_BYTES} byte limit (full-text fields are exempt)"
            )

        doc_bytes = len(json.dumps(doc).encode())
        if doc_bytes > MAX_DOCUMENT_BYTES:
            raise ConversionError(
                f"{record_id}: document is {doc_bytes} bytes, over the "
                f"{MAX_DOCUMENT_BYTES} byte limit"
            )


def _open_shard(out_dir: Path, shard: int, use_gzip: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    if use_gzip:
        return gzip.open(out_dir / f"{shard}.jsonl.gz", "wt")
    return open(out_dir / f"{shard}.jsonl", "w")


def parquet_to_jsonl(
    parquet_dir: Path,
    out_dir: Path,
    mapper: DocumentMapper,
    use_gzip: bool = True,
    rows_per_file: int = 25_000,
    strict: bool = True,
    progress: Any = None,
) -> MappingStats:
    """Convert every Parquet file in a namespace directory into sharded JSONL.

    `parquet_dir` is one namespace's export directory; `out_dir` is the matching
    namespace subdirectory under the import prefix. The dense dimension is taken from
    the first row and enforced across the rest, so this step needs no API access.

    Existing JSONL in `out_dir` is deleted first: a re-run after deletions produces
    fewer shards, and leaving the old ones behind would reload documents the source
    no longer has.
    """
    stats = MappingStats()
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise ConversionError(f"no .parquet files found in {parquet_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in [*out_dir.glob("*.jsonl"), *out_dir.glob("*.jsonl.gz")]:
        stale.unlink()

    shard = 0
    rows_in_shard = 0
    handle = _open_shard(out_dir, shard, use_gzip)
    try:
        for path in files:
            table = pq.read_table(path)
            for row in table.to_pylist():
                stats.rows += 1
                metadata = row.get("metadata")
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                try:
                    doc = mapper.from_record(row["id"], row.get("values"), metadata)
                except MissingTextError:
                    if strict:
                        raise
                    stats.skipped_missing_text += 1
                    continue
                except ConversionError as exc:
                    if strict:
                        raise
                    stats.record_error(str(exc))
                    continue
                if doc is None:
                    stats.skipped_missing_text += 1
                    continue

                handle.write(json.dumps(doc) + "\n")
                stats.converted += 1
                rows_in_shard += 1
                if progress is not None:
                    progress.update(1)
                if rows_in_shard >= rows_per_file:
                    handle.close()
                    shard += 1
                    rows_in_shard = 0
                    handle = _open_shard(out_dir, shard, use_gzip)
    finally:
        handle.close()

    if rows_in_shard == 0 and shard > 0:
        (out_dir / (f"{shard}.jsonl.gz" if use_gzip else f"{shard}.jsonl")).unlink()

    return stats


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield documents from a .jsonl or .jsonl.gz file."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_jsonl_dir(directory: Path) -> Iterator[dict[str, Any]]:
    """Yield documents from every JSONL file in a namespace directory, in shard order."""
    files = sorted(
        [*directory.glob("*.jsonl"), *directory.glob("*.jsonl.gz")],
        key=lambda p: int(p.name.split(".")[0]),
    )
    for path in files:
        yield from iter_jsonl(path)

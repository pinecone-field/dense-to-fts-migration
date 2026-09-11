"""Read-only access to the dense index that is still serving production traffic.

Every function here reads. Nothing in this module writes to, reconfigures, or deletes
the source index, which is the safety property the rest of the migration depends on.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pinecone import Pinecone

from .config import Settings

MAX_IDS_PER_FETCH = 100
"""Ids per vector-API fetch.

The vector API's fetch is a GET with the ids in the query string, so a chunk near the
documented 1000-id ceiling is rejected with 431 Request Header Fields Too Large. The
documents API takes its ids in a POST body and has no such ceiling."""


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dimension: int
    metric: str
    host: str


def connect(settings: Settings) -> Pinecone:
    return Pinecone(api_key=settings.api_key)


def describe_source(pc: Pinecone, name: str) -> SourceSpec:
    """Read the source index's dimension and metric so the target can match them."""
    model = pc.describe_index(name)
    dimension = getattr(model, "dimension", None)
    metric = getattr(model, "metric", None)
    if not dimension:
        raise RuntimeError(
            f"index {name!r} reports no dimension. This toolkit migrates dense indexes; "
            f"an index with a document schema is already on the documents API."
        )
    return SourceSpec(
        name=name,
        dimension=int(dimension),
        metric=str(metric),
        host=str(getattr(model, "host", "")),
    )


def open_index(pc: Pinecone, name: str) -> Any:
    return pc.Index(name=name)


def _field(record: Any, name: str) -> Any:
    """Read a field from either a dict-shaped or object-shaped API response.

    `dict.values` is a bound method, so a bare getattr on a dict silently returns
    something truthy and useless.
    """
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


DEFAULT_NAMESPACE_KEYS = ("__default__", "")


def record_count(index: Any, namespace: str) -> int:
    """Count records in a namespace.

    describe_index_stats has spelled the default namespace both ways — the SDK
    documents it as the empty string, the API docs as "__default__" — so try both
    rather than silently report zero.
    """
    stats = index.describe_index_stats()
    namespaces = _field(stats, "namespaces") or {}
    candidates = (
        DEFAULT_NAMESPACE_KEYS if namespace in DEFAULT_NAMESPACE_KEYS else (namespace,)
    )
    for key in candidates:
        entry = namespaces.get(key)
        if entry is not None:
            return int(_field(entry, "vector_count") or 0)
    return 0


def iter_ids(index: Any, namespace: str, limit: int | None = None) -> Iterator[str]:
    """Walk every record id in a namespace, cheapest way to enumerate the source."""
    seen = 0
    for page in index.list(namespace=namespace):
        ids = [getattr(item, "id", item) for item in page]
        for record_id in ids:
            yield record_id
            seen += 1
            if limit is not None and seen >= limit:
                return


def fetch_records(index: Any, ids: Sequence[str], namespace: str) -> dict[str, dict[str, Any]]:
    """Fetch full records by id, in chunks the API accepts."""
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), MAX_IDS_PER_FETCH):
        chunk = list(ids[start : start + MAX_IDS_PER_FETCH])
        response = index.fetch(ids=chunk, namespace=namespace)
        vectors = (
            response.get("vectors", {})
            if isinstance(response, Mapping)
            else getattr(response, "vectors", {})
        )
        for record_id, vector in vectors.items():
            out[record_id] = {
                "id": record_id,
                "values": list(_field(vector, "values") or []),
                "metadata": dict(_field(vector, "metadata") or {}),
            }
    return out


def iter_records(
    index: Any, namespace: str, batch_size: int = MAX_IDS_PER_FETCH, limit: int | None = None
) -> Iterator[dict[str, Any]]:
    """Stream every record in a namespace as {id, values, metadata}."""
    batch: list[str] = []
    for record_id in iter_ids(index, namespace, limit=limit):
        batch.append(record_id)
        if len(batch) >= batch_size:
            yield from fetch_records(index, batch, namespace).values()
            batch = []
    if batch:
        yield from fetch_records(index, batch, namespace).values()

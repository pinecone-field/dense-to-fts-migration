"""Create and talk to the target index — the one with the document schema.

The schema is fixed at index creation: fields cannot be added, removed or retyped
afterwards. Getting `build_schema` right is therefore the one irreversible decision
in this migration, which is why the dense field's dimension and metric are copied
from the source index rather than configured by hand.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from pinecone import Pinecone, SchemaBuilder

from .config import Settings
from .dense_source import SourceSpec

MAX_DOCS_PER_UPSERT = 1000
MAX_IDS_PER_FETCH = 1000
READY_POLL_SECONDS = 3
READY_TIMEOUT_SECONDS = 600


class TargetError(RuntimeError):
    """The target index is not in a state the migration can proceed from."""


def build_schema(settings: Settings, source: SourceSpec) -> dict[str, Any]:
    """Declare the dense field plus one string field per configured text field."""
    builder = SchemaBuilder().add_dense_vector_field(
        settings.target.dense_field,
        dimension=source.dimension,
        metric=source.metric,
    )
    fts_options = dict(settings.target.full_text_search or {})
    for name in settings.target.text_fields:
        builder = builder.add_string_field(name, full_text_search=fts_options or True)
    return builder.build()


def create_index(pc: Pinecone, settings: Settings, source: SourceSpec) -> Any:
    """Create the target index if it does not exist, and wait until it is ready."""
    name = settings.target.index
    if pc.indexes.exists(name):
        return pc.indexes.describe(name)

    kwargs: dict[str, Any] = {"name": name, "schema": build_schema(settings, source)}
    if settings.target.deployment:
        kwargs["deployment"] = {"deployment_type": "managed", **settings.target.deployment}
    if settings.target.read_capacity:
        kwargs["read_capacity"] = dict(settings.target.read_capacity)
    pc.indexes.create(**kwargs)
    return wait_until_ready(pc, name)


def wait_until_ready(pc: Pinecone, name: str, timeout: int = READY_TIMEOUT_SECONDS) -> Any:
    """Poll until status.ready, and until dedicated read capacity is also Ready.

    Searching before both are ready can return empty results rather than an error,
    which is the kind of false negative that derails a parity check.
    """
    deadline = time.time() + timeout
    while True:
        model = pc.indexes.describe(name)
        status = getattr(model, "status", None)
        ready = bool(getattr(status, "ready", False) if status is not None else False)
        capacity = getattr(model, "read_capacity", None)
        capacity_status = getattr(capacity, "status", None) if capacity is not None else None
        capacity_state = getattr(capacity_status, "state", "Ready") if capacity_status else "Ready"
        if ready and capacity_state == "Ready":
            return model
        if time.time() > deadline:
            raise TargetError(f"index {name!r} was not ready within {timeout}s")
        time.sleep(READY_POLL_SECONDS)


def open_index(pc: Pinecone, settings: Settings) -> Any:
    return pc.Index(name=settings.target.index)


def namespace_exists(index: Any, namespace: str) -> bool:
    try:
        index.describe_namespace(name=namespace)
        return True
    except Exception:
        return False


def assert_namespace_absent(index: Any, namespace: str) -> None:
    """Bulk import only creates namespaces, so an existing one fails the whole job."""
    if namespace_exists(index, namespace):
        raise TargetError(
            f"namespace {namespace!r} already exists in the target index. Bulk import can "
            f"only create new namespaces — delete it (index.delete_namespace) or import "
            f"into a different one before starting."
        )


def record_count(index: Any, namespace: str) -> int:
    try:
        description = index.describe_namespace(name=namespace)
    except Exception:
        return 0
    count = getattr(description, "record_count", None)
    if count is None and isinstance(description, Mapping):
        count = description.get("record_count")
    return int(count or 0)


def upsert_documents(index: Any, namespace: str, documents: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for start in range(0, len(documents), MAX_DOCS_PER_UPSERT):
        batch = list(documents[start : start + MAX_DOCS_PER_UPSERT])
        index.documents.upsert(namespace=namespace, documents=batch)
        total += len(batch)
    return total


def fetch_documents(
    index: Any, namespace: str, ids: Sequence[str], include_fields: Sequence[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Fetch documents by id. Missing ids are simply absent from the result."""
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), MAX_IDS_PER_FETCH):
        chunk = list(ids[start : start + MAX_IDS_PER_FETCH])
        response = index.documents.fetch(
            namespace=namespace, ids=chunk, include_fields=list(include_fields or []) or None
        )
        documents = getattr(response, "documents", None)
        if documents is None and isinstance(response, Mapping):
            documents = response.get("documents", {})
        for doc_id, doc in (documents or {}).items():
            out[str(doc_id)] = as_dict(doc)
    return out


def iter_document_ids(index: Any, namespace: str, limit: int | None = None) -> Iterator[str]:
    """Walk every document id in the target namespace."""
    seen = 0
    for entry in index.documents.list(namespace=namespace):
        yield str(getattr(entry, "id", entry))
        seen += 1
        if limit is not None and seen >= limit:
            return


def as_dict(doc: Any) -> dict[str, Any]:
    if isinstance(doc, Mapping):
        return dict(doc)
    to_dict = getattr(doc, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    data = getattr(doc, "_data", None)
    if isinstance(data, Mapping):
        return dict(data)
    return {"_id": getattr(doc, "id", None)}

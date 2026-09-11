"""Load the converted JSONL into the target index.

Two paths, same documents. `bulk import` reads the files from object storage and is
how a real migration loads at scale; `upsert` streams the same JSONL through the
documents API and needs no bucket, which is what makes the demo runnable anywhere.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .convert import iter_jsonl_dir
from .retry import with_retry
from .target_index import MAX_DOCS_PER_UPSERT, fetch_documents

IMPORT_POLL_SECONDS = 20
TERMINAL_STATES = {"Completed", "Failed", "Cancelled"}
MAX_UPSERT_BYTES = 2 * 1024 * 1024

DEFAULT_BATCH_DOCS = 200
DEFAULT_BATCH_BYTES = 1024 * 1024
"""Default load batch, deliberately under the 1,000-document / 2 MB API ceiling.

The ceiling is what the service accepts, not what uploads reliably. A 2 MB request body
is the first thing to time out on a slow or congested link, and a migration is exactly
when you don't want to restart a load."""


class BulkImportError(RuntimeError):
    """The load did not complete."""


@dataclass
class LoadResult:
    mode: str
    documents: int
    import_id: str | None = None
    status: str | None = None
    records_imported: int | None = None

    def summary(self) -> str:
        if self.mode == "import":
            return (
                f"import {self.import_id}: {self.status}, "
                f"{self.records_imported} records imported"
            )
        return f"upsert: {self.documents} documents"


def parse_uri(uri: str) -> tuple[str, str, str]:
    """Split a storage URI into (scheme, bucket_or_container, prefix)."""
    parsed = urlparse(uri)
    if parsed.scheme in ("s3", "gs"):
        return parsed.scheme, parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme == "https":
        container, _, prefix = parsed.path.lstrip("/").partition("/")
        return "azure", container, prefix
    raise BulkImportError(
        f"unsupported storage URI {uri!r}. Use s3://BUCKET/PREFIX, gs://BUCKET/PREFIX, "
        f"or https://ACCOUNT.blob.core.windows.net/CONTAINER/PREFIX."
    )


def upload_tree(local_dir: Path, uri: str, namespace: str) -> str:
    """Upload one namespace's JSONL files to <uri>/<namespace>/ and return that prefix."""
    scheme, bucket, prefix = parse_uri(uri)
    if scheme != "s3":
        raise BulkImportError(
            f"automatic upload is implemented for Amazon S3 only. Copy {local_dir} to "
            f"{uri.rstrip('/')}/{namespace}/ with your provider's CLI "
            f"(gcloud storage cp / az storage blob upload-batch), then rerun with --skip-upload."
        )
    import boto3

    client = boto3.client("s3")
    destination = f"{prefix.rstrip('/')}/{namespace}" if prefix else namespace
    for path in sorted(local_dir.glob("*.jsonl*")):
        client.upload_file(str(path), bucket, f"{destination}/{path.name}")
    return f"s3://{bucket}/{destination}"


def start_import(
    index: Any, uri: str, integration_id: str = "", error_mode: str = "continue"
) -> str:
    """Kick off a bulk import over the whole dataset prefix, not one namespace."""
    kwargs: dict[str, Any] = {"error_mode": error_mode}
    if integration_id:
        kwargs["integration_id"] = integration_id
    response = index.start_import(uri, **kwargs)
    import_id = getattr(response, "id", None)
    if import_id is None and isinstance(response, Mapping):
        import_id = response.get("id")
    if import_id is None:
        raise BulkImportError(f"start_import returned no id: {response!r}")
    return str(import_id)


def describe(index: Any, import_id: str) -> dict[str, Any]:
    model = index.describe_import(import_id)
    if isinstance(model, Mapping):
        return dict(model)
    return {
        "id": getattr(model, "id", import_id),
        "status": getattr(model, "status", None),
        "percent_complete": getattr(model, "percent_complete", None),
        "records_imported": getattr(model, "records_imported", None),
        "error": getattr(model, "error", None),
    }


def wait_for_import(
    index: Any, import_id: str, poll_seconds: int = IMPORT_POLL_SECONDS, on_poll: Any = None
) -> dict[str, Any]:
    """Poll until the import reaches a terminal state.

    An import takes at least ten minutes, so expect a long wait here. It is not a sign
    that anything is stuck.
    """
    while True:
        state = describe(index, import_id)
        if on_poll is not None:
            on_poll(state)
        status = str(state.get("status"))
        if status in TERMINAL_STATES:
            if status != "Completed":
                raise BulkImportError(
                    f"import {import_id} ended as {status}: {state.get('error')}"
                )
            return state
        time.sleep(poll_seconds)


def _batched(
    docs: Iterator[dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_DOCS,
    max_bytes: int = DEFAULT_BATCH_BYTES,
) -> Iterator[list[dict[str, Any]]]:
    batch_size = min(batch_size, MAX_DOCS_PER_UPSERT)
    max_bytes = min(max_bytes, MAX_UPSERT_BYTES)
    batch: list[dict[str, Any]] = []
    batch_bytes = 0
    for doc in docs:
        size = len(json.dumps(doc).encode())
        if batch and (len(batch) >= batch_size or batch_bytes + size > max_bytes):
            yield batch
            batch, batch_bytes = [], 0
        batch.append(doc)
        batch_bytes += size
    if batch:
        yield batch


def upsert_from_jsonl(
    index: Any,
    namespace: str,
    jsonl_dir: Path,
    progress: Any = None,
    batch_size: int = DEFAULT_BATCH_DOCS,
) -> int:
    """Load the JSONL through the documents API instead of object storage."""
    total = 0
    for batch in _batched(iter_jsonl_dir(jsonl_dir), batch_size=batch_size):
        with_retry(lambda b=batch: index.documents.upsert(namespace=namespace, documents=b))
        total += len(batch)
        if progress is not None:
            progress.update(len(batch))
    return total


def wait_until_searchable(
    index: Any,
    namespace: str,
    ids: Sequence[str],
    timeout: int = 900,
    poll_seconds: int = 15,
) -> bool:
    """Block until a sample of loaded ids is fetchable.

    Documents are indexed asynchronously after a load reports complete, so replaying the
    CDC backlog straight away can apply a delete before the document it removes has
    landed. Waiting here keeps replay ordered behind the load.
    """
    if not ids:
        return True
    deadline = time.time() + timeout
    sample = list(ids)[:100]
    while time.time() < deadline:
        found = fetch_documents(index, namespace, sample, include_fields=["_id"])
        if len(found) >= len(sample):
            return True
        time.sleep(poll_seconds)
    return False

"""Stand in for the two things a real migration brings with it.

A dense index that is being read from and written to the whole time, and a Parquet
export of it. Both are simulated here so the rest of the toolkit can be exercised end to end
before it is pointed at production. A real migration skips `seed_index` and `Workload`
entirely and replaces `export_namespace` with the Parquet files Pinecone Support
delivers from a backup export.
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pinecone import Pinecone, ServerlessSpec

from .config import Settings
from .dense_source import iter_records
from .retry import with_retry

TOPICS = {
    "databases": "vector database index namespace shard replica query latency recall",
    "search": "keyword lexical bm25 relevance ranking token stemming phrase match",
    "ml": "embedding model transformer training inference gradient checkpoint dataset",
    "infra": "kubernetes cluster autoscaling deployment rollout observability alerting",
    "finance": "invoice ledger reconciliation settlement clearing counterparty exposure",
}
CATEGORIES = list(TOPICS)
MAX_WORKLOAD_ERRORS = 5
SEED_BATCH_SIZE = 100
"""Records per seeding request.

Well under the 1,000-record / 2 MB upsert ceiling: at a few hundred dimensions a
larger batch is a megabyte-plus request body, which is slow to upload and the first
thing to time out on a poor connection."""


@dataclass
class WorkloadStats:
    upserts: int = 0
    updates: int = 0
    patches: int = 0
    deletes: int = 0
    queries: int = 0
    failures: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def writes(self) -> int:
        return self.upserts + self.updates + self.patches + self.deletes

    def summary(self) -> str:
        line = (
            f"{self.upserts} inserts, {self.updates} updates, {self.patches} patches, "
            f"{self.deletes} deletes, {self.queries} queries"
        )
        if self.failures:
            line += f", {self.failures} FAILED ({self.errors[0]})"
        return line


def _random_text(rng: random.Random, topic: str, words: int = 40) -> str:
    vocabulary = TOPICS[topic].split()
    filler = "the a of to and for with when where how is are was were on in".split()
    tokens = [rng.choice(vocabulary if rng.random() < 0.55 else filler) for _ in range(words)]
    return " ".join(tokens).capitalize() + "."


def _random_vector(rng: np.random.Generator, dimension: int) -> list[float]:
    vector = rng.normal(size=dimension)
    return (vector / np.linalg.norm(vector)).astype(float).tolist()


def make_record(
    record_id: str, dimension: int, rng: random.Random, vector_rng: np.random.Generator
) -> dict[str, Any]:
    topic = rng.choice(CATEGORIES)
    return {
        "id": record_id,
        "values": _random_vector(vector_rng, dimension),
        "metadata": {
            "text": _random_text(rng, topic),
            "title": f"{topic.title()} note {record_id.split('-')[-1]}",
            "category": topic,
            "year": rng.randint(2019, 2026),
        },
    }


def ensure_demo_index(pc: Pinecone, settings: Settings) -> None:
    """Create the throwaway dense index the demo migrates away from."""
    name = settings.source.index
    if pc.has_index(name):
        return
    pc.create_index(
        name=name,
        dimension=settings.demo.dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.target.deployment.get("cloud", "aws"),
            region=settings.target.deployment.get("region", "us-east-1"),
        ),
    )
    while not pc.describe_index(name).status.ready:
        time.sleep(2)


def seed_index(index: Any, settings: Settings, count: int, seed: int = 7) -> int:
    """Fill the source index with synthetic records that carry text in metadata."""
    rng = random.Random(seed)
    vector_rng = np.random.default_rng(seed)
    written = 0
    batch: list[dict[str, Any]] = []
    for i in range(count):
        batch.append(make_record(f"doc-{i:07d}", settings.demo.dimension, rng, vector_rng))
        if len(batch) >= SEED_BATCH_SIZE:
            with_retry(
                lambda b=batch: index.upsert(vectors=b, namespace=settings.source.namespace)
            )
            written += len(batch)
            batch = []
    if batch:
        with_retry(lambda b=batch: index.upsert(vectors=b, namespace=settings.source.namespace))
        written += len(batch)
    return written


class Workload:
    """Background reads and writes against the source index, as production would.

    Writes go through `CdcWrappedIndex` when one is supplied, which is the whole point:
    it proves the capture path sees the traffic that lands during the migration.
    """

    def __init__(
        self,
        index: Any,
        settings: Settings,
        rate_per_second: float = 5.0,
        seed: int = 11,
        start_serial: int = 1_000_000,
    ) -> None:
        self.index = index
        self.settings = settings
        self.interval = 1.0 / rate_per_second if rate_per_second > 0 else 0.2
        self.rng = random.Random(seed)
        self.vector_rng = np.random.default_rng(seed)
        self.stats = WorkloadStats()
        self.live_ids: list[str] = []
        self._serial = start_serial
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def prime(self, existing_ids: list[str]) -> None:
        self.live_ids = list(existing_ids)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> WorkloadStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        return self.stats

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:
                if len(self.stats.errors) < MAX_WORKLOAD_ERRORS:
                    self.stats.errors.append(f"{type(exc).__name__}: {exc}")
                self.stats.failures += 1
            self._stop.wait(self.interval)

    def _tick(self) -> None:
        roll = self.rng.random()
        namespace = self.settings.source.namespace
        if roll < 0.55 or not self.live_ids:
            self._serial += 1
            record = make_record(
                f"doc-{self._serial:07d}", self.settings.demo.dimension, self.rng, self.vector_rng
            )
            self.index.upsert(vectors=[record], namespace=namespace)
            self.live_ids.append(record["id"])
            self.stats.upserts += 1
        elif roll < 0.8:
            record_id = self.rng.choice(self.live_ids)
            record = make_record(
                record_id, self.settings.demo.dimension, self.rng, self.vector_rng
            )
            record["metadata"]["title"] = record["metadata"]["title"] + " (revised)"
            self.index.upsert(vectors=[record], namespace=namespace)
            self.stats.updates += 1
        elif roll < 0.87:
            record_id = self.rng.choice(self.live_ids)
            self.index.update(
                id=record_id,
                set_metadata={"title": f"Patched note {self._serial}"},
                namespace=namespace,
            )
            self.stats.patches += 1
        elif roll < 0.93:
            record_id = self.live_ids.pop(self.rng.randrange(len(self.live_ids)))
            self.index.delete(ids=[record_id], namespace=namespace)
            self.stats.deletes += 1
        else:
            self.index.query(
                vector=_random_vector(self.vector_rng, self.settings.demo.dimension),
                top_k=5,
                namespace=namespace,
            )
            self.stats.queries += 1


EXPORT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("values", pa.list_(pa.float32())),
        pa.field("metadata", pa.string()),
    ]
)


def export_namespace(
    index: Any,
    namespace: str,
    out_dir: Path,
    rows_per_file: int = 25_000,
    include_text: bool = True,
    text_key: str = "text",
    snapshot_seq: int = 0,
    source_index: str = "",
) -> dict[str, Any]:
    """Write the namespace to Parquet in the format a backup export produces.

    `include_text=False` reproduces the export shape teams hit most often: vectors and
    metadata, but the searchable text left behind in the system of record.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.parquet"):
        stale.unlink()

    rows: list[dict[str, Any]] = []
    files: list[str] = []
    total = 0

    def flush() -> None:
        nonlocal rows
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=EXPORT_SCHEMA)
        path = out_dir / f"{len(files)}.parquet"
        pq.write_table(table, path)
        files.append(path.name)
        rows = []

    for record in iter_records(index, namespace):
        metadata = dict(record["metadata"])
        if not include_text:
            metadata.pop(text_key, None)
        rows.append(
            {
                "id": record["id"],
                "values": record["values"],
                "metadata": json.dumps(metadata),
            }
        )
        total += 1
        if len(rows) >= rows_per_file:
            flush()
    flush()

    manifest = {
        "source_index": source_index,
        "namespace": namespace,
        "rows": total,
        "files": files,
        "include_text": include_text,
        "snapshot_seq": snapshot_seq,
        "exported_at": time.time(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_manifest(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no manifest.json in {out_dir} — run `migrate.py export` first")
    return json.loads(path.read_text())

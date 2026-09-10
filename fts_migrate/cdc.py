"""Change data capture: catch writes to the dense index while the migration runs.

Bulk import can only create namespaces that do not exist yet, so nothing can be
written into the target namespace until the import finishes. Every write that lands
on the source index in the meantime therefore has to be buffered somewhere durable
and replayed afterwards. That buffer is a SQLite log, and `CdcWrappedIndex` is the
drop-in the application writes through to fill it.

The log stores the whole record, not just its id, so replay never has to read back
from the source index — which matters because by replay time the source may already
have moved on.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .convert import DocumentMapper

UPSERT = "upsert"
UPDATE = "update"
DELETE = "delete"

MAX_DOCS_PER_UPSERT = 1000
MAX_IDS_PER_DELETE = 1000
MAX_UPSERT_BYTES = 2 * 1024 * 1024
SQLITE_PARAM_CHUNK = 500

UNCAPTURABLE_METHODS = frozenset(
    {"upsert_from_dataframe", "upsert_records", "delete_namespace", "start_import"}
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS changes (
  seq       INTEGER PRIMARY KEY AUTOINCREMENT,
  ts        REAL    NOT NULL,
  namespace TEXT    NOT NULL,
  doc_id    TEXT    NOT NULL,
  op        TEXT    NOT NULL CHECK (op IN ('upsert', 'update', 'delete')),
  payload   TEXT
);
CREATE INDEX IF NOT EXISTS changes_ns_id ON changes (namespace, doc_id);
CREATE TABLE IF NOT EXISTS cursor (
  name TEXT PRIMARY KEY,
  seq  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS unapplied (
  seq       INTEGER PRIMARY KEY,
  ts        REAL    NOT NULL,
  namespace TEXT    NOT NULL,
  doc_id    TEXT    NOT NULL,
  reason    TEXT    NOT NULL
);
"""


class CdcError(RuntimeError):
    """A change could not be captured or applied."""


@dataclass(frozen=True)
class Change:
    seq: int
    ts: float
    namespace: str
    doc_id: str
    op: str
    record: dict[str, Any] | None


@dataclass(frozen=True)
class Action:
    """The single write that brings one document up to date."""

    op: str
    doc_id: str
    seq: int = 0
    record: dict[str, Any] | None = None
    values: list[float] | None = None
    set_metadata: dict[str, Any] | None = None


@dataclass
class ApplyStats:
    upserted: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    through_seq: int = 0

    def summary(self) -> str:
        line = (
            f"{self.upserted} upserted, {self.updated} patched, {self.deleted} deleted, "
            f"cursor at seq {self.through_seq}"
        )
        if self.skipped:
            line += (
                f" | {self.skipped} PARKED as unapplied — those documents are stale on the "
                f"target until the text is joined in (see `migrate.py status`)"
            )
        return line


@dataclass(frozen=True)
class Lag:
    pending: int
    seconds: float
    cursor_seq: int
    head_seq: int


def _normalize_record(item: Any) -> dict[str, Any]:
    """Accept every shape `Index.upsert` accepts and return {id, values, metadata}."""
    if isinstance(item, Mapping):
        record = {
            "id": item.get("id"),
            "values": list(item.get("values") or []),
            "metadata": dict(item.get("metadata") or {}),
        }
    elif isinstance(item, tuple):
        record = {
            "id": item[0],
            "values": list(item[1]),
            "metadata": dict(item[2]) if len(item) > 2 and item[2] else {},
        }
    else:
        record = {
            "id": getattr(item, "id", None),
            "values": list(getattr(item, "values", []) or []),
            "metadata": dict(getattr(item, "metadata", None) or {}),
        }
    if not record["id"]:
        raise CdcError(f"cannot capture a record with no id: {item!r}")
    return record


class CdcLog:
    """A durable, ordered log of the writes applied to the source index.

    Safe to share across threads: an application writes to its index from whatever
    thread handles the request, and sqlite3 refuses a connection used off the thread
    that opened it unless told otherwise. The lock serialises access to the single
    connection that `check_same_thread=False` then allows.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> CdcLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def append_upserts(self, namespace: str, records: Iterable[Mapping[str, Any]]) -> int:
        rows = [
            (time.time(), namespace, r["id"], UPSERT, json.dumps(r))
            for r in (_normalize_record(item) for item in records)
        ]
        return self._append(rows)

    def append_update(
        self,
        namespace: str,
        doc_id: str,
        values: Sequence[float] | None = None,
        set_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        payload = {
            "id": doc_id,
            "values": list(values) if values is not None else None,
            "set_metadata": dict(set_metadata or {}),
        }
        return self._append([(time.time(), namespace, doc_id, UPDATE, json.dumps(payload))])

    def append_deletes(self, namespace: str, ids: Iterable[str]) -> int:
        rows = [(time.time(), namespace, doc_id, DELETE, None) for doc_id in ids]
        return self._append(rows)

    def _append(self, rows: Sequence[tuple[Any, ...]]) -> int:
        if not rows:
            return 0
        with self._lock:
            self.conn.executemany(
                "INSERT INTO changes (ts, namespace, doc_id, op, payload) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def record_unapplied(self, change_seq: int, namespace: str, doc_id: str, reason: str) -> None:
        """Park a change replay could not apply.

        The cursor still advances past it, so without this row the change would be
        counted once and then be unreachable — the log has no way to re-drive it.
        """
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO unapplied (seq, ts, namespace, doc_id, reason) "
                "VALUES (?, ?, ?, ?, ?)",
                (change_seq, time.time(), namespace, doc_id, reason),
            )

    def unapplied(self, limit: int = 100) -> list[sqlite3.Row]:
        return self._query(
            "SELECT seq, ts, namespace, doc_id, reason FROM unapplied ORDER BY seq LIMIT ?",
            (limit,),
        )

    def unapplied_count(self) -> int:
        return int(self._query("SELECT COUNT(*) AS n FROM unapplied")[0]["n"])

    def clear_unapplied(self, seqs: Sequence[int]) -> None:
        with self._lock:
            self.conn.executemany("DELETE FROM unapplied WHERE seq = ?", [(s,) for s in seqs])

    def head_seq(self) -> int:
        row = self._query("SELECT COALESCE(MAX(seq), 0) AS seq FROM changes")[0]
        return int(row["seq"])

    def get_cursor(self, name: str) -> int:
        rows = self._query("SELECT seq FROM cursor WHERE name = ?", (name,))
        return int(rows[0]["seq"]) if rows else 0

    def set_cursor(self, name: str, seq: int) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO cursor (name, seq) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET seq = excluded.seq",
                (name, seq),
            )

    def lag(self, cursor_name: str) -> Lag:
        cursor_seq = self.get_cursor(cursor_name)
        head = self.head_seq()
        row = self._query(
            "SELECT COUNT(*) AS n, COALESCE(MIN(ts), 0) AS oldest FROM changes WHERE seq > ?",
            (cursor_seq,),
        )[0]
        pending = int(row["n"])
        seconds = time.time() - float(row["oldest"]) if pending else 0.0
        return Lag(pending=pending, seconds=seconds, cursor_seq=cursor_seq, head_seq=head)

    def collapse(
        self, since_seq: int, until_seq: int, namespace: str | None = None
    ) -> list[Change]:
        """Return the last change per document in (since_seq, until_seq].

        Replaying only the final state of each document is safe because
        `documents.upsert` replaces the whole document — there is no partial state to
        rebuild by walking every intermediate write.
        """
        params: list[Any] = [since_seq, until_seq]
        namespace_clause = ""
        if namespace is not None:
            namespace_clause = " AND namespace = ?"
            params.append(namespace)
        rows = self._query(
            f"""
            SELECT c.seq, c.ts, c.namespace, c.doc_id, c.op, c.payload
            FROM changes c
            JOIN (
              SELECT namespace, doc_id, MAX(seq) AS seq
              FROM changes
              WHERE seq > ? AND seq <= ?{namespace_clause}
              GROUP BY namespace, doc_id
            ) latest ON latest.seq = c.seq
            ORDER BY c.seq
            """,
            params,
        )
        return [_row_to_change(r) for r in rows]

    def _history(
        self, since_seq: int, until_seq: int, doc_ids: Sequence[str], namespace: str | None
    ) -> dict[str, list[Change]]:
        history: dict[str, list[Change]] = {}
        for start in range(0, len(doc_ids), SQLITE_PARAM_CHUNK):
            chunk = list(doc_ids[start : start + SQLITE_PARAM_CHUNK])
            placeholders = ", ".join("?" for _ in chunk)
            params: list[Any] = [since_seq, until_seq, *chunk]
            namespace_clause = ""
            if namespace is not None:
                namespace_clause = " AND namespace = ?"
                params.append(namespace)
            rows = self._query(
                f"""
                SELECT seq, ts, namespace, doc_id, op, payload
                FROM changes
                WHERE seq > ? AND seq <= ? AND doc_id IN ({placeholders}){namespace_clause}
                ORDER BY seq
                """,
                params,
            )
            for row in rows:
                history.setdefault(row["doc_id"], []).append(_row_to_change(row))
        return history

    def pending_actions(
        self, since_seq: int, until_seq: int, namespace: str | None = None
    ) -> list[Action]:
        """Reduce the window to one write per document.

        An upsert or a delete stands on its own — it says everything about the document.
        A partial update does not, so any document whose last change is an update is
        replayed from its full history in the window: an update that follows an upsert
        folds into that upsert, and only an update with no upsert behind it stays a patch.
        """
        latest = self.collapse(since_seq, until_seq, namespace)
        patched = [c.doc_id for c in latest if c.op == UPDATE]
        history = self._history(since_seq, until_seq, patched, namespace) if patched else {}

        actions: list[Action] = []
        for change in latest:
            if change.op == DELETE:
                actions.append(Action(op=DELETE, doc_id=change.doc_id, seq=change.seq))
            elif change.op == UPSERT:
                actions.append(
                    Action(op=UPSERT, doc_id=change.doc_id, seq=change.seq, record=change.record)
                )
            else:
                actions.append(_fold(change.doc_id, history.get(change.doc_id, []), change.seq))
        return actions


def _row_to_change(row: Any) -> Change:
    return Change(
        seq=int(row["seq"]),
        ts=float(row["ts"]),
        namespace=row["namespace"],
        doc_id=row["doc_id"],
        op=row["op"],
        record=json.loads(row["payload"]) if row["payload"] else None,
    )


def _fold(doc_id: str, changes: Sequence[Change], seq: int) -> Action:
    record: dict[str, Any] | None = None
    values: list[float] | None = None
    set_metadata: dict[str, Any] = {}

    for change in changes:
        if change.op == UPSERT:
            record = dict(change.record or {})
            values, set_metadata = None, {}
        elif change.op == DELETE:
            record, values, set_metadata = None, None, {}
        else:
            payload = change.record or {}
            if record is not None:
                if payload.get("values") is not None:
                    record["values"] = payload["values"]
                record.setdefault("metadata", {}).update(payload.get("set_metadata") or {})
            else:
                if payload.get("values") is not None:
                    values = payload["values"]
                set_metadata.update(payload.get("set_metadata") or {})

    if record is not None:
        return Action(op=UPSERT, doc_id=doc_id, seq=seq, record=record)
    return Action(op=UPDATE, doc_id=doc_id, seq=seq, values=values, set_metadata=set_metadata)


class CdcWrappedIndex:
    """Wraps the source dense index so every write is also written to the CDC log.

    Point the application at this instead of `pc.Index(...)` before taking the export.
    The dense index is written first and the log second: a failed log append raises,
    because a silently dropped change is the one failure this migration cannot survive.
    """

    def __init__(self, index: Any, log: CdcLog, namespace: str = "__default__") -> None:
        self._index = index
        self._log = log
        self._namespace = namespace

    def __getattr__(self, name: str) -> Any:
        if name in UNCAPTURABLE_METHODS:
            raise CdcError(
                f"{name}() writes to the source index without going through the capture "
                f"path, so those changes would be missing from the target at cutover. "
                f"Use upsert()/update()/delete() on this wrapper for the duration of the "
                f"migration."
            )
        return getattr(self._index, name)

    def upsert(self, vectors: Sequence[Any], namespace: str | None = None, **kwargs: Any) -> Any:
        namespace = namespace or self._namespace
        if "batch_size" in kwargs:
            raise CdcError(
                "batch_size splits one call into several requests, and a failure part-way "
                "through commits the earlier batches to the source index while this call "
                "raises before anything is logged. Batch in your own code instead, so each "
                "call is one request that is either captured or not made."
            )
        result = self._index.upsert(vectors=vectors, namespace=namespace, **kwargs)
        self._log.append_upserts(namespace, vectors)
        return result

    def update(
        self,
        id: str | None = None,
        values: Sequence[float] | None = None,
        set_metadata: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        namespace = namespace or self._namespace
        if filter or not id:
            raise CdcError(
                "filtered updates cannot be captured as a per-document patch. During the "
                "migration, resolve the filter to ids first and update by id, so the same "
                "change can be replayed against the target index."
            )
        result = self._index.update(
            id=id, values=values, set_metadata=set_metadata, namespace=namespace, **kwargs
        )
        if kwargs.get("dry_run"):
            return result
        self._log.append_update(namespace, id, values=values, set_metadata=set_metadata)
        return result

    def delete(
        self,
        ids: Sequence[str] | None = None,
        namespace: str | None = None,
        delete_all: bool = False,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        namespace = namespace or self._namespace
        if delete_all or filter:
            raise CdcError(
                "delete_all and filtered deletes cannot be captured as individual ids. "
                "During the migration, resolve the filter to ids first (index.list or "
                "index.query) and delete by id, so the same deletes can be replayed "
                "against the target index."
            )
        if not ids:
            return None
        result = self._index.delete(ids=list(ids), namespace=namespace, **kwargs)
        self._log.append_deletes(namespace, ids)
        return result


def _batched_documents(
    docs: Sequence[Mapping[str, Any]],
) -> Iterator[list[Mapping[str, Any]]]:
    batch: list[Mapping[str, Any]] = []
    batch_bytes = 0
    for doc in docs:
        size = len(json.dumps(doc).encode())
        if batch and (len(batch) >= MAX_DOCS_PER_UPSERT or batch_bytes + size > MAX_UPSERT_BYTES):
            yield batch
            batch, batch_bytes = [], 0
        batch.append(doc)
        batch_bytes += size
    if batch:
        yield batch


def apply_changes(
    index: Any,
    log: CdcLog,
    mapper: DocumentMapper,
    namespace: str,
    cursor_name: str = "target",
    source_namespace: str | None = None,
    dry_run: bool = False,
) -> ApplyStats:
    """Replay the log onto the target index, from the cursor to the current head.

    Idempotent: upserts replace the whole document and deletes of absent ids are
    no-ops, so a re-run applies the same end state.
    """
    since = log.get_cursor(cursor_name)
    head = log.head_seq()
    stats = ApplyStats(through_seq=since)
    if head <= since:
        return stats

    actions = log.pending_actions(since, head, namespace=source_namespace)
    to_upsert: list[dict[str, Any]] = []
    to_patch: list[dict[str, Any]] = []
    to_delete: list[str] = []
    for action in actions:
        if action.op == DELETE:
            to_delete.append(action.doc_id)
        elif action.op == UPDATE:
            to_patch.append(
                mapper.from_partial(action.doc_id, action.values, action.set_metadata)
            )
        else:
            record = action.record or {}
            doc = mapper.from_record(record["id"], record.get("values"), record.get("metadata"))
            if doc is None:
                stats.skipped += 1
                if not dry_run:
                    log.record_unapplied(
                        action.seq,
                        source_namespace or namespace,
                        action.doc_id,
                        "no text for the full-text field",
                    )
                continue
            to_upsert.append(doc)

    if not dry_run:
        for batch in _batched_documents(to_upsert):
            index.documents.upsert(namespace=namespace, documents=batch)
        for batch in _batched_documents(to_patch):
            index.documents.update(namespace=namespace, documents=batch)
        for start in range(0, len(to_delete), MAX_IDS_PER_DELETE):
            index.documents.delete(
                namespace=namespace, ids=to_delete[start : start + MAX_IDS_PER_DELETE]
            )
        log.set_cursor(cursor_name, head)

    stats.upserted = len(to_upsert)
    stats.updated = len(to_patch)
    stats.deleted = len(to_delete)
    stats.through_seq = head
    return stats

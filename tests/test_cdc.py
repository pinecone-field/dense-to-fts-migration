from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fts_migrate.cdc import CdcError, CdcLog, CdcWrappedIndex, apply_changes
from fts_migrate.convert import DocumentMapper
from tests.test_convert import make_settings


class FakeVectorIndex:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def upsert(self, vectors, namespace="__default__", **kwargs):
        for record in vectors:
            self.records[record["id"]] = record
        return {"upserted_count": len(vectors)}

    def update(self, id=None, values=None, set_metadata=None, namespace="__default__", **kwargs):
        record = self.records.setdefault(id, {"id": id, "values": [], "metadata": {}})
        if values is not None:
            record["values"] = values
        record["metadata"].update(set_metadata or {})

    def delete(self, ids=None, namespace="__default__", **kwargs):
        for record_id in ids or []:
            self.records.pop(record_id, None)


class FakeDocuments:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.upsert_calls: list[int] = []
        self.update_calls: list[int] = []

    def upsert(self, *, namespace, documents, **kwargs):
        self.upsert_calls.append(len(documents))
        for doc in documents:
            self.docs[doc["_id"]] = doc

    def update(self, *, namespace, documents, **kwargs):
        self.update_calls.append(len(documents))
        for patch in documents:
            self.docs.setdefault(patch["_id"], {"_id": patch["_id"]}).update(patch)

    def delete(self, *, namespace, ids=None, **kwargs):
        for doc_id in ids or []:
            self.docs.pop(doc_id, None)


class FakeTargetIndex:
    def __init__(self) -> None:
        self.documents = FakeDocuments()


def record(record_id: str, text: str = "body") -> dict[str, Any]:
    return {"id": record_id, "values": [0.1, 0.2, 0.3], "metadata": {"text": text}}


@pytest.fixture()
def log(tmp_path: Path) -> CdcLog:
    with CdcLog(tmp_path / "cdc.sqlite") as instance:
        yield instance


def test_wrapper_writes_through_to_the_index_and_the_log(log: CdcLog):
    index = FakeVectorIndex()
    wrapped = CdcWrappedIndex(index, log)

    wrapped.upsert(vectors=[record("a"), record("b")])
    wrapped.delete(ids=["a"])

    assert set(index.records) == {"b"}
    assert log.head_seq() == 3


def test_wrapper_refuses_uncapturable_deletes(log: CdcLog):
    wrapped = CdcWrappedIndex(FakeVectorIndex(), log)
    with pytest.raises(CdcError, match="cannot be captured"):
        wrapped.delete(delete_all=True)
    with pytest.raises(CdcError, match="cannot be captured"):
        wrapped.delete(filter={"year": 2024})


def test_collapse_keeps_only_the_last_change_per_document(log: CdcLog):
    log.append_upserts("__default__", [record("a", "v1")])
    log.append_upserts("__default__", [record("a", "v2")])
    log.append_upserts("__default__", [record("b", "v1")])
    log.append_deletes("__default__", ["b"])

    changes = log.collapse(0, log.head_seq())

    assert [(c.doc_id, c.op) for c in changes] == [("a", "upsert"), ("b", "delete")]
    assert changes[0].record["metadata"]["text"] == "v2"


def test_replay_applies_the_final_state_and_advances_the_cursor(log: CdcLog):
    target = FakeTargetIndex()
    mapper = DocumentMapper(make_settings(), dimension=3)

    log.append_upserts("__default__", [record("a", "v1"), record("b")])
    log.append_upserts("__default__", [record("a", "v2")])
    log.append_deletes("__default__", ["b"])

    stats = apply_changes(target, log, mapper, namespace="__default__")

    assert target.documents.docs["a"]["text"] == "v2"
    assert "b" not in target.documents.docs
    assert stats.through_seq == log.head_seq()
    assert log.lag("target").pending == 0


def test_replay_is_idempotent(log: CdcLog):
    target = FakeTargetIndex()
    mapper = DocumentMapper(make_settings(), dimension=3)
    log.append_upserts("__default__", [record("a")])

    apply_changes(target, log, mapper, namespace="__default__")
    second = apply_changes(target, log, mapper, namespace="__default__")

    assert second.upserted == 0
    assert target.documents.upsert_calls == [1]


def test_dry_run_leaves_the_cursor_alone(log: CdcLog):
    target = FakeTargetIndex()
    mapper = DocumentMapper(make_settings(), dimension=3)
    log.append_upserts("__default__", [record("a")])

    stats = apply_changes(target, log, mapper, namespace="__default__", dry_run=True)

    assert stats.upserted == 1
    assert not target.documents.docs
    assert log.get_cursor("target") == 0


def test_only_the_configured_source_namespace_is_replayed(log: CdcLog):
    target = FakeTargetIndex()
    mapper = DocumentMapper(make_settings(), dimension=3)
    log.append_upserts("tenant-a", [record("a")])
    log.append_upserts("tenant-b", [record("b")])

    apply_changes(
        target, log, mapper, namespace="__default__", source_namespace="tenant-a"
    )

    assert set(target.documents.docs) == {"a"}


def test_upserts_are_split_into_batches(log: CdcLog):
    target = FakeTargetIndex()
    mapper = DocumentMapper(make_settings(), dimension=3)
    log.append_upserts("__default__", [record(f"doc-{i}") for i in range(500)])

    apply_changes(target, log, mapper, namespace="__default__", batch_size=200)

    assert target.documents.upsert_calls == [200, 200, 100]
    assert len(target.documents.docs) == 500


def test_a_batch_size_over_the_api_ceiling_is_clamped(log: CdcLog):
    """The service caps an upsert at 1,000 documents, whatever the config asks for."""
    target = FakeTargetIndex()
    mapper = DocumentMapper(make_settings(), dimension=3)
    log.append_upserts("__default__", [record(f"doc-{i}") for i in range(2500)])

    apply_changes(target, log, mapper, namespace="__default__", batch_size=5000)

    assert target.documents.upsert_calls == [1000, 1000, 500]
    assert len(target.documents.docs) == 2500


def test_wrapper_captures_partial_updates(log: CdcLog):
    index = FakeVectorIndex()
    wrapped = CdcWrappedIndex(index, log)
    wrapped.upsert(vectors=[record("a", "v1")])

    wrapped.update(id="a", set_metadata={"text": "patched"})

    assert index.records["a"]["metadata"]["text"] == "patched"
    assert log.head_seq() == 2


def test_wrapper_refuses_filtered_updates(log: CdcLog):
    wrapped = CdcWrappedIndex(FakeVectorIndex(), log)
    with pytest.raises(CdcError, match="filtered updates"):
        wrapped.update(id="a", filter={"year": 2024}, set_metadata={"text": "x"})


def test_update_after_an_upsert_folds_into_one_upsert(log: CdcLog):
    target = FakeTargetIndex()
    mapper = DocumentMapper(make_settings(), dimension=3)
    log.append_upserts("__default__", [record("a", "v1")])
    log.append_update("__default__", "a", set_metadata={"text": "patched"})

    stats = apply_changes(target, log, mapper, namespace="__default__")

    assert stats.upserted == 1
    assert stats.updated == 0
    assert target.documents.docs["a"]["text"] == "patched"
    assert target.documents.docs["a"]["embedding"] == [0.1, 0.2, 0.3]


def test_update_with_no_upsert_behind_it_stays_a_patch(log: CdcLog):
    target = FakeTargetIndex()
    mapper = DocumentMapper(make_settings(), dimension=3)
    log.append_update("__default__", "a", set_metadata={"text": "patched"})

    stats = apply_changes(target, log, mapper, namespace="__default__")

    assert stats.updated == 1
    assert target.documents.update_calls == [1]
    assert target.documents.docs["a"] == {"_id": "a", "text": "patched"}


def test_updates_accumulate_before_the_patch_is_sent(log: CdcLog):
    target = FakeTargetIndex()
    mapper = DocumentMapper(make_settings(), dimension=3)
    log.append_update("__default__", "a", set_metadata={"text": "first"})
    log.append_update("__default__", "a", set_metadata={"year": 2026})

    apply_changes(target, log, mapper, namespace="__default__")

    assert target.documents.docs["a"] == {"_id": "a", "text": "first", "year": 2026}


def test_a_delete_after_an_update_still_deletes(log: CdcLog):
    target = FakeTargetIndex()
    mapper = DocumentMapper(make_settings(), dimension=3)
    log.append_upserts("__default__", [record("a")])
    apply_changes(target, log, mapper, namespace="__default__")

    log.append_update("__default__", "a", set_metadata={"text": "patched"})
    log.append_deletes("__default__", ["a"])
    stats = apply_changes(target, log, mapper, namespace="__default__")

    assert stats.deleted == 1
    assert "a" not in target.documents.docs


def test_log_is_writable_from_another_thread(log: CdcLog):
    """sqlite refuses a cross-thread connection by default, and an application writes
    to its index from whatever thread serves the request."""
    import threading

    errors: list[Exception] = []
    wrapped = CdcWrappedIndex(FakeVectorIndex(), log)

    def write():
        try:
            wrapped.upsert(vectors=[record("a")])
            wrapped.update(id="a", set_metadata={"text": "patched"})
            wrapped.delete(ids=["a"])
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=write)
    thread.start()
    thread.join()

    assert not errors, errors
    assert log.head_seq() == 3


def test_concurrent_writers_all_land(log: CdcLog):
    import threading

    wrapped = CdcWrappedIndex(FakeVectorIndex(), log)
    errors: list[Exception] = []

    def write(worker: int):
        try:
            for i in range(20):
                wrapped.upsert(vectors=[record(f"w{worker}-{i}")])
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert log.head_seq() == 80


def test_uncapturable_write_methods_raise_instead_of_passing_through(log: CdcLog):
    class RichIndex(FakeVectorIndex):
        def upsert_from_dataframe(self, *a, **kw):
            self.records["leaked"] = {"id": "leaked"}

        def upsert_records(self, *a, **kw):
            self.records["leaked"] = {"id": "leaked"}

        def delete_namespace(self, *a, **kw):
            self.records.clear()

    index = RichIndex()
    wrapped = CdcWrappedIndex(index, log)
    for method in ("upsert_from_dataframe", "upsert_records", "delete_namespace"):
        with pytest.raises(CdcError, match="without going through the capture path"):
            getattr(wrapped, method)()
    assert index.records == {}


def test_batch_size_is_refused_because_a_partial_failure_would_not_be_logged(log: CdcLog):
    wrapped = CdcWrappedIndex(FakeVectorIndex(), log)
    with pytest.raises(CdcError, match="batch_size"):
        wrapped.upsert(vectors=[record("a")], batch_size=200)


def test_dry_run_on_a_by_id_update_is_refused(log: CdcLog):
    """dry_run previews only update-by-metadata. On a by-id update the write lands, so
    treating it as a no-op would drop the change from the log."""
    index = FakeVectorIndex()
    wrapped = CdcWrappedIndex(index, log)
    with pytest.raises(CdcError, match="dry_run"):
        wrapped.update(id="a", set_metadata={"text": "x"}, dry_run=True)
    assert log.head_seq() == 0
    assert index.records == {}


def test_unappliable_changes_are_parked_rather_than_forgotten(log: CdcLog):
    """The cursor advances past them, so without the parking table they'd be lost."""
    target = FakeTargetIndex()
    settings = make_settings()
    mapper = DocumentMapper(settings, dimension=3, allow_missing_text=True)
    log.append_upserts("__default__", [{"id": "a", "values": [0.1, 0.2, 0.3], "metadata": {}}])

    stats = apply_changes(target, log, mapper, namespace="__default__")

    assert stats.skipped == 1
    assert "a" not in target.documents.docs
    parked = log.unapplied()
    assert [(r["doc_id"], r["reason"]) for r in parked] == [
        ("a", "no text for the full-text field")
    ]
    assert log.unapplied_count() == 1
    assert "PARKED" in stats.summary()


def test_transient_failures_are_retried_but_client_errors_are_not():
    from pinecone.errors import NotFoundError, PineconeTimeoutError

    from fts_migrate.retry import with_retry

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise PineconeTimeoutError("The write operation timed out")
        return "ok"

    assert with_retry(flaky, backoff=0) == "ok"
    assert calls["n"] == 3

    def broken():
        calls["n"] += 1
        raise NotFoundError("index does not exist")

    calls["n"] = 0
    with pytest.raises(NotFoundError):
        with_retry(broken, backoff=0)
    assert calls["n"] == 1


def test_retry_gives_up_and_reraises():
    from pinecone.errors import PineconeTimeoutError

    from fts_migrate.retry import with_retry

    def always_times_out():
        raise PineconeTimeoutError("nope")

    with pytest.raises(PineconeTimeoutError):
        with_retry(always_times_out, attempts=2, backoff=0)

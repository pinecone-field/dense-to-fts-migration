from __future__ import annotations

from typing import Any

from fts_migrate import reconcile
from fts_migrate.cutover import DONE, RAMP, SHADOW, CutoverState, SearchRouter
from tests.test_convert import make_settings


class FakeSourceIndex:
    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self.records = records

    def describe_index_stats(self):
        return {"namespaces": {"__default__": {"vector_count": len(self.records)}}}

    def list(self, namespace="__default__"):
        yield list(self.records)

    def fetch(self, ids, namespace="__default__"):
        return {"vectors": {i: self.records[i] for i in ids if i in self.records}}

    def query(self, vector, top_k=10, namespace="__default__"):
        ids = list(self.records)[:top_k]
        return {"matches": [{"id": i, "score": 1.0} for i in ids]}


class FakeMatch:
    def __init__(self, doc_id: str) -> None:
        self.id = doc_id
        self.score = 1.0


class FakeSearchResponse:
    def __init__(self, ids: list[str]) -> None:
        self.matches = [FakeMatch(i) for i in ids]


class FakeDocuments:
    def __init__(self, docs: dict[str, dict[str, Any]]) -> None:
        self.docs = docs

    def list(self, *, namespace, **kwargs):
        for doc_id in self.docs:
            yield FakeMatch(doc_id)

    def fetch(self, *, namespace, ids=None, include_fields=None, **kwargs):
        return {"documents": {i: self.docs[i] for i in (ids or []) if i in self.docs}}

    def search(self, *, namespace, top_k, score_by, **kwargs):
        return FakeSearchResponse(list(self.docs)[:top_k])


class FakeTargetIndex:
    def __init__(self, docs: dict[str, dict[str, Any]]) -> None:
        self.docs = docs
        self.documents = FakeDocuments(docs)

    def describe_namespace(self, *, name):
        return {"record_count": len(self.docs)}


def source_record(record_id: str, text: str = "body", year: int = 2024) -> dict[str, Any]:
    return {
        "id": record_id,
        "values": [0.1, 0.2, 0.3],
        "metadata": {"text": text, "year": year},
    }


def target_doc(record_id: str, text: str = "body", year: int = 2024) -> dict[str, Any]:
    return {"_id": record_id, "embedding": [0.1, 0.2, 0.3], "text": text, "year": year}


def test_counts_and_ids_agree_when_both_sides_match():
    settings = make_settings()
    source = FakeSourceIndex({f"doc-{i}": source_record(f"doc-{i}") for i in range(5)})
    target = FakeTargetIndex({f"doc-{i}": target_doc(f"doc-{i}") for i in range(5)})

    assert reconcile.compare_counts(source, target, settings).ok
    assert reconcile.diff_ids(source, target, settings).ok


def test_missing_and_orphaned_ids_are_both_reported():
    settings = make_settings()
    source = FakeSourceIndex({"a": source_record("a"), "b": source_record("b")})
    target = FakeTargetIndex({"a": target_doc("a"), "c": target_doc("c")})

    report = reconcile.diff_ids(source, target, settings)

    assert report.missing == ["b"]
    assert report.orphaned == ["c"]
    assert not report.ok


def test_field_diff_flags_text_and_metadata_drift():
    settings = make_settings()
    source = FakeSourceIndex({"a": source_record("a", text="original", year=2024)})
    target = FakeTargetIndex({"a": target_doc("a", text="changed", year=2019)})

    report = reconcile.diff_fields(source, target, settings, ["a"])

    assert report.text_mismatches == ["a"]
    assert report.metadata_mismatches == ["a:year"]
    assert not report.ok


def test_query_parity_is_one_when_both_return_the_same_ids():
    settings = make_settings()
    records = {f"doc-{i}": source_record(f"doc-{i}") for i in range(10)}
    source = FakeSourceIndex(records)
    target = FakeTargetIndex({k: target_doc(k) for k in records})

    parity = reconcile.query_parity(source, target, settings, queries=3, top_k=5)

    assert parity.recall == 1.0
    assert parity.ok


def test_router_shadow_serves_source_and_compares():
    settings = make_settings()
    source = FakeSourceIndex({f"doc-{i}": source_record(f"doc-{i}") for i in range(3)})
    target = FakeTargetIndex({f"doc-{i}": target_doc(f"doc-{i}") for i in range(3)})
    router = SearchRouter(source, target, settings, CutoverState(mode=SHADOW))

    served = router.search([0.1, 0.2, 0.3], top_k=3)

    assert served == ["doc-0", "doc-1", "doc-2"]
    assert router.stats.source_reads == 1
    assert router.stats.shadow_comparisons == 1
    assert router.stats.shadow_diffs == 0


def test_router_rollback_is_a_percentage_change():
    settings = make_settings()
    source = FakeSourceIndex({"a": source_record("a")})
    target = FakeTargetIndex({"a": target_doc("a")})
    router = SearchRouter(source, target, settings, CutoverState(mode=RAMP, target_read_pct=100.0))

    router.search([0.1, 0.2, 0.3])
    assert router.stats.target_reads == 1

    router.state.target_read_pct = 0.0
    router.search([0.1, 0.2, 0.3])
    assert router.stats.source_reads == 1


def test_router_done_never_reads_the_source():
    settings = make_settings()
    source = FakeSourceIndex({"a": source_record("a")})
    target = FakeTargetIndex({"a": target_doc("a")})
    router = SearchRouter(source, target, settings, CutoverState(mode=DONE))

    router.search([0.1, 0.2, 0.3])

    assert router.stats.source_reads == 0
    assert router.stats.target_reads == 1


def test_rollback_from_done_returns_reads_to_the_source():
    """The documented rollback has to work from `done`, not only mid-ramp."""
    settings = make_settings()
    source = FakeSourceIndex({"a": source_record("a")})
    target = FakeTargetIndex({"a": target_doc("a")})
    router = SearchRouter(source, target, settings, CutoverState(mode=DONE))

    router.search([0.1, 0.2, 0.3])
    assert router.stats.target_reads == 1

    router.state.mode, router.state.target_read_pct = RAMP, 0.0
    router.search([0.1, 0.2, 0.3])
    assert router.stats.source_reads == 1


def test_shadow_comparisons_are_not_counted_as_target_reads():
    settings = make_settings()
    source = FakeSourceIndex({"a": source_record("a")})
    target = FakeTargetIndex({"a": target_doc("a")})
    router = SearchRouter(source, target, settings, CutoverState(mode=SHADOW))

    router.search([0.1, 0.2, 0.3])

    assert router.stats.source_reads == 1
    assert router.stats.target_reads == 0
    assert router.stats.shadow_comparisons == 1


def test_cutover_state_rejects_unknown_and_invalid_fields(tmp_path):
    import json as _json

    import pytest as _pytest

    path = tmp_path / "cutover.json"
    path.write_text(_json.dumps({"mode": "ramp", "target_read_pct": 10.0, "future_field": 1}))
    with _pytest.raises(ValueError, match="unrecognised"):
        CutoverState.load(path)

    path.write_text(_json.dumps({"mode": "ramp", "target_read_pct": 500.0}))
    with _pytest.raises(ValueError, match="between 0 and 100"):
        CutoverState.load(path)


def test_state_reload_picks_up_a_rollback(tmp_path):
    path = tmp_path / "cutover.json"
    CutoverState(mode=DONE, target_read_pct=100.0).save(path)
    state = CutoverState.load(path)

    CutoverState(mode=RAMP, target_read_pct=0.0).save(path)
    state.reload(path)

    assert (state.mode, state.target_read_pct) == (RAMP, 0.0)


def test_drop_fields_are_matched_after_rename_too():
    """convert drops on either spelling; reconcile must agree or it invents mismatches."""
    from fts_migrate.config import ConvertSettings

    settings = make_settings(
        convert=ConvertSettings(rename_fields={"body": "text_body"}, drop_fields=["text_body"])
    )
    source = FakeSourceIndex(
        {"a": {"id": "a", "values": [0.1, 0.2, 0.3], "metadata": {"body": "x"}}}
    )
    target = FakeTargetIndex({"a": {"_id": "a", "embedding": [0.1, 0.2, 0.3], "text": None}})

    report = reconcile.diff_fields(source, target, settings, ["a"])

    assert report.metadata_mismatches == []

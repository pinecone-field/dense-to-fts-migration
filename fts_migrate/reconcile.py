"""Prove the target index matches the source before anything is cut over.

Four independent checks, weakest to strongest: record counts, an id-level diff both
ways, a field-level comparison on a sample, and query parity. A migration is signed
off when the id diff is empty and dense parity is ~1.0 — counts alone can agree while
the wrong documents are present.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import dense_source, target_index
from .config import Settings

MAX_REPORTED_IDS = 25


@dataclass
class CountReport:
    source: int
    target: int

    @property
    def ok(self) -> bool:
        return self.source == self.target

    def render(self) -> str:
        verdict = "match" if self.ok else f"differ by {abs(self.source - self.target)}"
        return f"counts: source={self.source} target={self.target} ({verdict})"


@dataclass
class IdReport:
    checked: int
    missing: list[str] = field(default_factory=list)
    orphaned: list[str] = field(default_factory=list)
    sampled: bool = False

    @property
    def ok(self) -> bool:
        return not self.missing and not self.orphaned

    def render(self) -> str:
        scope = "sample" if self.sampled else "full sweep"
        lines = [
            f"ids ({scope}, {self.checked} checked): "
            f"{len(self.missing)} missing, {len(self.orphaned)} orphaned"
        ]
        if self.missing:
            lines.append(f"  missing in target: {', '.join(self.missing[:MAX_REPORTED_IDS])}")
        if self.orphaned:
            lines.append(f"  present only in target: {', '.join(self.orphaned[:MAX_REPORTED_IDS])}")
        return "\n".join(lines)


@dataclass
class FieldReport:
    checked: int
    vector_mismatches: list[str] = field(default_factory=list)
    text_mismatches: list[str] = field(default_factory=list)
    metadata_mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.vector_mismatches or self.text_mismatches or self.metadata_mismatches)

    def render(self) -> str:
        return (
            f"fields ({self.checked} checked): {len(self.vector_mismatches)} vector, "
            f"{len(self.text_mismatches)} text, {len(self.metadata_mismatches)} metadata mismatches"
        )


DEFAULT_MIN_RECALL = 0.99


@dataclass
class ParityReport:
    queries: int
    top_k: int
    recall: float
    jaccard: float
    absent_ids: list[str] = field(default_factory=list)
    min_recall: float = DEFAULT_MIN_RECALL

    @property
    def ok(self) -> bool:
        """Pass unless documents are genuinely missing.

        Two indexes holding identical vectors still rank approximately, so a hit can
        fall either side of the top_k boundary and drag recall below the threshold
        while every document is present. Only a document the target cannot return at
        all is a migration failure.
        """
        return self.recall >= self.min_recall or not self.absent_ids

    def render(self) -> str:
        line = (
            f"dense parity ({self.queries} queries, top_k={self.top_k}): "
            f"recall={self.recall:.3f} jaccard={self.jaccard:.3f}"
        )
        if self.recall >= self.min_recall:
            return line
        if self.absent_ids:
            return (
                f"{line}\n  {len(self.absent_ids)} document(s) in the source's results are "
                f"missing from the target: {', '.join(self.absent_ids[:MAX_REPORTED_IDS])}"
            )
        return (
            f"{line}\n  every document the source ranked is present on the target, so the "
            f"shortfall is ranking order at the top_k boundary rather than missing data"
        )


def compare_counts(source: Any, target: Any, settings: Settings) -> CountReport:
    return CountReport(
        source=dense_source.record_count(source, settings.source.namespace),
        target=target_index.record_count(target, settings.target.namespace),
    )


def diff_ids(source: Any, target: Any, settings: Settings, sample: int | None = None) -> IdReport:
    """Diff the id sets both ways.

    Orphans — documents on the target that the source no longer has — matter as much
    as missing ones: they are what a dropped delete looks like.
    """
    source_ids = set(dense_source.iter_ids(source, settings.source.namespace, limit=sample))
    if sample is not None:
        found = target_index.fetch_documents(
            target, settings.target.namespace, list(source_ids), include_fields=["_id"]
        )
        return IdReport(
            checked=len(source_ids),
            missing=sorted(source_ids - set(found)),
            sampled=True,
        )

    target_ids = set(target_index.iter_document_ids(target, settings.target.namespace))
    return IdReport(
        checked=len(source_ids | target_ids),
        missing=sorted(source_ids - target_ids),
        orphaned=sorted(target_ids - source_ids),
    )


def diff_fields(
    source: Any, target: Any, settings: Settings, ids: Sequence[str], tolerance: float = 1e-4
) -> FieldReport:
    """Compare vector, text and metadata for a set of ids on both sides."""
    ids = list(ids)
    source_records = dense_source.fetch_records(source, ids, settings.source.namespace)
    target_docs = target_index.fetch_documents(target, settings.target.namespace, ids)

    report = FieldReport(checked=len(ids))
    dense_field = settings.target.dense_field
    text_field = settings.target.primary_text_field

    for record_id in ids:
        record = source_records.get(record_id)
        doc = target_docs.get(record_id)
        if record is None or doc is None:
            continue
        source_vector = np.asarray(record["values"], dtype=float)
        target_vector = np.asarray(doc.get(dense_field) or [], dtype=float)
        if source_vector.shape != target_vector.shape or not np.allclose(
            source_vector, target_vector, atol=tolerance
        ):
            report.vector_mismatches.append(record_id)

        if record["metadata"].get(settings.source.text_metadata_key) != doc.get(text_field):
            report.text_mismatches.append(record_id)

        for raw_key, value in record["metadata"].items():
            if raw_key == settings.source.text_metadata_key:
                continue
            key = settings.convert.rename_fields.get(raw_key, raw_key)
            if raw_key in settings.convert.drop_fields or key in settings.convert.drop_fields:
                continue
            if key in settings.target.text_fields:
                continue
            if doc.get(key) != value:
                report.metadata_mismatches.append(f"{record_id}:{key}")
                break

    return report


def query_parity(
    source: Any,
    target: Any,
    settings: Settings,
    queries: int = 20,
    top_k: int = 10,
    seed: int = 3,
    min_recall: float = DEFAULT_MIN_RECALL,
) -> ParityReport:
    """Run the same dense queries against both indexes and compare the result sets.

    The vectors are identical and the metric is copied from the source, so anything
    below ~1.0 recall means documents are missing rather than ranked differently.

    The dense clause names its field as `field`, singular. The search guide writes it
    as `fields: [...]`, but SDK v10 models only `field` and rejects the plural form
    before the request is sent, so the singular is what works from Python.
    """
    rng = random.Random(seed)
    candidate_ids = list(dense_source.iter_ids(source, settings.source.namespace, limit=1000))
    if not candidate_ids:
        return ParityReport(queries=0, top_k=top_k, recall=0.0, jaccard=0.0)
    probe_ids = rng.sample(candidate_ids, min(queries, len(candidate_ids)))
    probes = dense_source.fetch_records(source, probe_ids, settings.source.namespace)

    recalls: list[float] = []
    jaccards: list[float] = []
    shortfall: set[str] = set()
    for record in probes.values():
        vector = record["values"]
        source_hits = source.query(
            vector=vector, top_k=top_k, namespace=settings.source.namespace
        )
        source_ids = {match["id"] for match in source_hits["matches"]}

        response = target.documents.search(
            namespace=settings.target.namespace,
            top_k=top_k,
            score_by=[
                {
                    "type": "dense_vector",
                    "field": settings.target.dense_field,
                    "values": vector,
                }
            ],
        )
        target_ids = {match.id for match in response.matches}

        if not source_ids:
            continue
        overlap = len(source_ids & target_ids)
        shortfall |= source_ids - target_ids
        recalls.append(overlap / len(source_ids))
        jaccards.append(overlap / len(source_ids | target_ids))

    absent: list[str] = []
    if shortfall:
        found = target_index.fetch_documents(
            target, settings.target.namespace, sorted(shortfall), include_fields=["_id"]
        )
        absent = sorted(shortfall - set(found))

    return ParityReport(
        queries=len(recalls),
        top_k=top_k,
        recall=float(np.mean(recalls)) if recalls else 0.0,
        jaccard=float(np.mean(jaccards)) if jaccards else 0.0,
        absent_ids=absent,
        min_recall=min_recall,
    )


def bm25_probe(
    target: Any, settings: Settings, query: str, top_k: int = 5
) -> list[tuple[str, float]]:
    """Run a keyword search — the capability the dense index did not have."""
    response = target.documents.search(
        namespace=settings.target.namespace,
        top_k=top_k,
        score_by=[{"type": "text", "query": query, "fields": list(settings.target.text_fields)}],
        include_fields=list(settings.target.text_fields),
    )
    return [(match.id, float(match.score or 0.0)) for match in response.matches]

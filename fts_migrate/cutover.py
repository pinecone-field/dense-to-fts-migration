"""Move read traffic from the dense index to the document index, reversibly.

The router is the piece the application swaps in. It keeps both indexes reachable and
moves a growing share of reads to the target, so rolling back is a percentage change
rather than a redeploy. Writes are not routed: dual-write stays on until the ramp is
finished and the old index is decommissioned.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

SHADOW = "shadow"
RAMP = "ramp"
DONE = "done"
MODES = (SHADOW, RAMP, DONE)


@dataclass
class RouterStats:
    source_reads: int = 0
    target_reads: int = 0
    shadow_comparisons: int = 0
    shadow_diffs: int = 0

    def render(self) -> str:
        line = f"reads: source={self.source_reads} target={self.target_reads}"
        if self.shadow_comparisons:
            rate = self.shadow_diffs / self.shadow_comparisons
            line += (
                f" | shadow: {self.shadow_comparisons} compared, "
                f"{self.shadow_diffs} differed ({rate:.1%})"
            )
        return line


@dataclass
class CutoverState:
    mode: str = SHADOW
    target_read_pct: float = 0.0
    note: str = ""

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not 0.0 <= self.target_read_pct <= 100.0:
            raise ValueError("target_read_pct must be between 0 and 100")

    @classmethod
    def load(cls, path: Path) -> CutoverState:
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"{path}: unrecognised cutover state field(s): {', '.join(unknown)}")
        state = cls(**{k: v for k, v in raw.items() if k in known})
        state.validate()
        return state

    def reload(self, path: Path) -> CutoverState:
        """Re-read the state file so a running process picks up a ramp or a rollback.

        A `SearchRouter` holds one state object for the life of the process, so without
        this a `migrate.py cutover` command changes the file and nothing else.
        """
        fresh = CutoverState.load(path)
        self.mode, self.target_read_pct, self.note = fresh.mode, fresh.target_read_pct, fresh.note
        return self

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


class SearchRouter:
    """Serves dense search from whichever index the current cutover state says.

    In `shadow` mode the source answers and the target is queried alongside for
    comparison only, so a mismatch shows up in the diff counter without ever reaching
    a user.
    """

    def __init__(
        self,
        source: Any,
        target: Any,
        settings: Any,
        state: CutoverState | None = None,
        seed: int | None = None,
    ) -> None:
        self.source = source
        self.target = target
        self.settings = settings
        self.state = state or CutoverState()
        self.state.validate()
        self.stats = RouterStats()
        self.diffs: list[dict[str, Any]] = []
        self._rng = random.Random(seed)

    def search(self, vector: Sequence[float], top_k: int = 10) -> list[str]:
        """Serve a dense search from whichever index the cutover state selects.

        `done` means every read goes to the target; `target_read_pct` only governs a
        `ramp`. Rolling back is therefore a move back into `ramp`, which is what
        setting a percentage does — see `migrate.py cutover`.
        """
        if self.state.mode == DONE or (
            self.state.mode == RAMP and self._rng.uniform(0, 100) < self.state.target_read_pct
        ):
            self.stats.target_reads += 1
            return self._target_search(vector, top_k)

        served = self._source_search(vector, top_k)
        self.stats.source_reads += 1
        if self.state.mode == SHADOW:
            self._compare(vector, top_k, served)
        return served

    def search_text(self, query: str, top_k: int = 10) -> list[str]:
        """Keyword search, which only the target index can answer."""
        self.stats.target_reads += 1
        response = self.target.documents.search(
            namespace=self.settings.target.namespace,
            top_k=top_k,
            score_by=[
                {"type": "text", "query": query, "fields": list(self.settings.target.text_fields)}
            ],
        )
        return [match.id for match in response.matches]

    def _source_search(self, vector: Sequence[float], top_k: int) -> list[str]:
        hits = self.source.query(
            vector=list(vector), top_k=top_k, namespace=self.settings.source.namespace
        )
        return [match["id"] for match in hits["matches"]]

    def _target_search(self, vector: Sequence[float], top_k: int) -> list[str]:
        response = self.target.documents.search(
            namespace=self.settings.target.namespace,
            top_k=top_k,
            score_by=[
                {
                    "type": "dense_vector",
                    "field": self.settings.target.dense_field,
                    "values": list(vector),
                }
            ],
        )
        return [match.id for match in response.matches]

    def _compare(self, vector: Sequence[float], top_k: int, served: list[str]) -> None:
        """Query the target for comparison only.

        Not counted as a target read, because the read split is how you tell how far the
        cutover has actually gone, and a shadow query was never served to anyone.
        """
        shadow = self._target_search(vector, top_k)
        self.stats.shadow_comparisons += 1
        if set(shadow) != set(served):
            self.stats.shadow_diffs += 1
            if len(self.diffs) < 50:
                self.diffs.append({"served": served, "shadow": shadow})

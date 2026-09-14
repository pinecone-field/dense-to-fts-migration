#!/usr/bin/env python3
"""Phase-by-phase driver for a dense -> full-text-search migration.

Run `python migrate.py --help` for the phase list, or `python migrate.py demo` to
watch every phase run end to end against throwaway indexes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fts_migrate import cutover as cutover_mod
from fts_migrate import dense_source, importer, reconcile, simulate, target_index
from fts_migrate.cdc import CdcLog, CdcWrappedIndex, apply_changes
from fts_migrate.config import Settings, load_settings
from fts_migrate.convert import DocumentMapper, iter_jsonl_dir, parquet_to_jsonl

CURSOR_NAME = "target"
CUTOVER_STATE_PATH = Path("work/cutover.json")
HEARTBEAT_SECONDS = 60


@dataclass
class Context:
    settings: Settings
    _pc: Any = None

    @property
    def pc(self) -> Any:
        """Connect lazily so the offline phases work without an API key."""
        if self._pc is None:
            self._pc = dense_source.connect(self.settings)
        return self._pc

    @property
    def source_spec(self) -> dense_source.SourceSpec:
        return dense_source.describe_source(self.pc, self.settings.source.index)

    def source(self) -> Any:
        return dense_source.open_index(self.pc, self.settings.source.index)

    def target(self) -> Any:
        return target_index.open_index(self.pc, self.settings)

    def log(self) -> CdcLog:
        return CdcLog(self.settings.cdc.db)

    def mapper(self, allow_missing_text: bool = False) -> DocumentMapper:
        return DocumentMapper(
            self.settings,
            dimension=self.source_spec.dimension,
            allow_missing_text=allow_missing_text,
        )

    def export_dir(self) -> Path:
        return self.settings.export.dir / self.settings.source.namespace

    def jsonl_dir(self) -> Path:
        return self.settings.convert.dir / self.settings.target.namespace


def build_context(args: argparse.Namespace) -> Context:
    return Context(settings=load_settings(args.config))


def say(message: str) -> None:
    print(message, flush=True)


def cmd_simulate_source(ctx: Context, args: argparse.Namespace) -> int:
    settings = ctx.settings
    simulate.ensure_demo_index(ctx.pc, settings)
    index = ctx.source()
    count = args.records or settings.demo.records
    written = simulate.seed_index(index, settings, count)
    say(f"seeded {written} records into {settings.source.index}/{settings.source.namespace}")
    say("give the index a few seconds to make them listable before exporting")
    return 0


def cmd_enable_cdc(ctx: Context, args: argparse.Namespace) -> int:
    with ctx.log() as log:
        head = log.head_seq()
    say(f"CDC log ready at {ctx.settings.cdc.db} (head seq {head})")
    say("")
    say("Wrap the index your application writes through, BEFORE taking the export:")
    say("")
    say("    from fts_migrate.cdc import CdcLog, CdcWrappedIndex")
    say(f"    log = CdcLog({str(ctx.settings.cdc.db)!r})")
    say(f"    index = CdcWrappedIndex(pc.Index(name={ctx.settings.source.index!r}), log,")
    say(f"                            namespace={ctx.settings.source.namespace!r})")
    say("")
    say("Every upsert and delete through that wrapper is captured for replay.")
    return 0


def cmd_workload(ctx: Context, args: argparse.Namespace) -> int:
    settings = ctx.settings
    with ctx.log() as log:
        source = ctx.source()
        index = CdcWrappedIndex(source, log, namespace=settings.source.namespace)
        workload = simulate.Workload(index, settings, rate_per_second=args.rate)
        workload.prime(
            list(dense_source.iter_ids(source, settings.source.namespace, limit=500))
        )
        workload.start()
        say(f"writing to {settings.source.index} at {args.rate}/s for {args.seconds}s")
        try:
            time.sleep(args.seconds)
        except KeyboardInterrupt:
            pass
        stats = workload.stop()
        say(f"workload: {stats.summary()}; CDC head seq {log.head_seq()}")
        if stats.errors:
            say(f"workload errors ({len(stats.errors)}): {stats.errors[0]}")
    return 0


def cmd_export(ctx: Context, args: argparse.Namespace) -> int:
    settings = ctx.settings
    with ctx.log() as log:
        snapshot_seq = log.head_seq()
    manifest = simulate.export_namespace(
        ctx.source(),
        settings.source.namespace,
        ctx.export_dir(),
        rows_per_file=settings.export.rows_per_file,
        include_text=not args.no_text,
        text_key=settings.source.text_metadata_key,
        snapshot_seq=snapshot_seq,
        source_index=settings.source.index,
    )
    say(
        f"exported {manifest['rows']} rows to {ctx.export_dir()} "
        f"({len(manifest['files'])} files, text_in_metadata={manifest['include_text']}, "
        f"snapshot at CDC seq {snapshot_seq})"
    )
    return 0


def cmd_convert(ctx: Context, args: argparse.Namespace) -> int:
    settings = ctx.settings
    mapper = DocumentMapper(settings, allow_missing_text=args.allow_missing_text)
    out_dir = ctx.jsonl_dir()
    stats = parquet_to_jsonl(
        ctx.export_dir(),
        out_dir,
        mapper,
        use_gzip=settings.convert.gzip,
        rows_per_file=settings.export.rows_per_file,
        strict=not args.lenient,
    )
    say(f"converted {ctx.export_dir()} -> {out_dir}: {stats.summary()}")
    for error in stats.errors:
        say(f"  {error}")
    if stats.skipped_missing_text:
        say(
            f"WARNING: {stats.skipped_missing_text} rows had no text and were skipped. "
            f"They will be absent from the target index until you join the text in and reload."
        )
    return 0 if stats.converted else 1


def cmd_schema(ctx: Context, args: argparse.Namespace) -> int:
    """Print the exact schema `create-target` would send, and create nothing."""
    schema = target_index.build_schema(ctx.settings, ctx.source_spec)
    print(json.dumps(schema, indent=2))
    return 0


def cmd_create_target(ctx: Context, args: argparse.Namespace) -> int:
    settings = ctx.settings
    source_spec = ctx.source_spec
    schema = target_index.build_schema(settings, source_spec)
    say(f"schema: {json.dumps(schema, indent=2)}")
    model = target_index.create_index(ctx.pc, settings, source_spec)
    say(f"target index {settings.target.index} ready (host {getattr(model, 'host', '?')})")
    if settings.import_.mode == "import":
        target_index.assert_namespace_absent(ctx.target(), settings.target.namespace)
        say(f"namespace {settings.target.namespace!r} is absent, as bulk import requires")
    return 0


def cmd_import(ctx: Context, args: argparse.Namespace) -> int:
    settings = ctx.settings
    target = ctx.target()
    jsonl_dir = ctx.jsonl_dir()
    mode = args.mode or settings.import_.mode

    if mode == "upsert":
        loaded = importer.upsert_from_jsonl(
            target, settings.target.namespace, jsonl_dir, batch_size=settings.import_.batch_size
        )
        say(f"upserted {loaded} documents into {settings.target.index}/{settings.target.namespace}")
    else:
        settings.require_env(
            "PINECONE_STORAGE_INTEGRATION_ID",
            "bulk import from a private bucket needs a storage integration id",
        )
        if not settings.import_.uri:
            say("import.uri is not set in the config — nowhere to upload to or import from")
            return 1
        target_index.assert_namespace_absent(target, settings.target.namespace)
        if not args.skip_upload:
            prefix = importer.upload_tree(
                jsonl_dir, settings.import_.uri, settings.target.namespace
            )
            say(f"uploaded {jsonl_dir} -> {prefix}")
        import_id = importer.start_import(
            target,
            settings.import_.uri,
            integration_id=settings.import_.integration_id,
            error_mode=settings.import_.error_mode,
        )
        say(f"import {import_id} started against {settings.import_.uri}")
        if not args.wait:
            say(f"track it with: python migrate.py import-status {import_id}")
            return 0
        seen: set[tuple[Any, Any, Any]] = set()
        started = last_line = time.time()

        def report(state: dict[str, Any]) -> None:
            """Print when the import moves, and at least once a minute regardless.

            An import runs for ten minutes or more at a 20-second poll, so echoing
            every poll buries the line that changed. Going silent is worse, though:
            over a wait that long, no output at all is indistinguishable from a hang,
            and progress can sit on one percentage while records climb.
            """
            nonlocal last_line
            key = (
                state.get("status"),
                state.get("percent_complete"),
                state.get("records_imported"),
            )
            if key in seen and time.time() - last_line < HEARTBEAT_SECONDS:
                return
            seen.add(key)
            last_line = time.time()
            say(
                f"  {state.get('status')} {state.get('percent_complete')}% "
                f"({state.get('records_imported')} records, "
                f"{int(time.time() - started)}s elapsed)"
            )

        state = importer.wait_for_import(target, import_id, on_poll=report)
        say(f"import {import_id} completed: {state.get('records_imported')} records")

    sample_ids = [doc["_id"] for _, doc in zip(range(100), iter_jsonl_dir(jsonl_dir))]
    fetchable = importer.wait_until_searchable(target, settings.target.namespace, sample_ids)
    with ctx.log() as log:
        backlog = log.lag(CURSOR_NAME).pending
    if not fetchable:
        say("WARNING: sampled documents are not fetchable yet. Wait before going further.")
    elif backlog:
        say(f"loaded documents are fetchable — safe to replay the {backlog} captured changes")
    else:
        say("loaded documents are fetchable, and no changes were captured during the load")
    return 0


def cmd_import_status(ctx: Context, args: argparse.Namespace) -> int:
    state = importer.describe(ctx.target(), args.import_id)
    say(json.dumps(state, indent=2, default=str))
    return 0


def cmd_replay(ctx: Context, args: argparse.Namespace) -> int:
    settings = ctx.settings
    with ctx.log() as log:
        stats = apply_changes(
            ctx.target(),
            log,
            ctx.mapper(allow_missing_text=True),
            namespace=settings.target.namespace,
            cursor_name=CURSOR_NAME,
            source_namespace=settings.source.namespace,
            dry_run=args.dry_run,
            batch_size=settings.cdc.batch_size,
        )
        say(("dry run: " if args.dry_run else "") + stats.summary())
        say(f"lag now: {log.lag(CURSOR_NAME)}")
    return 0


def cmd_tail(ctx: Context, args: argparse.Namespace) -> int:
    settings = ctx.settings
    deadline = time.time() + args.seconds if args.seconds else None
    with ctx.log() as log:
        target = ctx.target()
        mapper = ctx.mapper(allow_missing_text=True)
        try:
            while deadline is None or time.time() < deadline:
                stats = apply_changes(
                    target,
                    log,
                    mapper,
                    namespace=settings.target.namespace,
                    cursor_name=CURSOR_NAME,
                    source_namespace=settings.source.namespace,
                    batch_size=settings.cdc.batch_size,
                )
                lag = log.lag(CURSOR_NAME)
                if stats.upserted or stats.deleted:
                    say(f"{stats.summary()} | pending {lag.pending}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            say("stopped")
    return 0


def cmd_reconcile(ctx: Context, args: argparse.Namespace) -> int:
    settings = ctx.settings
    source, target = ctx.source(), ctx.target()

    with ctx.log() as log:
        lag = log.lag(CURSOR_NAME)
        parked = log.unapplied_count()
    if parked:
        say(
            f"NOTE: {parked} change(s) were parked as unapplied and are NOT reflected below. "
            f"Run `migrate.py status` to see them."
        )
    if lag.pending:
        say(
            f"NOTE: {lag.pending} CDC changes are unapplied ({lag.seconds:.0f}s behind). "
            f"Run `replay` first — differences below will include that backlog."
        )

    counts = reconcile.compare_counts(source, target, settings)
    say(counts.render())

    ids = reconcile.diff_ids(source, target, settings, sample=args.sample)
    say(ids.render())

    sample_ids = list(dense_source.iter_ids(source, settings.source.namespace, limit=args.fields))
    fields = reconcile.diff_fields(source, target, settings, sample_ids)
    say(fields.render())
    for mismatch in (fields.vector_mismatches + fields.text_mismatches)[:10]:
        say(f"  mismatch: {mismatch}")

    ok = counts.ok and ids.ok and fields.ok
    say("RECONCILED" if ok else "NOT RECONCILED")
    return 0 if ok else 1


def cmd_parity(ctx: Context, args: argparse.Namespace) -> int:
    settings = ctx.settings
    source, target = ctx.source(), ctx.target()
    parity = reconcile.query_parity(
        source,
        target,
        settings,
        queries=args.queries,
        top_k=args.top_k,
        min_recall=args.min_recall,
    )
    say(parity.render())
    if args.bm25:
        say(f"BM25 {args.bm25!r} (a query the dense index could not answer):")
        for doc_id, score in reconcile.bm25_probe(target, settings, args.bm25):
            say(f"  {doc_id}  {score:.4f}")
    return 0 if parity.ok else 1


def cmd_cutover(ctx: Context, args: argparse.Namespace) -> int:
    state = cutover_mod.CutoverState.load(CUTOVER_STATE_PATH)
    if args.mode:
        state.mode = args.mode
    if args.pct is not None:
        state.target_read_pct = args.pct
        if not args.mode and state.mode != cutover_mod.RAMP:
            say(f"moving from {state.mode} to ramp, since a percentage only governs a ramp")
            state.mode = cutover_mod.RAMP
    if args.note:
        state.note = args.note
    state.save(CUTOVER_STATE_PATH)
    say(f"cutover state: mode={state.mode} target_read_pct={state.target_read_pct}")

    if state.mode == cutover_mod.SHADOW:
        say("the source serves every read; the target is queried alongside for comparison")
    if state.mode == cutover_mod.RAMP:
        say(
            f"{state.target_read_pct}% of reads go to the target. Roll back with "
            f"`python migrate.py cutover --pct 0` — dual-write is still on."
        )
    if state.mode == cutover_mod.DONE:
        say("all reads on the target index. Keep dual-write until you delete the source index.")
        say("roll back with `python migrate.py cutover --pct 0`, which returns you to a ramp.")
    say("a long-running reader picks this up via CutoverState.reload(path).")
    return 0


def cmd_status(ctx: Context, args: argparse.Namespace) -> int:
    settings = ctx.settings
    say(f"config: {settings.path}")
    say(f"source: {settings.source.index}/{settings.source.namespace}")
    say(f"target: {settings.target.index}/{settings.target.namespace}")
    export_dir, jsonl_dir = ctx.export_dir(), ctx.jsonl_dir()
    say(f"export: {export_dir} ({'present' if export_dir.exists() else 'missing'})")
    say(f"jsonl:  {jsonl_dir} ({'present' if jsonl_dir.exists() else 'missing'})")
    if settings.cdc.db.exists():
        with ctx.log() as log:
            say(f"cdc:    {log.lag(CURSOR_NAME)}")
            parked = log.unapplied_count()
            if parked:
                say(f"unapplied: {parked} change(s) replay could not apply:")
                for row in log.unapplied(limit=10):
                    say(f"  seq {row['seq']}  {row['doc_id']}  {row['reason']}")
                say("  those documents are stale on the target until you reload them")
    else:
        say(f"cdc:    {settings.cdc.db} (not created)")
    state = cutover_mod.CutoverState.load(CUTOVER_STATE_PATH)
    say(f"cutover: mode={state.mode} target_read_pct={state.target_read_pct}")
    return 0


def cmd_demo(ctx: Context, args: argparse.Namespace) -> int:
    """Run every phase against throwaway indexes, with writes landing throughout."""
    settings = ctx.settings
    say("== phase 0: a dense index under load")
    cmd_simulate_source(ctx, argparse.Namespace(records=args.records))
    time.sleep(10)

    say("\n== phase 1: start capturing writes (before the export)")
    cmd_enable_cdc(ctx, args)

    with ctx.log() as log:
        source = ctx.source()
        wrapped = CdcWrappedIndex(source, log, namespace=settings.source.namespace)
        workload = simulate.Workload(wrapped, settings, rate_per_second=settings.demo.write_rate)
        workload.prime(list(dense_source.iter_ids(source, settings.source.namespace, limit=500)))
        workload.start()
        say(f"workload running at {settings.demo.write_rate}/s")

        try:
            say("\n== phase 2: export the index to Parquet")
            cmd_export(ctx, argparse.Namespace(no_text=False))

            say("\n== phase 3: convert Parquet -> JSONL")
            cmd_convert(ctx, argparse.Namespace(allow_missing_text=False, lenient=False))

            say("\n== phase 4: create the target index")
            cmd_create_target(ctx, args)

            say("\n== phase 5: load the documents")
            cmd_import(ctx, argparse.Namespace(mode=args.mode, skip_upload=False, wait=True))
        finally:
            stats = workload.stop()
        say(f"\nworkload stopped: {stats.summary()}")

        captured = log.head_seq()
        say(f"CDC captured {captured} changes from {stats.writes} writes")
        if stats.failures or (stats.writes and not captured):
            say(
                "FAILED: the workload wrote to the source index but the CDC log did not "
                "record it. Every one of those writes would be lost in a real migration."
            )
            return 1

    say("\n== phase 6: replay the writes that landed during the load")
    cmd_replay(ctx, argparse.Namespace(dry_run=False))
    time.sleep(15)
    cmd_replay(ctx, argparse.Namespace(dry_run=False))

    say("\n== phase 7: reconcile")
    time.sleep(15)
    reconciled = cmd_reconcile(ctx, argparse.Namespace(sample=None, fields=50))
    cmd_parity(ctx, argparse.Namespace(queries=10, top_k=10, bm25="bm25 relevance ranking"))

    say("\n== phase 8: cutover")
    cmd_cutover(ctx, argparse.Namespace(mode="shadow", pct=0.0, note="demo"))
    say("\ndemo complete" + ("" if reconciled == 0 else " (reconciliation reported differences)"))
    return reconciled


def cmd_teardown(ctx: Context, args: argparse.Namespace) -> int:
    if not args.yes:
        say("refusing to delete indexes without --yes")
        return 1
    settings = ctx.settings
    for name in (settings.source.index, settings.target.index):
        if ctx.pc.has_index(name):
            ctx.pc.delete_index(name)
            say(f"deleted index {name}")
    if settings.cdc.db.exists():
        settings.cdc.db.unlink()
        say(f"deleted {settings.cdc.db}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate-source", help="create and seed a throwaway dense index")
    p.add_argument("--records", type=int, default=0)
    p.set_defaults(func=cmd_simulate_source)

    p = sub.add_parser("enable-cdc", help="create the CDC log and print the wrapper snippet")
    p.set_defaults(func=cmd_enable_cdc)

    p = sub.add_parser("workload", help="drive writes at the source index through the CDC wrapper")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--rate", type=float, default=5)
    p.set_defaults(func=cmd_workload)

    p = sub.add_parser("export", help="write the source namespace to Parquet")
    p.add_argument("--no-text", action="store_true", help="drop the text field, as some exports do")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("convert", help="convert exported Parquet into import-ready JSONL")
    p.add_argument("--allow-missing-text", action="store_true")
    p.add_argument("--lenient", action="store_true", help="skip bad rows instead of stopping")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser(
        "schema", help="print the resolved schema JSON without creating anything"
    )
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("create-target", help="create the index with the document schema")
    p.set_defaults(func=cmd_create_target)

    p = sub.add_parser("import", help="load the JSONL by bulk import or by upsert")
    p.add_argument("--mode", choices=["import", "upsert"])
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument("--no-wait", dest="wait", action="store_false", default=True)
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("import-status", help="describe a running or finished import")
    p.add_argument("import_id")
    p.set_defaults(func=cmd_import_status)

    p = sub.add_parser("replay", help="apply the CDC backlog to the target index")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("tail", help="keep applying changes as they arrive")
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--interval", type=float, default=5)
    p.set_defaults(func=cmd_tail)

    p = sub.add_parser("reconcile", help="compare counts, ids and fields")
    p.add_argument("--sample", type=int, help="check this many source ids instead of sweeping both")
    p.add_argument("--fields", type=int, default=100, help="records to compare field by field")
    p.set_defaults(func=cmd_reconcile)

    p = sub.add_parser("parity", help="compare dense query results across both indexes")
    p.add_argument("--queries", type=int, default=20)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--min-recall",
        type=float,
        default=reconcile.DEFAULT_MIN_RECALL,
        help="recall below this is investigated rather than assumed fatal",
    )
    p.add_argument("--bm25", help="also run this keyword query against the target")
    p.set_defaults(func=cmd_parity)

    p = sub.add_parser("cutover", help="set the read-routing state")
    p.add_argument("--mode", choices=list(cutover_mod.MODES))
    p.add_argument("--pct", type=float)
    p.add_argument("--note")
    p.set_defaults(func=cmd_cutover)

    p = sub.add_parser("status", help="where the migration currently stands")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("demo", help="run every phase end to end")
    p.add_argument("--records", type=int, default=0)
    p.add_argument("--mode", choices=["import", "upsert"], default="upsert")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("teardown", help="delete the demo indexes and the CDC log")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_teardown)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ctx = build_context(args)
    return int(args.func(ctx, args) or 0)


if __name__ == "__main__":
    sys.exit(main())

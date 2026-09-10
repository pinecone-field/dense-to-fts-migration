# Migrating a live dense index to Pinecone full-text search — a step-by-step manual

This manual walks you through moving a running workload from an **index of dense vectors**
to an **index with a document schema** — the index type that does BM25 keyword ranking,
Lucene query syntax, and text-match filters, alongside vector search.

**The guiding principle is safety.** Your dense index stays the source of truth and keeps
serving traffic the entire time. Every step against it is **read-only**. You only cut reads
over after you have proven the new index returns the same results, and you can roll back
instantly by routing reads back to the dense index.

> Want to see it run first? `python migrate.py demo` does every step below against throwaway
> indexes, with writes landing the whole time, so you can watch the flow end to end before
> touching anything real. `migration_walkthrough.ipynb` is the same thing, one phase per cell.

### Why this is a migration and not a setting

A document schema is fixed at index creation, and an existing dense index cannot be given
one. Adding full-text search means **creating a new index and reloading your data** — there
is no in-place upgrade. That is the whole reason this repo exists.

### What you will do

1. Install, configure, and point the toolkit at your index.
2. Choose the target schema — the one decision you cannot change later.
3. **Turn on change capture, before anything else.**
4. Export the dense index to Parquet.
5. Convert the Parquet to JSONL, the format document-schema imports take.
6. Create the target index.
7. Bulk import the JSONL (or upsert it, for smaller datasets).
8. Replay the writes that landed during the load, then stay caught up.
9. Reconcile the two indexes and check query parity.
10. Ramp reads over, with a rollback that is one command.

---

## Background: how the two index types differ

| Dense index (vector API) | Document-schema index (documents API) |
|---|---|
| A **record**: `id`, `values`, `metadata` | A **document**: `_id` plus named fields |
| `metadata` is an untyped blob, filterable | Fields you declare in a **schema** are searchable; everything else is auto-indexed as filterable metadata |
| Text lives in metadata and is not searchable as text | A `string` field with `full_text_search` is BM25-ranked and supports `$match_phrase` / `$match_all` / `$match_any` |
| `index.upsert` / `query` / `fetch` / `list` | `index.documents.upsert` / `search` / `fetch` / `list` / `update` / `delete` |
| Ranking is the index metric | Ranking is chosen per request with `score_by`: `text`, `query_string`, `dense_vector`, `sparse_vector` |
| Bulk import reads **Parquet** | Bulk import reads **JSONL** |
| Metadata size limit applies to everything | The 40 KB metadata limit does **not** apply to full-text `string` fields |

The two endpoint families do not cross over: a document-schema index is not reachable through
`/vectors/*`, and a dense index is not reachable through `/namespaces/*/documents/*`.

Reference: [Full-text search](https://docs.pinecone.io/guides/search/full-text-search),
[Import records](https://docs.pinecone.io/guides/index-data/import-data),
[Data modeling](https://docs.pinecone.io/guides/index-data/data-modeling).

---

## Step 0 — Set up

Requires Python 3.10+, and the Pinecone Python SDK **v10 or later** (the first version with
the documents API). Full-text search uses API version `2026-07`.

```bash
git clone https://github.com/pinecone-field/dense-to-fts-migration
cd dense-to-fts-migration
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # add your PINECONE_API_KEY
cp config.example.yaml config.yaml
set -a; source .env; set +a
```

Edit `config.yaml`: name your source index and namespace, the metadata key holding your text,
and what you want the target index and its fields to be called. Then:

```bash
python migrate.py status
```

---

## Step 1 — Choose the target schema

**This is the irreversible step.** Fields cannot be added, removed, or retyped after the index
is created; a schema change means another migration. Decide now.

The default in `config.yaml` declares a dense vector field plus one full-text field:

```yaml
target:
  dense_field: embedding
  text_fields: [text]
  full_text_search:
    language: en
    stemming: true
    stop_words: true
```

which produces:

```python
SchemaBuilder()
  .add_dense_vector_field("embedding", dimension=1536, metric="cosine")
  .add_string_field("text", full_text_search={"language": "en", "stemming": True, "stop_words": True})
  .build()
```

Carrying the vector across is what makes the new index a **superset** of the old one: it can
still answer every semantic query the dense index answers, and it adds keyword ranking. The
dimension and metric are read from your source index rather than configured by hand, so they
cannot drift.

Things to settle before you create the index:

- **Which metadata keys become searchable text.** One `string` field per searchable chunk of
  text; up to 100 of them. Everything else stays metadata and is auto-indexed for filtering.
- **Stemming and stop words**, per field. `stemming: true` matches "running" to "run";
  `stop_words: true` drops "the", "a", "of" from the index.
- **Substring search** (`ngram`) if you need it — it cannot be combined with stemming or stop
  words on the same field.
- **At most one `dense_vector` and one `sparse_vector` field** per index.
- Field names may not start with `_` or `$`, and are limited to 64 bytes. If your metadata has
  such a key, map it with `convert.rename_fields` or drop it with `convert.drop_fields`.

```bash
python migrate.py create-target        # prints the schema before creating anything
```

You can run this now to see the schema, or leave it until step 5.

---

## Step 2 — Turn on change capture, **before** you export

Your index is still taking writes. Bulk import can only create namespaces that do not yet
exist, so nothing can be written into the target namespace until the import finishes — which
means every write that lands in between has to be buffered and replayed afterwards.

**Order matters.** Capture must start before the export snapshot, or writes in the gap are
lost with no way to detect it.

```bash
python migrate.py enable-cdc
```

That creates the log and prints the two lines to add to your application:

```python
from fts_migrate.cdc import CdcLog, CdcWrappedIndex

log = CdcLog("work/cdc.sqlite")
index = CdcWrappedIndex(pc.Index(name="your-dense-index"), log, namespace="__default__")
```

`CdcWrappedIndex` is a drop-in for the index object you already write through. It writes to
the dense index exactly as before, then appends the change to the log. Reads and every other
method pass straight through. The log records the whole record, not just its id, so replay
never has to read back from an index that has since moved on.

Two constraints worth knowing:

- **A log append failure raises.** A silently dropped change is the one failure this migration
  cannot survive, so it is loud by design.
- **`delete_all` and filtered deletes are refused** while the wrapper is in place: neither can
  be replayed as a set of ids. Resolve the filter to ids first (`index.list` or `index.query`),
  then delete by id.

If you use a queue, Kafka topic, or outbox table already, point it at `CdcLog.append_upserts`
and `CdcLog.append_deletes` instead of wrapping the index — the rest of the flow is unchanged.

---

## Step 3 — Export the dense index to Parquet

The real export path is a backup export: create a backup, then
[ask Support to export it](https://docs.pinecone.io/guides/manage-data/export-backup) to your
S3 or GCS bucket. What you get back is Parquet in the import format — `id`, `values`,
optional `sparse_values`, and `metadata` as a JSON string, one directory per namespace.

This repo simulates that so the flow is runnable today:

```bash
python migrate.py export                 # text in metadata, the happy path
python migrate.py export --no-text       # text stripped, see "No source text?" below
```

It writes `work/export/<namespace>/N.parquet` plus a `manifest.json` recording the row count
and the **CDC sequence number at snapshot time**, so you always know which writes the export
predates.

If you already have export files, skip this step and drop them in
`work/export/<namespace>/` instead.

---

## Step 4 — Convert Parquet to JSONL

Document-schema imports read [JSON Lines](https://jsonlines.org/), not Parquet — one JSON
document per line, identical in shape to what you would pass to `documents.upsert`.

```bash
python migrate.py convert
```

The mapping is: `id` → `_id`, `values` → your dense field, and the `metadata` JSON string
spread into top-level fields. Keys that match a schema field are validated against it; the
rest are stored and auto-indexed as filterable metadata.

```jsonl
{"_id": "doc-42", "embedding": [0.12, 0.34, ...], "text": "the searchable body", "category": "search", "year": 2024}
```

Conversion enforces the document-API limits up front, so problems surface on your workstation
rather than as per-row import errors twenty minutes in:

| Limit | Value |
|---|---|
| Document size | 2 MB |
| Documents per upsert request | 1000, and 2 MB per request |
| Full-text `string` field | 100 KB and 10,000 tokens |
| Metadata per document | 40 KB total (full-text fields exempt) |
| Field name | 64 bytes, not starting with `_` or `$` |
| Dense vector | length must equal the schema `dimension`, and present on every document |

`--lenient` skips bad rows and reports them instead of stopping at the first one.

### No source text in the export?

Some exports carry vectors and metadata but not the text — it lives in the system of record,
not in the index. **Conversion stops with an error naming the field and the row.** That is
deliberate: an index whose full-text field is empty cannot do BM25, and a silent load would
leave you with an index that looks fine and ranks nothing.

There is nothing the toolkit can invent here. Before converting, join the text back in from
wherever it lives — a database, a document store, an object-storage corpus — keyed by record
id, and write it into the `metadata` column under the key named in `source.text_metadata_key`.
The `demo` and the tests both cover the happy path where the text is already in metadata.

If you want to load the vectors now and the text later, `--allow-missing-text` skips those
rows and reports how many were skipped. They will be **absent** from the target index until
you join the text in and reload them, so treat that count as a to-do, not a pass.

---

## Step 5 — Create the target index

```bash
python migrate.py create-target
```

Creates the index from the schema in step 1, waits for `status.ready` (and, for dedicated read
capacity, for the read nodes to be `Ready` too — searching before then can return empty
results rather than an error), and checks that the target namespace does **not** exist, which
is what bulk import requires.

---

## Step 6 — Load the documents

Two modes, same documents.

**Bulk import** — the real path, from object storage:

```bash
python migrate.py import --mode import
```

It uploads `work/jsonl/<namespace>/` to `<uri>/<namespace>/`, calls `start_import` against the
whole dataset prefix, and polls until the import finishes. You need a
[storage integration](https://docs.pinecone.io/guides/operations/integrations/manage-storage-integrations)
unless the bucket is public. Set `import.uri` and `import.integration_id` in `config.yaml`.
Automatic upload is implemented for S3; for GCS or Azure, copy the tree up with your provider's
CLI and pass `--skip-upload`.

What to expect: **an import takes at least ten minutes**, up to 10,000 namespaces and 100,000
files, 10 GB per file, and 1 TB of input per import for on-demand indexes (no cap with
dedicated read nodes). `.jsonl.gz` files are measured against an estimated 10× uncompressed
size, so aim for about 1 GB compressed per file. With `error_mode: continue`, bad documents
are skipped and the rest load — compare `records_imported` against the row count `convert`
reported.

**Upsert** — no bucket needed, good for demos and smaller indexes:

```bash
python migrate.py import --mode upsert
```

Streams the same JSONL through `documents.upsert` in batches inside the 1000-document / 2 MB
limits.

Either way, the load ends with a **freshness probe**: documents are indexed asynchronously
after a load reports complete, so the toolkit fetches a sample of loaded ids until they
resolve. Do not replay the backlog before that passes — a delete applied ahead of the document
it deletes is a delete that does nothing.

---

## Step 7 — Replay the backlog, then stay caught up

```bash
python sync.py replay        # drain everything captured since capture started
python sync.py tail          # keep applying, leave this running
```

Replay collapses the log to the **last change per document** and applies it: upserts in
batches of 1000, deletes in batches of 1000. That is safe because `documents.upsert` replaces
the whole document — there is no partial state to rebuild by walking every intermediate write
— and it makes replay **idempotent**, so re-running it is always fine.

The first pass runs from sequence 0 rather than from the export's snapshot sequence. Cheaper
to reason about, and it absorbs any delete of a record the export still contains.

**Partial updates fold in.** `index.update(id=..., set_metadata=...)` on the source is a
patch, not a whole record, so it cannot simply replace the document. Replay handles it by
looking at the document's full history in the window: a patch that follows an upsert is
folded into that upsert and sent as one document; a patch with no upsert behind it — the
document came from the bulk load — is sent as a `documents.update` patch. Filtered updates
are refused by the wrapper for the same reason filtered deletes are.

`tail` reports lag as both a pending-change count and how far behind the oldest unapplied
change is. Leave it running through reconciliation and the whole read ramp.

---

## Step 8 — Prove the two indexes agree

```bash
python sync.py reconcile
python migrate.py parity --bm25 "your keyword query"
```

Four checks, weakest to strongest:

1. **Counts** — `describe_index_stats` on the source against `describe_namespace().record_count`
   on the target. Cheap, and can agree while the wrong documents are present.
2. **Id diff, both directions** — every source id missing from the target, *and* every target
   document the source no longer has. Orphans matter as much as gaps: an orphan is what a
   dropped delete looks like. `--sample N` for a fast pass; the full sweep for sign-off.
3. **Field comparison** — vector (`allclose`), full-text field, and metadata, on a sample.
4. **Query parity** — the same dense queries against both indexes, reported as recall@k and
   Jaccard@k. The vectors and metric are identical, so anything below ~1.0 means documents are
   missing, not ranked differently.

`reconcile` prints the CDC lag first: if changes are unapplied, the differences below it are
that backlog, not corruption. Replay, then re-check.

The `--bm25` probe runs a keyword search the dense index could never have answered — the
capability you did all of this for.

Sign-off is: **id diff empty, field mismatches zero, recall ≈ 1.0, CDC lag near zero.**

---

## Step 9 — Cut over, with a way back

Reads move in stages. Writes keep going to both indexes the entire time, which is what makes
rollback a percentage change rather than a redeploy.

```bash
python migrate.py cutover --mode shadow            # serve dense, query FTS alongside, compare
python migrate.py cutover --mode ramp --pct 1      # 1% of reads served by the new index
python migrate.py cutover --pct 10
python migrate.py cutover --pct 50
python migrate.py cutover --pct 100
python migrate.py cutover --mode done              # all reads on the new index
```

Your application reads through the router, which honours that state:

```python
from fts_migrate.cutover import CutoverState, SearchRouter

router = SearchRouter(dense_index, fts_index, settings, CutoverState.load(Path("work/cutover.json")))
router.search(query_vector, top_k=10)          # routed per the current state
router.search_text("exact phrase here", 10)    # only the new index can answer this
```

In `shadow` mode the dense index answers every request and the document index is queried
alongside for comparison only, so a mismatch shows up in the diff counter without ever
reaching a user.

**Rollback is `python migrate.py cutover --pct 0`.** Nothing else — dual-write is still on, so
the dense index has never fallen behind.

### Decommissioning

Only after the new index has served 100% of reads long enough to trust:

1. Stop dual-write; point writes at the document index only.
2. Stop `sync.py tail` and archive `work/cdc.sqlite`.
3. Enable `deletion_protection` on the new index.
4. Delete the old dense index.

---

## The timeline, in one picture

```
   capture ON                                                     rollback window
       │                                                        ├──────────────────┤
       ▼
 ──────┼────────┬──────────┬───────────┬──────────┬─────────────┬─────┬─────┬──────►
       │        │          │           │          │             │     │     │
    step 2   export     convert    create+load  replay      reconcile shadow ramp  done
             (step 3)   (step 4)   (steps 5-6)  (step 7)     (step 8)   (step 9)

 dense index serves 100% of reads ─────────────────────────────────►│ ramping ►│
 writes go to the dense index AND the CDC log ─────────────────────────────────►│ then FTS only
```

The gap between **export** and **replay** is exactly what the CDC log exists to cover.

---

## Try it end to end first

```bash
python migrate.py demo                 # ~5 minutes, no bucket needed
python migrate.py demo --mode import   # the real bulk-import path, needs S3 + integration
python migrate.py teardown --yes       # delete the throwaway indexes
```

`demo` creates a throwaway dense index, seeds it, starts capture, runs a background workload
that inserts, updates and deletes throughout the export/convert/load, replays the backlog,
reconciles, and prints a BM25 query at the end. A clean run ends with an empty id diff and
recall ≈ 1.0.

The unit tests cover the parts that do not need an API key:

```bash
pytest
```

---

## What has been verified

| Path | Status |
|---|---|
| Everything except bulk import, end to end against live Pinecone | Verified. `migrate.py demo` on a real project: 2,050 records both sides, 88 writes captured and replayed during the load, id diff empty, field diff empty, dense recall 1.000, BM25 returning ranked hits. |
| `--mode import` (bulk import from object storage) | **Not yet run against the service.** The JSONL, directory layout, `start_import` call and polling follow the documented contract, but nobody has watched a real import of these files finish. Run it once against your own bucket before relying on it, and please file an issue if the service disagrees with what this repo produces. |
| Unit tests (`pytest`) | 33 tests, no API key needed: conversion and its limits, the missing-text paths, CDC folding and idempotency, cross-thread capture, reconcile diffing, router routing. |

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `The namespace "x" already exists. Imports are only allowed into nonexistent namespaces.` | Bulk import only creates namespaces. Delete it, or import into a new one. To import into `__default__`, it must be empty. |
| Conversion stops with `no text for full-text field` | The export has no searchable text. Join it in from your system of record — see "No source text?" above. |
| `records_imported` is lower than the rows converted | With `error_mode: continue`, invalid documents are skipped. Describe the import to see the file, row and `_id` of each error. |
| Search returns nothing right after a load | Documents index asynchronously. Wait for the freshness probe, and check `status.ready` (plus read-capacity state, on dedicated). |
| Reconcile reports missing ids on a live index | Check the CDC lag first. Replay, wait for freshness, then re-check. |
| `delete_all`/filtered delete raises from the wrapper | Neither can be replayed as ids. Resolve to ids first, then delete by id. |
| Import fails on an S3 bucket | The bucket must be on the same cloud as the index; S3 Express One Zone is not supported. |

---

## See also

- [Full-text search](https://docs.pinecone.io/guides/search/full-text-search)
- [Import records](https://docs.pinecone.io/guides/index-data/import-data)
- [Export a backup](https://docs.pinecone.io/guides/manage-data/export-backup)
- [Data modeling — schema patterns](https://docs.pinecone.io/guides/index-data/data-modeling)
- [Manage storage integrations](https://docs.pinecone.io/guides/operations/integrations/manage-storage-integrations)
- [Understanding cost](https://docs.pinecone.io/guides/manage-cost/understanding-cost)

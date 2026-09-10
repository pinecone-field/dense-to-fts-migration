#!/usr/bin/env python3
"""Keep the target index in sync with the source after the bulk load.

This is the process you leave running for the length of the migration:

    python sync.py replay            # drain the backlog once
    python sync.py tail              # stay caught up, until you stop it
    python sync.py reconcile         # check both indexes agree

Every subcommand is the same code `migrate.py` runs; this is the operational entry
point, so the phase CLI can be left alone once the migration is underway.
"""

from __future__ import annotations

import argparse
import sys

from migrate import build_context, cmd_reconcile, cmd_replay, cmd_tail


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("replay", help="apply the CDC backlog once and exit")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("tail", help="apply changes continuously")
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--interval", type=float, default=5)
    p.set_defaults(func=cmd_tail)

    p = sub.add_parser("reconcile", help="compare counts, ids and fields")
    p.add_argument("--sample", type=int)
    p.add_argument("--fields", type=int, default=100)
    p.set_defaults(func=cmd_reconcile)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(build_context(args), args) or 0)


if __name__ == "__main__":
    sys.exit(main())

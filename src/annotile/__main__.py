"""Command line for a full run and the explainer page."""

from __future__ import annotations

import argparse
from pathlib import Path

from annotile.pipeline import run
from annotile.report import write_site


def main() -> None:
    parser = argparse.ArgumentParser(prog="annotile")
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("run")
    fit.add_argument("--root", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--window-um", type=float, default=400.0)
    fit.add_argument("--min-cells", type=int, default=30)

    page = sub.add_parser("report")
    page.add_argument("--result", type=Path, required=True)
    page.add_argument("--docs", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "run":
        summary = run(args.root, args.out, args.window_um, args.min_cells)
        print(
            f"windows={summary['windows']} fine_k={summary['method']['fine_k']} "
            f"patterns={summary['method']['merged_patterns']}"
        )
        return
    write_site(args.result, args.docs)
    print(args.docs / "index.html")


if __name__ == "__main__":
    main()

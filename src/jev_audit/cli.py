"""Command line: jev-audit {fetch,build,run,evaluate,all,demo}."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from jev_audit import pipeline
from jev_audit.config import find_repo_root, load_config
from jev_audit.data.eia import EIAError
from jev_audit.runner import SetupError, run_arm

DEFAULT_CONFIG = "configs/experiment.yaml"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jev-audit", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("-c", "--config", default=DEFAULT_CONFIG)
        sp.add_argument(
            "--out-root",
            default=None,
            help="write data/results/reports under this folder instead of the repo",
        )
        return sp

    add("fetch", "download the EIA series (or generate synthetic ones)")
    add("build", "build states and ground-truth labels")
    r = add("run", "run one arm (cached; resumes where it stopped)")
    r.add_argument("--arm", required=True)
    r.add_argument("--limit", type=int, default=None, help="only the first N uncached states")
    r.add_argument("--force", action="store_true", help="discard this arm's cache first")
    add("evaluate", "compute metrics and write the markdown report")
    a = add("all", "fetch, build, run every enabled arm, evaluate")
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--skip-fetch", action="store_true")
    d = sub.add_parser("demo", help="offline end-to-end run on synthetic data (no keys)")
    d.add_argument("--out-root", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (EIAError, SetupError, FileNotFoundError, KeyError, ValueError) as exc:
        if os.environ.get("JEV_AUDIT_DEBUG"):
            raise
        print(f"error: {exc}", file=sys.stderr)
        print("(set JEV_AUDIT_DEBUG=1 for the full traceback)", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "demo":
        config = find_repo_root(Path.cwd()) / "configs" / "demo.yaml"
        cfg = load_config(config, args.out_root)
        pipeline.fetch(cfg)
        pipeline.build(cfg)
        pipeline.run_enabled(cfg)
        pipeline.evaluate_and_report(cfg)
        return 0

    cfg = load_config(args.config, args.out_root)
    if args.cmd == "fetch":
        pipeline.fetch(cfg)
    elif args.cmd == "build":
        pipeline.build(cfg)
    elif args.cmd == "run":
        run_arm(cfg, args.arm, limit=args.limit, force=args.force)
    elif args.cmd == "evaluate":
        pipeline.evaluate_and_report(cfg)
    elif args.cmd == "all":
        if not args.skip_fetch:
            pipeline.fetch(cfg)
        pipeline.build(cfg)
        pipeline.run_enabled(cfg, limit=args.limit)
        pipeline.evaluate_and_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

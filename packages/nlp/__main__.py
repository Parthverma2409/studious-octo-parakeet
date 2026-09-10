"""CLI for the D1 pipeline.

    python -m packages.nlp build                 # generate + split + write JSONL
    python -m packages.nlp evaluate              # baseline on the test split
    python -m packages.nlp evaluate --split dev
    python -m packages.nlp evaluate --extractor ollama --model qwen2.5:7b-instruct
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from packages.nlp.baseline import BaselineExtractor
from packages.nlp.dataset import read_jsonl, split_by_template, write_jsonl
from packages.nlp.generate import DATA_DIR, generate
from packages.nlp.metrics import evaluate as score
from packages.nlp.metrics import safety_critical_confusions

SPLIT_DIR = DATA_DIR / "splits"


def cmd_build(args: argparse.Namespace) -> int:
    utterances = generate(variants_per_template=args.variants, seed=args.seed)
    splits = split_by_template(utterances, seed=args.seed)

    for name, rows in splits.as_items():
        write_jsonl(rows, SPLIT_DIR / ("%s.jsonl" % name))

    print("generated %d utterances from %d template families"
          % (len(utterances), len({u.template_id for u in utterances})))
    print("splits (leakage-checked): %s" % splits.counts())
    print("")
    for name, rows in splits.as_items():
        by_intent = Counter(r.intent.value for r in rows)
        print("%-6s %s" % (name, dict(sorted(by_intent.items()))))
    print("")
    print("written to %s" % SPLIT_DIR)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    path = SPLIT_DIR / ("%s.jsonl" % args.split)
    if not path.exists():
        print("%s not found -- run `python -m packages.nlp build` first" % path)
        return 1
    gold = read_jsonl(path)

    if args.extractor == "baseline":
        extractor = BaselineExtractor()
    else:
        from packages.nlp.extractor import FallbackExtractor, OllamaExtractor

        extractor = FallbackExtractor(OllamaExtractor(model=args.model), BaselineExtractor())

    predictions = [extractor.extract(u.text) for u in gold]
    report = score(gold, predictions)
    unsafe = safety_critical_confusions(gold, predictions)

    print("extractor         : %s" % extractor.name)
    print("split             : %s" % args.split)
    print(report.summary())
    print("")
    print("safety-critical confusions : %d   (decline read as confirm, or out-of-scope"
          " read as cancel/confirm -- must be 0)" % unsafe)

    if args.show_errors:
        print("\nfirst %d errors:" % args.show_errors)
        shown = 0
        for g, p in zip(gold, predictions):
            if shown >= args.show_errors:
                break
            if g.intent != p.intent or g.slot_keys() != {s.key() for s in p.slots}:
                print("  %r" % g.text)
                print("     gold: %-20s %s" % (g.intent.value, [(s.name, s.value) for s in g.slots]))
                print("     pred: %-20s %s" % (p.intent.value, [(s.name, s.value) for s in p.slots]))
                shown += 1

    return 2 if unsafe else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="packages.nlp")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="generate D1 and write leakage-checked splits")
    build.add_argument("--variants", type=int, default=18, help="max fills per template family")
    build.add_argument("--seed", type=int, default=20260907)
    build.set_defaults(func=cmd_build)

    ev = sub.add_parser("evaluate", help="score an extractor against a split")
    ev.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ev.add_argument("--extractor", default="baseline", choices=["baseline", "ollama"])
    ev.add_argument("--model", default="qwen2.5:7b-instruct")
    ev.add_argument("--show-errors", type=int, default=0, metavar="N")
    ev.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

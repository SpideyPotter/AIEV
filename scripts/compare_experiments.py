#!/usr/bin/env python3
"""Results ledger + conference-extension decision from per-class PQ reports.

Reads the ``--metrics-json`` files written by eval_val_perclass_pq.py (and,
for the external-data reference, the older text report format), prints one
row per experiment with PQ / PQ_th / PQ_st / tail PQ, per-class deltas
against the baseline, and with ``--decide`` applies the Path D vs Path B rule.

Usage (headnode is fine, stdlib only):
    python3 scripts/compare_experiments.py
    python3 scripts/compare_experiments.py --decide
    python3 scripts/compare_experiments.py --glob "outputs/m2f_r50_idd_scratch/sweep_*.json"
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TAIL = ["bicycle", "traffic light", "traffic sign", "guard rail", "bridge", "vehicle fallback"]
FALLBACK = ["vehicle fallback", "obs-str-bar-fallback"]
REFERENCE_TXT = REPO / "outputs" / "perclass_pq_44999.txt"
REFERENCE_PQ = 50.97

_ROW = re.compile(r"^(?P<name>.+?)\s+id=\s*\d+\s+thing=(?P<thing>\d)\s+PQ=\s*(?P<pq>[\d.]+)"
                  r"\s+SQ=\s*(?P<sq>[\d.]+)\s+RQ=\s*(?P<rq>[\d.]+)")
_AGG = re.compile(r"^(?P<name>All|Things|Stuff)\s*:\s*PQ=(?P<pq>[\d.]+)\s+SQ=(?P<sq>[\d.]+)"
                  r"\s+RQ=(?P<rq>[\d.]+)")


def load(path: Path) -> dict:
    """Return {'aggregate': {All/Things/Stuff: {pq,sq,rq}}, 'per_class': {name: {pq,sq,rq}}} in %."""
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        scale = lambda r: {k: 100 * r[k] for k in ("pq", "sq", "rq")}  # noqa: E731
        return {"aggregate": {k: scale(v) for k, v in d["aggregate"].items()},
                "per_class": {k: scale(v) for k, v in d["per_class"].items()},
                "coarse": {lv: 100 * c["aggregate"]["All"]["pq"]
                           for lv, c in d.get("coarse", {}).items()}}
    agg, per = {}, {}
    for line in path.read_text().splitlines():
        if m := _ROW.match(line):
            per[m["name"].strip()] = {k: float(m[k]) for k in ("pq", "sq", "rq")}
        elif m := _AGG.match(line):
            agg[m["name"]] = {k: float(m[k]) for k in ("pq", "sq", "rq")}
    return {"aggregate": agg, "per_class": per}


def label_for(path: Path) -> str:
    if path.name.startswith("perclass_pq") and path.parent.name != "outputs":
        return path.parent.name.replace("m2f_r50_idd_scratch", "A0").replace("A0_", "")
    return path.stem


def tail_pq(r: dict) -> float:
    return sum(r["per_class"][c]["pq"] for c in TAIL) / len(TAIL)


def decide(results: dict[str, dict]) -> str:
    """Path D (hierarchy loss) vs Path B (in-domain SSL).

    D if the fallback classes remain a *recognition* failure after the
    long-tail recipe: their SQ-RQ gap exceeds the other classes' by >10
    points and at least one fallback class is in the bottom three.
    B if the scratch model is instead weak everywhere: best PQ below 80 % of
    the external-data reference and SQ more than 5 points below it, i.e.
    masks themselves are poor, which better features (SSL) address.
    Otherwise D, as the cheaper experiment.
    """
    scratch = {k: v for k, v in results.items() if k != "reference"}
    best = max(scratch, key=lambda k: scratch[k]["aggregate"]["All"]["pq"])
    r = scratch[best]
    per = r["per_class"]
    gap = lambda c: per[c]["sq"] - per[c]["rq"]  # noqa: E731
    others = [c for c in per if c not in FALLBACK]
    fb_gap = sum(gap(c) for c in FALLBACK) / len(FALLBACK)
    other_gap = sum(gap(c) for c in others) / len(others)
    bottom3 = sorted(per, key=lambda c: per[c]["pq"])[:3]
    best_pq, best_sq = r["aggregate"]["All"]["pq"], r["aggregate"]["All"]["sq"]
    ref_sq = results["reference"]["aggregate"]["All"]["sq"] if "reference" in results else None

    lines = [f"best scratch experiment: {best} (PQ {best_pq:.2f}, SQ {best_sq:.2f})",
             f"fallback SQ-RQ gap {fb_gap:.1f} vs others {other_gap:.1f}; bottom-3 classes: {bottom3}"]
    if fb_gap - other_gap > 10 and any(c in bottom3 for c in FALLBACK):
        lines.append("=> Path D: fallback classes are still a recognition (label) failure.")
    elif best_pq < 0.8 * REFERENCE_PQ and ref_sq is not None and ref_sq - best_sq > 5:
        lines.append(f"=> Path B: masks are weak across the board (SQ {best_sq:.1f} vs "
                     f"reference {ref_sq:.1f}); try IDD-only self-supervised pretraining "
                     "(confirm with supervisor that SSL on IDD images is allowed).")
    else:
        lines.append("=> Path D (default): neither condition is decisive; D is the cheaper run.")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default=str(REPO / "outputs" / "m2f_r50_idd_scratch*" / "perclass_pq.json"))
    p.add_argument("--baseline", default="A0", help="label of the row to diff against")
    p.add_argument("--no-reference", action="store_true")
    p.add_argument("--decide", action="store_true")
    args = p.parse_args()

    results: dict[str, dict] = {}
    if not args.no_reference and REFERENCE_TXT.exists():
        results["reference"] = load(REFERENCE_TXT)
    matches = sorted(glob.glob(args.glob))
    if not matches:
        print(f"no reports match {args.glob}")
    for f in matches:
        results[label_for(Path(f))] = load(Path(f))

    print("| Exp | PQ | PQ_th | PQ_st | SQ | RQ | Tail PQ | L2 PQ | L1 PQ |\n"
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in results.items():
        a, c = r["aggregate"], r.get("coarse", {})
        lv = lambda k: f"{c[k]:.2f}" if k in c else "-"  # noqa: E731
        print(f"| {name} | {a['All']['pq']:.2f} | {a['Things']['pq']:.2f} | {a['Stuff']['pq']:.2f} "
              f"| {a['All']['sq']:.2f} | {a['All']['rq']:.2f} | {tail_pq(r):.2f} "
              f"| {lv('level2')} | {lv('level1')} |")

    base = results.get(args.baseline)
    others = [k for k in results if k not in (args.baseline, "reference")]
    if base and others:
        print(f"\nPer-class PQ delta vs {args.baseline}:")
        print(f"{'class':24s}" + "".join(f"{k[:12]:>13s}" for k in others))
        for c in sorted(base["per_class"], key=lambda c: base["per_class"][c]["pq"]):
            print(f"{c:24s}" + "".join(
                f"{results[k]['per_class'][c]['pq'] - base['per_class'][c]['pq']:+13.2f}"
                for k in others))

    if args.decide:
        if not any(k != "reference" for k in results):
            print("\n--decide needs at least one from-scratch report")
            return 1
        print("\n" + decide(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

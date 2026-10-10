"""Class-frequency statistics over panoptic ``segments_info`` records.

Shared by the repeat-factor sampler (A1), copy-paste (A2) and the
class-weighted loss (A3). Records are the dicts produced by
``idd_panoptic._load_records``; ``category_id`` is already contiguous.

Run as a script to print the repeat factors and loss weights that a given
config would produce:
    python3 -m src.datasets.class_stats --thresholds 0.1 0.3 0.5
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


def image_frequency(records: Sequence[dict], num_classes: int) -> list[float]:
    """Fraction of images that contain each class."""
    counts: Counter = Counter()
    for r in records:
        counts.update({s["category_id"] for s in r["segments_info"] if not s["iscrowd"]})
    n = max(len(records), 1)
    return [counts[c] / n for c in range(num_classes)]


def segment_counts(records: Iterable[dict], num_classes: int) -> list[int]:
    """Number of (non-crowd) segments per class."""
    counts: Counter = Counter()
    for r in records:
        counts.update(s["category_id"] for s in r["segments_info"] if not s["iscrowd"])
    return [counts[c] for c in range(num_classes)]


def repeat_factors(records: Sequence[dict], num_classes: int, threshold: float) -> list[float]:
    """LVIS repeat factor per image: max over its classes of max(1, sqrt(t / f(c)))."""
    freq = image_frequency(records, num_classes)
    class_rf = [max(1.0, math.sqrt(threshold / f)) if f > 0 else 1.0 for f in freq]
    return [
        max((class_rf[s["category_id"]] for s in r["segments_info"] if not s["iscrowd"]),
            default=1.0)
        for r in records
    ]


def class_loss_weights(records: Sequence[dict], num_classes: int, mode: str,
                       power: float, max_weight: float) -> list[float]:
    """w(c) = (n_max / n(c)) ** power, clamped to [1, max_weight].

    ``img_freq`` counts images containing c; ``seg_freq`` counts segments of c,
    which matches how often c appears as a matched target in the set loss.
    """
    if mode == "img_freq":
        n = image_frequency(records, num_classes)
    elif mode == "seg_freq":
        n = [float(x) for x in segment_counts(records, num_classes)]
    else:
        raise ValueError(f"unknown class-weight mode: {mode}")
    n_max = max(n)
    return [min(max_weight, (n_max / x) ** power) if x > 0 else max_weight for x in n]


def _main() -> None:
    import argparse
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from src.datasets.idd_panoptic import load_split

    p = argparse.ArgumentParser()
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    p.add_argument("--power", type=float, default=0.5)
    p.add_argument("--max-weight", type=float, default=5.0)
    args = p.parse_args()

    records, meta = load_split(repo, "train")
    names = meta["stuff_classes"]
    k = len(names)
    freq = image_frequency(records, k)
    print(f"{len(records)} train images, {k} classes")
    for t in args.thresholds:
        rf = repeat_factors(records, k, t)
        boosted = [names[c] for c in range(k) if freq[c] and freq[c] < t]
        print(f"t={t}: epoch grows x{sum(rf) / len(rf):.3f}, "
              f"{sum(r > 1 for r in rf)} images repeated; classes boosted: {boosted}")
    for mode in ("img_freq", "seg_freq"):
        w = class_loss_weights(records, k, mode, args.power, args.max_weight)
        print(f"\nloss weights ({mode}, power={args.power}, max={args.max_weight}):")
        for c in sorted(range(k), key=lambda c: -w[c]):
            print(f"  {names[c]:24s} {w[c]:.2f}")


if __name__ == "__main__":
    _main()

"""Accuracy and calibration.

Calibration matters here more than it usually does: a hosted decision model's
central claim is that its probabilities mean something. A local classifier emits
probabilities too. Putting the two expected calibration errors side by side is
the comparison that decides the verdict when accuracy ties.
"""

from __future__ import annotations

from ..types import Example, Prediction


def score(
    predictions: list[Prediction], examples: list[Example]
) -> tuple[float, float]:
    """Return (accuracy, macro-F1) over the examples that produced a prediction.

    Errored predictions count as wrong rather than being dropped - a backend
    that fails on 10% of inputs is not 100% accurate on the rest.
    """
    if not predictions:
        return 0.0, 0.0
    gold = [ex.gold for ex in examples[: len(predictions)]]
    pred = [p.label if p.ok else "\x00error" for p in predictions]

    correct = sum(g == p for g, p in zip(gold, pred, strict=True))
    accuracy = correct / len(gold)

    labels = sorted(set(gold))
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, pred, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, pred, strict=True))
        if tp == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall))
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return accuracy, macro_f1


def expected_calibration_error(
    predictions: list[Prediction], examples: list[Example], bins: int = 10
) -> tuple[float | None, list[dict]]:
    """ECE plus the reliability curve behind it.

    Confidence is taken from the backend's own ``confidence`` when it reports
    one, else the probability it assigned to the label it chose. Predictions
    carrying neither are skipped, and if none survive the ECE is None rather
    than a misleading zero.
    """
    pairs: list[tuple[float, bool]] = []
    n = min(len(predictions), len(examples))
    for pred, ex in zip(predictions[:n], examples[:n], strict=True):
        if not pred.ok:
            continue
        conf = pred.confidence
        if conf is None and pred.probabilities:
            conf = pred.probabilities.get(pred.label)
        if conf is None:
            continue
        pairs.append((float(conf), pred.label == ex.gold))

    if not pairs:
        return None, []

    curve: list[dict] = []
    ece = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        # last bin is closed so confidence == 1.0 is not discarded
        in_bin = [
            (c, ok) for c, ok in pairs if (lo <= c < hi or (i == bins - 1 and c == hi))
        ]
        if not in_bin:
            curve.append(
                {"bin_lo": lo, "bin_hi": hi, "count": 0, "confidence": None,
                 "accuracy": None}
            )
            continue
        mean_conf = sum(c for c, _ in in_bin) / len(in_bin)
        mean_acc = sum(ok for _, ok in in_bin) / len(in_bin)
        ece += (len(in_bin) / len(pairs)) * abs(mean_acc - mean_conf)
        curve.append(
            {
                "bin_lo": round(lo, 3),
                "bin_hi": round(hi, 3),
                "count": len(in_bin),
                "confidence": round(mean_conf, 4),
                "accuracy": round(mean_acc, 4),
            }
        )
    return round(ece, 4), curve

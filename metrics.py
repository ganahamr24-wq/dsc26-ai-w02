import numpy as np
import pandas as pd
import math

def _validate(y_true, scores):
    yt = np.asarray(y_true)
    sc = np.asarray(scores, dtype=float)
    if yt.size == 0 or sc.size == 0 or yt.shape != sc.shape:
        raise ValueError("Invalid or mismatched input shapes/empty inputs")
    if not np.all(np.isin(yt, [0, 1])):
        raise ValueError("Labels must be 0 or 1")
    if np.isnan(sc).any():
        raise ValueError("Scores contain NaN")
    return yt, sc

def confusion_at(y_true, scores, threshold: float) -> dict:
    yt, sc = _validate(y_true, scores)
    flagged = sc >= threshold
    tp = int(((flagged == 1) & (yt == 1)).sum())
    fp = int(((flagged == 1) & (yt == 0)).sum())
    tn = int(((flagged == 0) & (yt == 0)).sum())
    fn = int(((flagged == 0) & (yt == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    c = confusion_at(y_true, scores, threshold)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return precision, recall

def precision_at_k(y_true, scores, k: int) -> float:
    yt, sc = _validate(y_true, scores)
    if k <= 0:
        raise ValueError("k must be > 0")
    n = len(yt)
    k = min(k, n)
    df = pd.DataFrame({"y": yt, "s": sc})
    df_sorted = df.sort_values(by="s", ascending=False, kind="stable")
    top_k = df_sorted.head(k)
    pos_count = ((top_k["y"] == 1)).sum()
    return float(pos_count / k)

def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    c = confusion_at(y_true, scores, threshold)
    return float(c["fp"] * cost_fp + c["fn"] * cost_fn)

def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    yt, sc = _validate(y_true, scores)
    candidates = list(np.unique(sc)) + [math.inf]
    best_t = candidates[0]
    min_c = math.inf
    for t in candidates:
        cost = expected_cost(yt, sc, t, cost_fp=cost_fp, cost_fn=cost_fn)
        if cost < min_c:
            min_c = cost
            best_t = t
        elif cost == min_c:
            if t > best_t:
                best_t = t
    return float(best_t)

def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    yt, sc = _validate(y_true, scores)
    candidates = sorted(list(np.unique(sc)))
    valid_thresholds = []
    for t in candidates:
        p, _ = precision_recall_at(yt, sc, t)
        if p >= min_precision:
            valid_thresholds.append(t)
    if not valid_thresholds:
        return None
    return float(min(valid_thresholds))
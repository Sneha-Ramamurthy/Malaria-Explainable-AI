"""TensorFlow-free helpers for Step 3 (threshold selection and uncertainty), so they can be unit-tested."""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def youden_threshold(y_true, prob):
    """Decision threshold maximising Youden's J = TPR - FPR on (validation) data.

    Unlike a fixed percentile of the predictions (which forces a fixed share of positives whatever the model
    learned) or F1 (which depends on class prevalence), J is prevalence-independent.  Ties go to the higher
    threshold.  Falls back to 0.5 if only one class / constant predictions.  Result is always inside (0, 1).
    """
    y = np.asarray(y_true).astype(int); p = np.asarray(prob, float)
    if len(np.unique(y)) < 2 or np.ptp(p) == 0:
        return 0.5
    best_j, best_t = -np.inf, 0.5
    pos, neg = (y == 1).sum(), (y == 0).sum()
    for t in np.unique(p):
        pred = p >= t
        j = (pred & (y == 1)).sum() / pos - (pred & (y == 0)).sum() / neg
        if j >= best_j:
            best_j, best_t = j, float(t)
    return float(np.clip(best_t, 1e-6, 1 - 1e-6))


def block_bootstrap_ci(y_true, prob, thr, blocks, n_boot=500, seed=0, alpha=0.05):
    """95% (default) confidence intervals of patch metrics, resampling whole spatial BLOCKS with replacement
    (patches inside one block are strongly correlated, so resampling single patches would be over-confident).
    Returns {metric: {"low", "high"}, "n_boot_used": k}; draws containing one class are skipped."""
    y = np.asarray(y_true).astype(int); p = np.asarray(prob, float); b = np.asarray(blocks)
    ub = np.unique(b); members = {k: np.flatnonzero(b == k) for k in ub}
    rng = np.random.default_rng(seed)
    out = {m: [] for m in ("auc", "precision", "recall", "f1")}
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.choice(ub, size=len(ub), replace=True)])
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        pred = (pp >= thr).astype(int)
        out["auc"].append(roc_auc_score(yy, pp))
        out["precision"].append(precision_score(yy, pred, zero_division=0))
        out["recall"].append(recall_score(yy, pred, zero_division=0))
        out["f1"].append(f1_score(yy, pred, zero_division=0))
    res = {m: ({"low": float(np.quantile(v, alpha / 2)), "high": float(np.quantile(v, 1 - alpha / 2))} if v else None)
           for m, v in out.items()}
    res["n_boot_used"] = len(out["auc"])
    return res

"""
metrics.py
==========
Evaluation metrics for the Bengali Answer Evaluation System.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)


# ─── Bootstrap confidence interval helper ─────────────────────────────────────

def _bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                  stat_fn, n_boot: int = 1000,
                  ci: float = 0.95, seed: int = 42) -> tuple:
    """
    Return a (lo, hi) percentile bootstrap confidence interval for *stat_fn*.

    Parameters
    ----------
    y_true, y_pred : 1-D float arrays
    stat_fn        : callable(y_true, y_pred) -> scalar
    n_boot         : number of bootstrap resamples
    ci             : confidence level (default 0.95 → 95 % CI)
    seed           : random seed for reproducibility

    Returns
    -------
    (lo, hi) : lower and upper CI bounds
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            stats[i] = stat_fn(y_true[idx], y_pred[idx])
        except Exception:
            stats[i] = np.nan
    stats = stats[~np.isnan(stats)]
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


# ─── Core metric computation ───────────────────────────────────────────────────

def compute_full_metrics(y_true, y_pred, name: str,
                         n_boot: int = 1000) -> dict:
    """
    Compute regression and classification metrics with 95 % bootstrap CIs.

    Parameters
    ----------
    y_true, y_pred : array-like of float scores in [0, 100]
    name           : label used in display tables
    n_boot         : number of bootstrap resamples for CI estimation

    Returns
    -------
    dict with keys:
        name, pearson, spearman, mae, rmse, r2,
        within_10, within_15, within_20,
        cls_acc, cls_prec, cls_rec, cls_f1,
        pearson_ci_lo, pearson_ci_hi,
        mae_ci_lo,     mae_ci_hi,
        rmse_ci_lo,    rmse_ci_hi,
        f1_ci_lo,      f1_ci_hi,
        y_true_cls, y_pred_cls
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    p_corr, _ = pearsonr(y_true, y_pred)
    s_corr, _ = spearmanr(y_true, y_pred)
    mae        = mean_absolute_error(y_true, y_pred)
    rmse       = np.sqrt(mean_squared_error(y_true, y_pred))
    r2         = r2_score(y_true, y_pred)

    within_10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100
    within_15 = np.mean(np.abs(y_true - y_pred) <= 15) * 100
    within_20 = np.mean(np.abs(y_true - y_pred) <= 20) * 100

    y_true_cls = (y_true >= 50).astype(int)
    y_pred_cls = (y_pred >= 50).astype(int)

    cls_f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0) * 100

    # ── Bootstrap CIs ─────────────────────────────────────────────────────────
    def _pearson(yt, yp):
        return pearsonr(yt, yp)[0]

    def _mae(yt, yp):
        return mean_absolute_error(yt, yp)

    def _rmse(yt, yp):
        return np.sqrt(mean_squared_error(yt, yp))

    def _f1(yt, yp):
        return f1_score((yt >= 50).astype(int),
                        (yp >= 50).astype(int), zero_division=0) * 100

    p_lo,  p_hi  = _bootstrap_ci(y_true, y_pred, _pearson, n_boot)
    m_lo,  m_hi  = _bootstrap_ci(y_true, y_pred, _mae,     n_boot)
    r_lo,  r_hi  = _bootstrap_ci(y_true, y_pred, _rmse,    n_boot)
    f_lo,  f_hi  = _bootstrap_ci(y_true, y_pred, _f1,      n_boot)

    return {
        'name': name,
        'pearson':  p_corr, 'spearman': s_corr,
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'within_10': within_10, 'within_15': within_15, 'within_20': within_20,
        'cls_acc':  accuracy_score(y_true_cls, y_pred_cls) * 100,
        'cls_prec': precision_score(y_true_cls, y_pred_cls, zero_division=0) * 100,
        'cls_rec':  recall_score   (y_true_cls, y_pred_cls, zero_division=0) * 100,
        'cls_f1':   cls_f1,
        # 95 % bootstrap confidence intervals
        'pearson_ci_lo': p_lo, 'pearson_ci_hi': p_hi,
        'mae_ci_lo':     m_lo, 'mae_ci_hi':     m_hi,
        'rmse_ci_lo':    r_lo, 'rmse_ci_hi':    r_hi,
        'f1_ci_lo':      f_lo, 'f1_ci_hi':      f_hi,
        'y_true_cls': y_true_cls, 'y_pred_cls': y_pred_cls,
    }


# ─── Display helpers ──────────────────────────────────────────────────────────

def print_regression_table(all_metrics: list):
    """Print formatted tables for all metrics dicts in *all_metrics*."""
    print('\n' + '=' * 100)
    print('  REGRESSION METRICS (Test Set)')
    print('=' * 100)
    print(f"  {'Model':<38} | {'Pearson':>8} | {'Spearman':>8} | {'MAE':>6} | {'RMSE':>6} | {'R^2':>6}")
    print(f"  {'-'*38}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for m in all_metrics:
        print(f"  {m['name']:<38} | {m['pearson']:>8.4f} | {m['spearman']:>8.4f} "
              f"| {m['mae']:>6.2f} | {m['rmse']:>6.2f} | {m['r2']:>6.4f}")
    print('=' * 100)

    print('\n' + '=' * 78)
    print('  TOLERANCE-BASED ACCURACY')
    print('=' * 78)
    print(f"  {'Model':<38} | {'+/-10':>8} | {'+/-15':>8} | {'+/-20':>8}")
    print(f"  {'-'*38}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for m in all_metrics:
        print(f"  {m['name']:<38} | {m['within_10']:>7.1f}% | {m['within_15']:>7.1f}% | {m['within_20']:>7.1f}%")
    print('=' * 78)

    print('\n' + '=' * 85)
    print('  CLASSIFICATION METRICS (>=50 correct, <50 incorrect)')
    print('=' * 85)
    print(f"  {'Model':<38} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>8}")
    print(f"  {'-'*38}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}")
    for m in all_metrics:
        print(f"  {m['name']:<38} | {m['cls_acc']:>7.1f}% | {m['cls_prec']:>8.1f}% "
              f"| {m['cls_rec']:>7.1f}% | {m['cls_f1']:>7.1f}%")
    print('=' * 85)


def print_ci_table(all_metrics: list):
    """
    Print a 95 % bootstrap confidence interval summary for each model.
    Only shown for models that have CI keys (computed via compute_full_metrics).
    """
    has_ci = [m for m in all_metrics if 'pearson_ci_lo' in m]
    if not has_ci:
        print('[INFO] No confidence intervals available — run compute_full_metrics with n_boot > 0.')
        return

    print('\n' + '=' * 110)
    print('  95% BOOTSTRAP CONFIDENCE INTERVALS  (n_boot=1000, seed=42)')
    print('=' * 110)
    print(f"  {'Model':<38} | {'Pearson [lo, hi]':^22} | {'MAE [lo, hi]':^20} | {'RMSE [lo, hi]':^20} | {'F1 [lo, hi]':^20}")
    print(f"  {'-'*38}-+-{'-'*22}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")
    for m in has_ci:
        p_str = f"[{m['pearson_ci_lo']:.3f}, {m['pearson_ci_hi']:.3f}]"
        m_str = f"[{m['mae_ci_lo']:.2f}, {m['mae_ci_hi']:.2f}]"
        r_str = f"[{m['rmse_ci_lo']:.2f}, {m['rmse_ci_hi']:.2f}]"
        f_str = f"[{m['f1_ci_lo']:.1f}%, {m['f1_ci_hi']:.1f}%]"
        print(f"  {m['name']:<38} | {p_str:^22} | {m_str:^20} | {r_str:^20} | {f_str:^20}")
    print('=' * 110)

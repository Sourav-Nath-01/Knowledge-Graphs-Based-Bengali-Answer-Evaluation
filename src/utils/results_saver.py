"""
results_saver.py
================
Utility that saves all run outputs to a timestamped directory so they can be
used directly for report writing.

Saved files
-----------
    results/<timestamp>/
    ├── metrics_table.csv       – One row per model (regression + classification metrics)
    ├── cv_results.csv          – Per-fold CV metrics + mean±std summary row
    ├── predictions.csv         – Per-sample y_true / y_pred_raw / y_pred_final / overrides
    ├── feature_importances.csv – XGBoost feature importance ranking
    ├── run_config.json         – Dataset, epochs, seed, timestamp …
    ├── console_log.txt         – Full stdout captured during the run
    └── run_summary.md          – Auto-generated Markdown report (paste into your report)
"""

import io
import json
import os
import sys
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd


# ─── Stdout tee ───────────────────────────────────────────────────────────────

class _Tee:
    """Write to both the real stdout and an in-memory buffer simultaneously."""

    def __init__(self, stream):
        self._real   = stream
        self._buffer = io.StringIO()

    def write(self, data):
        self._real.write(data)
        self._buffer.write(data)

    def flush(self):
        self._real.flush()

    def getvalue(self):
        return self._buffer.getvalue()

    # Forward everything else to the real stream
    def __getattr__(self, attr):
        return getattr(self._real, attr)


# ─── ResultsSaver ─────────────────────────────────────────────────────────────

class ResultsSaver:
    """
    Collect run artefacts and flush them to ``output_dir`` when
    ``save_all()`` is called (or when ``__exit__`` is triggered if used as a
    context manager).

    Parameters
    ----------
    base_dir : str
        Parent directory for result folders.  A sub-folder named after the
        current timestamp is created automatically.
    label : str
        Optional label prepended to the folder name (e.g. ``"train"``).
    """

    FEATURE_NAMES = [
        'base_sim', 'graph_sim', 'penalty', 'entity_mm', 'coverage', 'neg_mm',
        'sim*pen', 'sim*ent', 'grph*pen', 'pen*ent', 'sim_diff', 'avg_sim',
        'max_pen', 'cov*sim', 'neg*sim',
    ]

    def __init__(self, base_dir: str = 'results', label: str = 'run'):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(base_dir, f'{label}_{ts}')
        os.makedirs(self.output_dir, exist_ok=True)

        # Internal state
        self._tee          = None
        self._metrics      = []          # list[dict] from compute_full_metrics
        self._cv_folds     = []          # list[dict] per fold
        self._cv_summary   = {}          # mean/std summary
        self._config       = {}
        self._pred_df      = None        # DataFrame

        print(f'[ResultsSaver] Outputs will be saved to: {self.output_dir}')

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.start_log_capture()
        return self

    def __exit__(self, *_):
        self.save_all()

    # ── Log capture ───────────────────────────────────────────────────────────

    def start_log_capture(self):
        """Tee stdout so every print() is also captured to console_log.txt."""
        if self._tee is None:
            self._tee = _Tee(sys.stdout)
            sys.stdout = self._tee

    def _stop_log_capture(self):
        if self._tee is not None:
            sys.stdout = self._tee._real   # restore original stdout
            log_text   = self._tee.getvalue()
            self._tee  = None
            return log_text
        return ''

    # ── Data registration ─────────────────────────────────────────────────────

    def set_config(self, config: dict):
        """Register run configuration (dataset path, epochs, etc.)."""
        self._config = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            **config,
        }

    def add_metrics(self, metrics_list: list):
        """
        Register a list of metric dicts produced by ``compute_full_metrics``.
        Can be called multiple times; results are appended.
        """
        self._metrics.extend(metrics_list)

    def add_cv_fold(self, fold: int, pearson: float, mae: float, f1: float):
        """Register results for one CV fold."""
        self._cv_folds.append({'fold': fold, 'pearson': pearson,
                               'mae': mae, 'f1_pct': f1})

    def add_cv_summary(self, pearsons, maes, f1s):
        """Register CV mean ± std summary vectors."""
        self._cv_summary = {
            'pearson_mean': float(np.mean(pearsons)),
            'pearson_std':  float(np.std(pearsons)),
            'mae_mean':     float(np.mean(maes)),
            'mae_std':      float(np.std(maes)),
            'f1_mean':      float(np.mean(f1s)),
            'f1_std':       float(np.std(f1s)),
        }

    def add_predictions(self, test_df: pd.DataFrame,
                        y_true, y_pred_raw, y_pred_final,
                        override_log=None):
        """
        Register per-sample predictions.

        Parameters
        ----------
        test_df       : DataFrame with at least ``question``, ``reference_answer``,
                        ``student_answer`` columns.
        y_true        : array of ground-truth scores
        y_pred_raw    : array of ensemble scores before overrides
        y_pred_final  : array of scores after hard overrides
        override_log  : list of lists of str reasons (one per sample), optional
        """
        n = len(y_true)
        if override_log is None:
            override_log = [[] for _ in range(n)]

        rows = []
        for i in range(n):
            row = {
                'idx':           i,
                'question':      test_df.iloc[i].get('question', ''),
                'reference':     test_df.iloc[i].get('reference_answer', ''),
                'student':       test_df.iloc[i].get('student_answer', ''),
                'y_true':        float(y_true[i]),
                'y_pred_raw':    float(y_pred_raw[i]),
                'y_pred_final':  float(y_pred_final[i]),
                'abs_error':     abs(float(y_true[i]) - float(y_pred_final[i])),
                'overrides':     '; '.join(override_log[i]) if override_log[i] else '',
            }
            rows.append(row)
        self._pred_df = pd.DataFrame(rows)

    def add_feature_importances(self, scorer):
        """
        Extract and store XGBoost feature importances from an ``AnswerScorer``.
        Safe to call even if the scorer is not trained.
        """
        if not getattr(scorer, 'is_trained', False):
            return
        imp  = scorer.xgb_model.feature_importances_
        n    = len(imp)
        names = self.FEATURE_NAMES[:n] + [f'feat_{i}' for i in range(n - len(self.FEATURE_NAMES))]
        self._feat_imp = pd.DataFrame({
            'feature':    names,
            'importance': imp,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_metrics_table(self):
        if not self._metrics:
            return
        rows = []
        for m in self._metrics:
            rows.append({
                'model':       m['name'],
                'pearson':     round(m['pearson'],  4),
                'spearman':    round(m['spearman'], 4),
                'mae':         round(m['mae'],      2),
                'rmse':        round(m['rmse'],     2),
                'r2':          round(m['r2'],       4),
                'within_10':   round(m['within_10'],1),
                'within_15':   round(m['within_15'],1),
                'within_20':   round(m['within_20'],1),
                'cls_acc':     round(m['cls_acc'],  1),
                'cls_prec':    round(m['cls_prec'], 1),
                'cls_rec':     round(m['cls_rec'],  1),
                'cls_f1':      round(m['cls_f1'],   1),
            })
        path = os.path.join(self.output_dir, 'metrics_table.csv')
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f'[ResultsSaver] Saved → {path}')

    def save_cv_results(self):
        if not self._cv_folds:
            return
        rows = [{'fold': f['fold'],
                 'pearson': round(f['pearson'], 4),
                 'mae':     round(f['mae'], 2),
                 'f1_pct':  round(f['f1_pct'], 1)}
                for f in self._cv_folds]
        if self._cv_summary:
            s = self._cv_summary
            rows.append({
                'fold':    'MEAN±STD',
                'pearson': f"{s['pearson_mean']:.4f}±{s['pearson_std']:.4f}",
                'mae':     f"{s['mae_mean']:.2f}±{s['mae_std']:.2f}",
                'f1_pct':  f"{s['f1_mean']:.1f}±{s['f1_std']:.1f}",
            })
        path = os.path.join(self.output_dir, 'cv_results.csv')
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f'[ResultsSaver] Saved → {path}')

    def save_predictions(self):
        if self._pred_df is None:
            return
        path = os.path.join(self.output_dir, 'predictions.csv')
        self._pred_df.to_csv(path, index=False)
        print(f'[ResultsSaver] Saved → {path}')

    def save_feature_importances(self):
        if not hasattr(self, '_feat_imp'):
            return
        path = os.path.join(self.output_dir, 'feature_importances.csv')
        self._feat_imp.to_csv(path, index=False)
        print(f'[ResultsSaver] Saved → {path}')

    def save_run_config(self):
        if not self._config:
            return
        path = os.path.join(self.output_dir, 'run_config.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
        print(f'[ResultsSaver] Saved → {path}')

    def save_console_log(self, log_text: str):
        path = os.path.join(self.output_dir, 'console_log.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(log_text)
        print(f'[ResultsSaver] Saved → {path}')
    def add_adversarial_breakdown(self, num_adversarial: int = 50):
        """
        Derive per-category adversarial metrics from the already-registered
        predictions.  Must be called after add_predictions().

        The 50 adversarial rows are assumed to follow this layout
        (matches _ADVERSARIAL_SAMPLES in dataset.py):
          rows  0-16 : Karak Reversal   (17 samples)
          rows 17-33 : Negation         (17 samples)
          rows 34-37 : Wrong Number     (4 samples)
          rows 38-39 : Wrong Location   (2 samples)
          rows 40    : Partial Cover    (1 sample)
          rows 41-49 : Voice Change     (9 samples)
        """
        if self._pred_df is None:
            return
        adv_df = self._pred_df.tail(min(num_adversarial, len(self._pred_df))).reset_index(drop=True)
        categories = [
            ('Karak Reversal',  list(range(0,  17))),
            ('Negation',        list(range(17, 34))),
            ('Wrong Number',    list(range(34, 38))),
            ('Wrong Location',  list(range(38, 40))),
            ('Partial Cover',   [40]),
            ('Voice Change',    list(range(41, min(num_adversarial, 50)))),
        ]
        rows = []
        for cat, idxs in categories:
            valid = [i for i in idxs if i < len(adv_df)]
            if not valid:
                continue
            sub = adv_df.iloc[valid]
            rows.append({
                'category':       cat,
                'n':              len(sub),
                'mean_true':      round(sub['y_true'].mean(), 2),
                'mean_pred_raw':  round(sub['y_pred_raw'].mean(), 2),
                'mean_pred_final':round(sub['y_pred_final'].mean(), 2),
                'mean_abs_error': round(sub['abs_error'].mean(), 2),
                'n_overridden':   int((sub['overrides'] != '').sum()),
            })
        self._adv_breakdown = pd.DataFrame(rows)

    def save_adversarial_breakdown(self):
        if not hasattr(self, '_adv_breakdown') or self._adv_breakdown.empty:
            return
        path = os.path.join(self.output_dir, 'adversarial_breakdown.csv')
        self._adv_breakdown.to_csv(path, index=False)
        print(f'[ResultsSaver] Saved → {path}')

    def add_error_analysis(self):
        """
        Detailed error analysis on registered predictions.
        Must be called after add_predictions().

        Produces:
          - error_analysis.csv      : per-sample error category + FP/FN flags
          - error_summary.csv       : category-level summary stats
          - worst_predictions.csv   : top-20 highest-error samples for qualitative review
        """
        if self._pred_df is None:
            return

        df = self._pred_df.copy()
        err = df['y_pred_final'] - df['y_true']

        # ── Error magnitude categories ────────────────────────────────────────
        def _cat(e):
            if   e >  20: return 'Large Overestimate (>+20)'
            elif e >   8: return 'Slight Overestimate (+8 to +20)'
            elif e >= -8: return 'Near-Correct (±8)'
            elif e >= -20: return 'Slight Underestimate (-8 to -20)'
            else:          return 'Large Underestimate (<-20)'

        df['error']          = err
        df['error_category'] = err.apply(_cat)

        # ── Classification errors (FP / FN) ───────────────────────────────────
        true_cls  = (df['y_true']        >= 50).astype(int)
        pred_cls  = (df['y_pred_final']  >= 50).astype(int)
        df['true_label'] = true_cls.map({1: 'PASS', 0: 'FAIL'})
        df['pred_label'] = pred_cls.map({1: 'PASS', 0: 'FAIL'})
        df['fp'] = ((pred_cls == 1) & (true_cls == 0)).astype(int)  # predicted PASS, actually FAIL
        df['fn'] = ((pred_cls == 0) & (true_cls == 1)).astype(int)  # predicted FAIL, actually PASS
        df['correct_cls'] = (pred_cls == true_cls).astype(int)

        self._error_df = df

        # ── Category summary ─────────────────────────────────────────────────
        cat_order = [
            'Large Overestimate (>+20)',
            'Slight Overestimate (+8 to +20)',
            'Near-Correct (±8)',
            'Slight Underestimate (-8 to -20)',
            'Large Underestimate (<-20)',
        ]
        summary_rows = []
        for cat in cat_order:
            sub = df[df['error_category'] == cat]
            if sub.empty:
                continue
            summary_rows.append({
                'error_category':     cat,
                'count':              len(sub),
                'pct':                round(len(sub) / len(df) * 100, 1),
                'mean_abs_error':     round(sub['abs_error'].mean(), 2),
                'mean_true_score':    round(sub['y_true'].mean(), 2),
                'mean_pred_score':    round(sub['y_pred_final'].mean(), 2),
                'fp_count':           int(sub['fp'].sum()),
                'fn_count':           int(sub['fn'].sum()),
            })

        # Add FP/FN totals row
        summary_rows.append({
            'error_category':  '— TOTAL False Positives (pred PASS, actual FAIL)',
            'count':           int(df['fp'].sum()),
            'pct':             round(df['fp'].mean() * 100, 1),
            'mean_abs_error':  round(df[df['fp']==1]['abs_error'].mean(), 2) if df['fp'].sum() else 0,
            'mean_true_score': round(df[df['fp']==1]['y_true'].mean(), 2) if df['fp'].sum() else 0,
            'mean_pred_score': round(df[df['fp']==1]['y_pred_final'].mean(), 2) if df['fp'].sum() else 0,
            'fp_count': int(df['fp'].sum()), 'fn_count': 0,
        })
        summary_rows.append({
            'error_category':  '— TOTAL False Negatives (pred FAIL, actual PASS)',
            'count':           int(df['fn'].sum()),
            'pct':             round(df['fn'].mean() * 100, 1),
            'mean_abs_error':  round(df[df['fn']==1]['abs_error'].mean(), 2) if df['fn'].sum() else 0,
            'mean_true_score': round(df[df['fn']==1]['y_true'].mean(), 2) if df['fn'].sum() else 0,
            'mean_pred_score': round(df[df['fn']==1]['y_pred_final'].mean(), 2) if df['fn'].sum() else 0,
            'fp_count': 0, 'fn_count': int(df['fn'].sum()),
        })

        self._error_summary = pd.DataFrame(summary_rows)
        self._worst_preds   = df.nlargest(20, 'abs_error')[[
            'idx', 'question', 'reference', 'student',
            'y_true', 'y_pred_final', 'error', 'error_category',
            'true_label', 'pred_label', 'overrides',
        ]].reset_index(drop=True)

    def save_error_analysis(self):
        if not hasattr(self, '_error_df'):
            return
        # Full per-sample error details
        path = os.path.join(self.output_dir, 'error_analysis.csv')
        cols = ['idx', 'y_true', 'y_pred_raw', 'y_pred_final', 'abs_error',
                'error', 'error_category', 'true_label', 'pred_label',
                'fp', 'fn', 'correct_cls', 'overrides']
        self._error_df[cols].to_csv(path, index=False)
        print(f'[ResultsSaver] Saved → {path}')

        # Category summary
        path2 = os.path.join(self.output_dir, 'error_summary.csv')
        self._error_summary.to_csv(path2, index=False)
        print(f'[ResultsSaver] Saved → {path2}')

        # Worst predictions (for qualitative analysis in report)
        path3 = os.path.join(self.output_dir, 'worst_predictions.csv')
        self._worst_preds.to_csv(path3, index=False)
        print(f'[ResultsSaver] Saved → {path3}')

        # Print summary to console
        print('\n[Error Analysis Summary]')
        print(self._error_summary[['error_category', 'count', 'pct', 'mean_abs_error']].to_string(index=False))

    def save_summary_markdown(self):
        """Generate a ready-to-paste Markdown summary for the project report."""
        lines = []
        ts    = self._config.get('timestamp', datetime.now().isoformat(timespec='seconds'))

        lines += [
            '# Bengali Answer Evaluation — Run Summary',
            '',
            f'**Generated:** {ts}  ',
            f'**Dataset:** {self._config.get("dataset", "N/A")}  ',
            f'**GNN epochs:** {self._config.get("epochs", "N/A")}  ',
            f'**Random seed:** {self._config.get("random_state", "N/A")}  ',
            '',
        ]

        # ── Metrics table ─────────────────────────────────────────────────────
        if self._metrics:
            lines += [
                '## Evaluation Results (Test Set)',
                '',
                '### Regression Metrics',
                '',
                '| Model | Pearson | Spearman | MAE | RMSE | R² |',
                '|-------|---------|----------|-----|------|----|',
            ]
            for m in self._metrics:
                lines.append(
                    f"| {m['name']} | {m['pearson']:.4f} | {m['spearman']:.4f} "
                    f"| {m['mae']:.2f} | {m['rmse']:.2f} | {m['r2']:.4f} |"
                )
            lines += [
                '',
                '### Tolerance-Based Accuracy',
                '',
                '| Model | ±10% | ±15% | ±20% |',
                '|-------|------|------|------|',
            ]
            for m in self._metrics:
                lines.append(
                    f"| {m['name']} | {m['within_10']:.1f}% "
                    f"| {m['within_15']:.1f}% | {m['within_20']:.1f}% |"
                )
            lines += [
                '',
                '### Classification Metrics (≥50 = correct)',
                '',
                '| Model | Accuracy | Precision | Recall | F1 |',
                '|-------|----------|-----------|--------|----|',
            ]
            for m in self._metrics:
                lines.append(
                    f"| {m['name']} | {m['cls_acc']:.1f}% | {m['cls_prec']:.1f}% "
                    f"| {m['cls_rec']:.1f}% | {m['cls_f1']:.1f}% |"
                )
            lines.append('')

        # ── CV results ────────────────────────────────────────────────────────
        if self._cv_folds:
            lines += [
                '## 5-Fold Cross-Validation Results',
                '',
                '| Fold | Pearson | MAE | F1 (%) |',
                '|------|---------|-----|--------|',
            ]
            for f in self._cv_folds:
                lines.append(
                    f"| {f['fold']} | {f['pearson']:.4f} | {f['mae']:.2f} | {f['f1_pct']:.1f} |"
                )
            if self._cv_summary:
                s = self._cv_summary
                lines += [
                    f"| **Mean±Std** | **{s['pearson_mean']:.4f}±{s['pearson_std']:.4f}** "
                    f"| **{s['mae_mean']:.2f}±{s['mae_std']:.2f}** "
                    f"| **{s['f1_mean']:.1f}±{s['f1_std']:.1f}** |",
                    '',
                ]

        # ── Feature importances ───────────────────────────────────────────────
        if hasattr(self, '_feat_imp'):
            lines += [
                '## XGBoost Feature Importances (Top 10)',
                '',
                '| Rank | Feature | Importance |',
                '|------|---------|------------|',
            ]
            for rank, row in self._feat_imp.head(10).iterrows():
                lines.append(f"| {rank+1} | {row['feature']} | {row['importance']:.4f} |")
            lines.append('')

        # ── Prediction error analysis ─────────────────────────────────────────
        if self._pred_df is not None:
            df   = self._pred_df
            n_ov = (df['overrides'] != '').sum()
            lines += [
                '## Prediction Error Analysis',
                '',
                f'- **Test samples:** {len(df)}',
                f'- **Overrides applied:** {n_ov}',
                f"- **Mean absolute error (final):** {df['abs_error'].mean():.2f}",
                f"- **Samples within ±10:** {(df['abs_error'] <= 10).sum()} "
                f"({(df['abs_error'] <= 10).mean()*100:.1f}%)",
                f"- **Samples within ±20:** {(df['abs_error'] <= 20).sum()} "
                f"({(df['abs_error'] <= 20).mean()*100:.1f}%)",
                '',
                '> Full per-sample predictions are in `predictions.csv`',
                '',
            ]

        lines += [
            '---',
            '',
            '*This file was auto-generated by `ResultsSaver`. '
            'Do not edit — re-run the script to regenerate.*',
        ]

        path = os.path.join(self.output_dir, 'run_summary.md')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'[ResultsSaver] Saved → {path}')

    # ── Master save ───────────────────────────────────────────────────────────

    def save_all(self, num_adversarial: int = 50, generate_plots: bool = True):
        """Stop log capture and flush everything to disk, then generate figures."""
        log_text = self._stop_log_capture()
        self.save_metrics_table()
        self.save_cv_results()
        self.save_predictions()
        self.save_feature_importances()
        self.add_adversarial_breakdown(num_adversarial=num_adversarial)
        self.save_adversarial_breakdown()
        self.add_error_analysis()
        self.save_error_analysis()
        self.save_run_config()
        self.save_summary_markdown()
        if log_text:
            self.save_console_log(log_text)

        # Auto-generate figures
        if generate_plots:
            try:
                from plot_results import generate_all_plots
                generate_all_plots(self.output_dir,
                                   num_adversarial=num_adversarial)
            except Exception as e:
                print(f'[ResultsSaver] Plot generation skipped: {e}')

        print(f'\n[ResultsSaver] All outputs saved to: {self.output_dir}')
        return self.output_dir

"""
plot_results.py
===============
Auto-generates publication-quality figures from a results directory
created by ResultsSaver.

Produced figures
----------------
  figures/
  ├── ablation_bar.png          – Pearson / F1 per model variant
  ├── pred_vs_true_scatter.png  – Predicted vs true score, coloured by sample type
  ├── confusion_matrix.png      – Binary correct/incorrect heatmap
  └── adversarial_breakdown.png – Metrics per adversarial error category

Usage
-----
    python plot_results.py --results-dir results/train_20260409_023500
    python plot_results.py --results-dir results/train_20260409_023500 --save-pdf
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe on all platforms
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


# ─── Colour palette ───────────────────────────────────────────────────────────
PALETTE     = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']
NORMAL_CLR  = '#4C72B0'
ADV_CLR     = '#C44E52'
GRID_CLR    = '#E8E8E8'


def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_facecolor('#F8F9FA')
    ax.grid(axis='y', color=GRID_CLR, linewidth=1.2, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)


# ─── 1. Ablation bar chart ─────────────────────────────────────────────────────

def plot_ablation(metrics_csv: str, out_path: str):
    """
    Grouped bar chart comparing Pearson correlation and F1 (%) across all
    model variants in *metrics_csv*.
    """
    df = pd.read_csv(metrics_csv)
    models  = df['model'].tolist()
    pearson = df['pearson'].tolist()
    f1      = (df['cls_f1'] / 100.0).tolist()   # normalise to [0,1] for shared axis

    x   = np.arange(len(models))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(max(9, len(models) * 1.6), 5.5))

    bars_p = ax.bar(x - w/2, pearson, w, label='Pearson r', color=PALETTE[0],
                    zorder=3, edgecolor='white', linewidth=0.8)
    bars_f = ax.bar(x + w/2, f1,      w, label='F1 (normalised)', color=PALETTE[1],
                    zorder=3, edgecolor='white', linewidth=0.8)

    for bar in list(bars_p) + list(bars_f):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=8, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.set_ylim(0, min(1.15, max(pearson + f1) + 0.12))
    ax.legend(framealpha=0.9, fontsize=10)
    _style_ax(ax,
              title='Model Ablation Study — Pearson Correlation & F1',
              xlabel='Model Variant', ylabel='Score')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] Saved → {out_path}')


# ─── 2. Predicted vs True scatter ─────────────────────────────────────────────

def plot_scatter(predictions_csv: str, out_path: str, num_adversarial: int = 50):
    """
    Scatter plot of predicted vs true score.
     • Normal samples  → blue dots
     • Adversarial     → red triangles (last *num_adversarial* rows)
    """
    df = pd.read_csv(predictions_csv)
    n  = len(df)

    mask_adv  = np.zeros(n, dtype=bool)
    mask_adv[-min(num_adversarial, n):] = True
    mask_norm = ~mask_adv

    y_true  = df['y_true'].values
    y_pred  = df['y_pred_final'].values

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true[mask_norm], y_pred[mask_norm],
               c=NORMAL_CLR, alpha=0.5, s=30, label='Normal', zorder=3)
    ax.scatter(y_true[mask_adv],  y_pred[mask_adv],
               c=ADV_CLR, alpha=0.8, s=50, marker='^', label='Adversarial', zorder=4)

    lims = [0, 105]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.4, label='Perfect prediction')
    ax.fill_between(lims, [l - 10 for l in lims], [l + 10 for l in lims],
                    alpha=0.07, color='grey', label='±10 band')

    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.legend(fontsize=9, framealpha=0.9)
    _style_ax(ax,
              title='Predicted vs True Score',
              xlabel='Human Score (true)', ylabel='Predicted Score')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] Saved → {out_path}')


# ─── 3. Confusion matrix ──────────────────────────────────────────────────────

def plot_confusion_matrix(predictions_csv: str, out_path: str, threshold: float = 50.0):
    """
    2×2 confusion matrix for binary correct (≥50) / incorrect (<50) classification.
    """
    df = pd.read_csv(predictions_csv)
    y_true_cls = (df['y_true']       >= threshold).astype(int).values
    y_pred_cls = (df['y_pred_final'] >= threshold).astype(int).values

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_cls, y_pred_cls)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    cmap = LinearSegmentedColormap.from_list('bw', ['#FFFFFF', PALETTE[0]])
    im = ax.imshow(cm, cmap=cmap, vmin=0)

    labels = ['Incorrect (<50)', 'Correct (≥50)']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9, rotation=45)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title('Classification Confusion Matrix\n(correct ≥50 threshold)',
                 fontsize=12, fontweight='bold', pad=10)

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f'{v}\n({v/total*100:.1f}%)',
                    ha='center', va='center', fontsize=12,
                    color='white' if v > total * 0.3 else '#333333')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] Saved → {out_path}')


# ─── 4. Adversarial-type breakdown ────────────────────────────────────────────

_ADV_CATEGORIES = {
    'Karak Reversal':  list(range(0, 17)),    # first 17 adversarial rows
    'Negation':        list(range(17, 34)),   # next 17
    'Wrong Number':    list(range(34, 38)),   # 4 rows
    'Wrong Location':  list(range(38, 40)),   # 2 rows
    'Partial Cover':   list(range(40, 41)),   # 1 row
    'Voice Change':    list(range(41, 50)),   # remaining
}


def plot_adversarial_breakdown(predictions_csv: str, out_path: str,
                                num_adversarial: int = 50):
    """
    Bar chart showing mean absolute error and mean predicted score
    for each adversarial error category.
    """
    df     = pd.read_csv(predictions_csv)
    adv_df = df.tail(min(num_adversarial, len(df))).reset_index(drop=True)

    rows = []
    for cat, indices in _ADV_CATEGORIES.items():
        valid = [i for i in indices if i < len(adv_df)]
        if not valid:
            continue
        sub = adv_df.iloc[valid]
        rows.append({
            'category':   cat,
            'count':      len(sub),
            'mean_error': sub['abs_error'].mean(),
            'mean_pred':  sub['y_pred_final'].mean(),
            'mean_true':  sub['y_true'].mean(),
        })
    if not rows:
        print('[Plot] No adversarial data found — skipping adversarial breakdown.')
        return

    bdf   = pd.DataFrame(rows)
    cats  = bdf['category'].tolist()
    x     = np.arange(len(cats))
    w     = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # — Left: Mean MAE per category —
    bars = ax1.bar(x, bdf['mean_error'], color=PALETTE[:len(cats)],
                   zorder=3, edgecolor='white')
    for bar, row in zip(bars, rows):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f'{bar.get_height():.1f}\n(n={row["count"]})',
                 ha='center', va='bottom', fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=25, ha='right', fontsize=9)
    _style_ax(ax1, title='Mean Absolute Error by Adversarial Type',
              xlabel='Error Category', ylabel='MAE (score points)')

    # — Right: Mean predicted vs true per category —
    ax2.bar(x - w/2, bdf['mean_true'], w, label='True score',
            color=NORMAL_CLR, zorder=3, edgecolor='white')
    ax2.bar(x + w/2, bdf['mean_pred'], w, label='Predicted score',
            color=ADV_CLR, zorder=3, edgecolor='white', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, rotation=25, ha='right', fontsize=9)
    ax2.legend(fontsize=9, framealpha=0.9)
    _style_ax(ax2, title='True vs Predicted Score by Adversarial Type',
              xlabel='Error Category', ylabel='Score')

    fig.suptitle('Adversarial Sample Analysis', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] Saved → {out_path}')


# ─── 5. CV fold chart ─────────────────────────────────────────────────────────

def plot_cv_folds(cv_csv: str, out_path: str):
    """Line chart showing Pearson and MAE across CV folds (if CV data available)."""
    df = pd.read_csv(cv_csv)
    # Only keep numeric fold rows (exclude the MEAN±STD summary row)
    df_folds = df[pd.to_numeric(df['fold'], errors='coerce').notna()].copy()
    if df_folds.empty:
        print('[Plot] No numeric fold data — skipping CV folds plot.')
        return
    df_folds['fold'] = df_folds['fold'].astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ax1.plot(df_folds['fold'], df_folds['pearson'], 'o-',
             color=PALETTE[0], linewidth=2, markersize=7)
    ax1.axhline(df_folds['pearson'].mean(), linestyle='--',
                color=PALETTE[0], alpha=0.5, label=f"Mean = {df_folds['pearson'].mean():.4f}")
    ax1.set_xticks(df_folds['fold'])
    ax1.legend(fontsize=9)
    _style_ax(ax1, title='Pearson r across CV Folds',
              xlabel='Fold', ylabel='Pearson r')

    ax2.plot(df_folds['fold'], df_folds['mae'], 's-',
             color=PALETTE[1], linewidth=2, markersize=7)
    ax2.axhline(df_folds['mae'].mean(), linestyle='--',
                color=PALETTE[1], alpha=0.5, label=f"Mean = {df_folds['mae'].mean():.2f}")
    ax2.set_xticks(df_folds['fold'])
    ax2.legend(fontsize=9)
    _style_ax(ax2, title='MAE across CV Folds',
              xlabel='Fold', ylabel='MAE (score points)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] Saved → {out_path}')


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Bengali Answer Eval — Plot Generator')
    p.add_argument('--results-dir', type=str, required=True,
                   help='Path to a timestamped run directory (e.g. results/train_20260409_023500)')
    p.add_argument('--num-adversarial', type=int, default=50)
    p.add_argument('--save-pdf', action='store_true',
                   help='Also save a .pdf alongside each .png')
    return p.parse_args()


def generate_all_plots(results_dir: str, num_adversarial: int = 50,
                       save_pdf: bool = False):
    """
    Generate all available plots for a given results directory.
    Skips any plot whose source CSV is missing.

    Can be called programmatically (e.g. from ResultsSaver).
    """
    fig_dir = os.path.join(results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    def _paths(stem):
        png = os.path.join(fig_dir, f'{stem}.png')
        pdf = os.path.join(fig_dir, f'{stem}.pdf') if save_pdf else None
        return png, pdf

    def _save_both(fn, *args):
        png, pdf = _paths(fn.__name__.replace('plot_', ''))
        fn(*args, png)
        if pdf:
            fn(*args, pdf)

    metrics_csv     = os.path.join(results_dir, 'metrics_table.csv')
    predictions_csv = os.path.join(results_dir, 'predictions.csv')
    cv_csv          = os.path.join(results_dir, 'cv_results.csv')

    if os.path.exists(metrics_csv):
        plot_ablation(metrics_csv, os.path.join(fig_dir, 'ablation_bar.png'))

    if os.path.exists(predictions_csv):
        plot_scatter(predictions_csv,
                     os.path.join(fig_dir, 'pred_vs_true_scatter.png'),
                     num_adversarial=num_adversarial)
        plot_confusion_matrix(predictions_csv,
                              os.path.join(fig_dir, 'confusion_matrix.png'))
        plot_adversarial_breakdown(predictions_csv,
                                   os.path.join(fig_dir, 'adversarial_breakdown.png'),
                                   num_adversarial=num_adversarial)

    if os.path.exists(cv_csv):
        plot_cv_folds(cv_csv, os.path.join(fig_dir, 'cv_folds.png'))

    print(f'\n[Plot] All figures saved to: {fig_dir}')
    return fig_dir


def main():
    args = parse_args()
    if not os.path.isdir(args.results_dir):
        raise FileNotFoundError(f'Results directory not found: {args.results_dir}')
    generate_all_plots(args.results_dir,
                       num_adversarial=args.num_adversarial,
                       save_pdf=args.save_pdf)


if __name__ == '__main__':
    main()

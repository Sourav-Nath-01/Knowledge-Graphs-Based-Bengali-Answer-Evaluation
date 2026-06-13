"""
run_all_datasets.py
===================
Runs the full training + evaluation pipeline on all three datasets
(single-sentence, medium, long) one by one, saves separate models and
results for each, then prints a cross-dataset comparison table.

Usage
-----
    python run_all_datasets.py
    python run_all_datasets.py --epochs 10 --skip-cv
    python run_all_datasets.py --datasets data/dataset_single_sentence.csv data/dataset_medium_sentences.csv

Output
------
    models/
        single/    <- GNN + scorer for single-sentence
        medium/    <- GNN + scorer for medium-sentences
        long/      <- GNN + scorer for long-sentences

    results/
        single/    <- metrics, predictions, figures for single-sentence
        medium/    <- ...
        long/      <- ...

    results/comparison_all_datasets.csv   <- side-by-side summary
"""

import argparse
import os
import subprocess
import sys
import time

import pandas as pd


# ─── Dataset configuration ────────────────────────────────────────────────────

DEFAULT_DATASETS = [
    {
        'name':    'single',
        'label':   'Single-Sentence',
        'csv':     'data/dataset_single_sentence.csv',
        'out_dir': 'models/single',
        'results': 'results/single',
    },
    {
        'name':    'medium',
        'label':   'Medium-Sentences',
        'csv':     'data/dataset_medium_sentences.csv',
        'out_dir': 'models/medium',
        'results': 'results/medium',
    },
    {
        'name':    'long',
        'label':   'Long-Sentences',
        'csv':     'data/dataset_long_sentences.csv',
        'out_dir': 'models/long',
        'results': 'results/long',
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h}h {m}m {s}s' if h else f'{m}m {s}s'


def run_dataset(dataset_cfg: dict, epochs: int, skip_cv: bool,
                offline: bool = False) -> dict:
    """
    Run train.py then evaluate.py for one dataset configuration.
    Returns a dict with timing + the path to the results directory.
    """
    name    = dataset_cfg['name']
    csv     = dataset_cfg['csv']
    out_dir = dataset_cfg['out_dir']
    results = dataset_cfg['results']
    label   = dataset_cfg['label']

    if not os.path.exists(csv):
        print(f'\n[SKIP] Dataset not found: {csv}')
        return {'name': name, 'label': label, 'status': 'skipped', 'results_dir': None}

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(results, exist_ok=True)

    print(f'\n{"="*70}')
    print(f'  Running: {label}')
    print(f'  Dataset : {csv}')
    print(f'  Models  : {out_dir}/')
    print(f'  Results : {results}/')
    print(f'{"="*70}\n')

    # Dataset-specific model paths (matching evaluate.py naming convention)
    import os as _os
    ds_stem      = _os.path.splitext(_os.path.basename(csv))[0]
    gnn_path     = f'nlp_dominator_gnn_{ds_stem}.pth'
    scorer_path  = f'nlp_dominator_scorer_{ds_stem}.pkl'

    env = os.environ.copy()
    if offline:
        env['HF_HUB_OFFLINE']       = '1'
        env['TRANSFORMERS_OFFLINE'] = '1'

    t0 = time.time()

    # ── Step 1: Train ─────────────────────────────────────────────────────────
    train_cmd = [
        sys.executable, 'train.py',
        '--dataset',        csv,
        '--epochs',         str(epochs),
        '--output-dir',     out_dir,
        '--output-results', results,
    ]
    if offline:
        train_cmd.append('--offline')

    print(f'[{label}] Step 1/2: Training...')
    train_proc = subprocess.run(train_cmd, text=True, env=env)
    if train_proc.returncode != 0:
        elapsed = time.time() - t0
        print(f'\n[{label}] Training FAILED (exit {train_proc.returncode})')
        return {
            'name': name, 'label': label,
            'status': f'FAILED (train exit {train_proc.returncode})',
            'elapsed': _fmt_time(elapsed), 'results_dir': None,
        }

    # ── Step 2: Evaluate (uses dataset-specific fine-tuned models) ───────────
    eval_cmd = [
        sys.executable, 'evaluate.py',
        '--dataset',        csv,
        '--gnn-model',      gnn_path,
        '--scorer-model',   scorer_path,
        '--output-results', results,
    ]
    if skip_cv:
        eval_cmd.append('--skip-cv')
    if offline:
        eval_cmd.append('--offline') if '--offline' in train_cmd else None

    print(f'[{label}] Step 2/2: Evaluating...')
    eval_proc = subprocess.run(eval_cmd, text=True, env=env)

    elapsed = time.time() - t0
    status = 'ok' if eval_proc.returncode == 0 else f'FAILED (eval exit {eval_proc.returncode})'
    print(f'\n[{label}] Finished in {_fmt_time(elapsed)} — {status}')

    # Find the newest timestamped sub-dir
    results_dir = None
    if os.path.isdir(results):
        sub_dirs = sorted(
            [d for d in os.listdir(results) if os.path.isdir(os.path.join(results, d))],
            reverse=True,
        )
        if sub_dirs:
            results_dir = os.path.join(results, sub_dirs[0])

    return {
        'name':        name,
        'label':       label,
        'status':      status,
        'elapsed':     _fmt_time(elapsed),
        'results_dir': results_dir,
    }


def build_comparison(run_results: list) -> pd.DataFrame:
    """
    Load metrics_table.csv from each results dir and merge them
    into a single cross-dataset comparison DataFrame.
    """
    rows = []
    for r in run_results:
        if r['status'] != 'ok' or r['results_dir'] is None:
            continue
        metrics_csv = os.path.join(r['results_dir'], 'metrics_table.csv')
        if not os.path.exists(metrics_csv):
            continue
        df = pd.read_csv(metrics_csv)
        # Keep only the best model row (Proposed / Full + Overrides)
        best = df[df['model'].str.contains('Proposed|Override', case=False, regex=True)]
        if best.empty:
            best = df.tail(1)
        row = best.iloc[0].to_dict()
        row['dataset'] = r['label']
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    comparison = pd.DataFrame(rows)
    # Reorder columns
    front_cols = ['dataset', 'model', 'pearson', 'spearman', 'mae', 'rmse',
                  'within_10', 'within_15', 'within_20', 'cls_f1']
    other_cols = [c for c in comparison.columns if c not in front_cols]
    comparison = comparison[[c for c in front_cols if c in comparison.columns] + other_cols]
    return comparison


def print_comparison(df: pd.DataFrame):
    if df.empty:
        return
    print(f'\n{"="*90}')
    print('  CROSS-DATASET COMPARISON — Proposed Model (Full + Overrides)')
    print(f'{"="*90}')
    print(f"  {'Dataset':<22} | {'Pearson':>8} | {'Spearman':>8} | "
          f"{'MAE':>6} | {'RMSE':>6} | {'±10%':>6} | {'F1':>6}")
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for _, row in df.iterrows():
        print(f"  {row.get('dataset',''):<22} | "
              f"{row.get('pearson', 0):>8.4f} | "
              f"{row.get('spearman', 0):>8.4f} | "
              f"{row.get('mae', 0):>6.2f} | "
              f"{row.get('rmse', 0):>6.2f} | "
              f"{row.get('within_10', 0):>5.1f}% | "
              f"{row.get('cls_f1', 0):>5.1f}%")
    print(f'{"="*90}')


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Run training on all 3 datasets and compare results.'
    )
    p.add_argument('--datasets', nargs='+', default=None,
                   help='Override dataset CSV paths (space-separated). '
                        'Default: all 3 standard datasets.')
    p.add_argument('--epochs', type=int, default=5,
                   help='GNN training epochs per dataset (default: 5)')
    p.add_argument('--skip-cv', action='store_true',
                   help='Pass --skip-cv to evaluate.py (not used for train.py)')
    p.add_argument('--offline', action='store_true',
                   help='Load all HuggingFace models from local cache only '
                        '(no internet). Use after a first successful online run.')
    return p.parse_args()


def main():
    args = parse_args()

    # Build dataset list
    if args.datasets:
        datasets = []
        for csv_path in args.datasets:
            name = os.path.splitext(os.path.basename(csv_path))[0]
            datasets.append({
                'name':    name,
                'label':   name.replace('_', ' ').title(),
                'csv':     csv_path,
                'out_dir': f'models/{name}',
                'results': f'results/{name}',
            })
    else:
        datasets = DEFAULT_DATASETS

    print(f'\n{"#"*70}')
    print(f'  Bengali Answer Evaluation — Multi-Dataset Run')
    print(f'  Datasets : {len(datasets)}')
    print(f'  Epochs   : {args.epochs}')
    print(f'{"#"*70}')

    total_start = time.time()
    run_results = []

    for cfg in datasets:
        result = run_dataset(cfg, epochs=args.epochs, skip_cv=args.skip_cv,
                             offline=args.offline)
        run_results.append(result)

    total_elapsed = time.time() - total_start

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f'\n\n{"#"*70}')
    print(f'  ALL RUNS COMPLETE  (total: {_fmt_time(total_elapsed)})')
    print(f'{"#"*70}')
    for r in run_results:
        status_icon = '✅' if r['status'] == 'ok' else ('⏭️' if r['status'] == 'skipped' else '❌')
        elapsed_str = r.get('elapsed', '')
        print(f'  {status_icon}  {r["label"]:25s}  {r["status"]:15s}  {elapsed_str}')

    # ── Cross-dataset comparison table ──────────────────────────────────────
    comparison_df = build_comparison(run_results)
    print_comparison(comparison_df)

    # Save comparison CSV
    if not comparison_df.empty:
        out_path = os.path.join('results', 'comparison_all_datasets.csv')
        os.makedirs('results', exist_ok=True)
        comparison_df.to_csv(out_path, index=False)
        print(f'\n[OK] Comparison table saved → {out_path}')

    # ── Results directory listing ─────────────────────────────────────────
    print(f'\n  Models saved:')
    for cfg in datasets:
        if os.path.isdir(cfg['out_dir']):
            files = os.listdir(cfg['out_dir'])
            print(f'    {cfg["out_dir"]:30s}  {files}')

    print()


if __name__ == '__main__':
    main()

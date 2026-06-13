"""
train.py
========
End-to-end training script for the Bengali Answer Evaluation System.

Usage
-----
    python train.py [--dataset PATH] [--epochs N] [--output-dir DIR]

This script reproduces the full training pipeline from the original notebook:
  1. Loads and augments the dataset
  2. Initialises all NLP/graph components
  3. Extracts features for train and test sets
  4. Trains the Siamese GAT-GNN (epochs controlled by --epochs)
  5. Trains the XGBoost + MLP ensemble scorer
  6. Applies hard overrides on the test set
  7. Prints evaluation metrics
  8. Saves trained models to --output-dir
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from transformers import pipeline as hf_pipeline

from src.config import (
    GNN_EPOCHS, NUM_ADVERSARIAL,
    RANDOM_STATE, TEST_SIZE, GNN_MODEL_PATH, SCORER_MODEL_PATH,
)
from src.data.dataset import find_dataset_csv, load_and_augment_dataset, split_dataset
from src.evaluation.metrics import compute_full_metrics, print_regression_table, print_ci_table
from src.evaluation.overrides import apply_hard_overrides, compute_wrong_num_flags
from src.features.pipeline import (
    create_graphs_and_features, get_gnn_similarities, train_gnn,
)
from src.graph.embedder import BanglaBERTEmbedder
from src.graph.kg_constructor import KnowledgeGraphConstructor
from src.models.answer_scorer import AnswerScorer
from src.models.siamese_gnn import SiameseGNN
from src.nlp.coreference import BengaliCoreferenceResolver
from src.nlp.dependency_parser import BengaliDependencyParser
from src.nlp.triple_extractor import TripleExtractor
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.utils.helpers import compute_rouge_l, setup_bengali_font
from src.utils.results_saver import ResultsSaver
from src.validation.karak_validator import KarakValidator

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Bengali Answer Evaluation — Training')
    parser.add_argument('--dataset',        type=str, default=None,
                        help='Path to the original dataset CSV (auto-detected if omitted)')
    parser.add_argument('--epochs',         type=int, default=GNN_EPOCHS,
                        help=f'GNN training epochs (default: {GNN_EPOCHS})')
    parser.add_argument('--output-dir',     type=str, default='models',
                        help='Directory to save trained models (default: models/)')
    parser.add_argument('--output-results', type=str, default='results',
                        help='Parent directory for run result folders (default: results)')
    parser.add_argument('--offline',        action='store_true',
                        help='Force offline mode — use only locally cached HuggingFace models')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Offline mode: use only cached HuggingFace models ────────────────────
    if args.offline:
        os.environ['HF_HUB_OFFLINE']       = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        print('[INFO] Offline mode — using locally cached models only.')

    setup_bengali_font()

    # ── Results saver (starts log capture immediately) ────────────────────────
    saver = ResultsSaver(base_dir=args.output_results, label='train')
    saver.start_log_capture()

    # ── 1. Dataset ──────────────────────────────────────────────────────────
    print('\n' + '=' * 57)
    print('  Bengali Answer Evaluation (Final — BNLP, no Stanza)')
    print('=' * 57)

    # Resolve the source CSV first so we can derive the augmented path from it.
    # This ensures each dataset gets its OWN augmented file:
    #   dataset_single_sentence.csv   → dataset_single_sentence_augmented.csv
    #   dataset_medium_sentences.csv  → dataset_medium_sentences_augmented.csv
    # (prevents stale augmented data being reused when switching datasets)
    original_csv = args.dataset or find_dataset_csv()
    src_stem      = os.path.splitext(os.path.basename(original_csv))[0]  # e.g. "dataset_single_sentence"
    src_dir       = os.path.dirname(os.path.abspath(original_csv))
    augmented_csv = os.path.join(src_dir, f'{src_stem}_augmented.csv')

    if not os.path.exists(augmented_csv):
        print(f'\n[OK] Dataset found at: {original_csv}')
        print(f'     Augmented will be written to: {augmented_csv}')
        df_augmented = load_and_augment_dataset(original_csv, augmented_csv)
    else:
        df_augmented = pd.read_csv(augmented_csv)
        print(f'\nAugmented dataset loaded: {augmented_csv}')
        print(f'  ({len(df_augmented)} samples, source: {os.path.basename(original_csv)})')

    train_df, test_df = split_dataset(df_augmented, num_adversarial=NUM_ADVERSARIAL,
                                      test_size=TEST_SIZE, random_state=RANDOM_STATE)
    n_adv_test = min(NUM_ADVERSARIAL, len(test_df))
    print(f'  Normal: {len(df_augmented) - NUM_ADVERSARIAL} | Adversarial: {NUM_ADVERSARIAL}')
    print(f'  Train: {len(train_df)} | Test: {len(test_df)} ({n_adv_test} adversarial)')

    # ── 2. NER pipeline ─────────────────────────────────────────────────────
    print('\n[1/7] Loading NER pipeline...')
    ner_pipe = None
    _ner_model = 'Davlan/bert-base-multilingual-cased-ner-hrl'
    for local_only in (True, False):
        try:
            ner_pipe = hf_pipeline('ner', model=_ner_model,
                                    aggregation_strategy='simple', device=-1,
                                    model_kwargs={'local_files_only': local_only})
            src = 'cache' if local_only else 'download'
            print(f'  [OK] NER pipeline loaded from {src}.')
            break
        except Exception as e:
            if local_only:
                continue      # try download next
            print(f'  [WARN] NER unavailable: {e}')

    # ── 3. Components ────────────────────────────────────────────────────────
    print('\n[2/7] Initializing components...')
    text_processor   = TextPreprocessor()
    dep_parser       = BengaliDependencyParser()
    coref_resolver   = BengaliCoreferenceResolver()
    triple_extractor = TripleExtractor()
    kg_constructor   = KnowledgeGraphConstructor()
    embedder         = BanglaBERTEmbedder()
    validator        = KarakValidator()

    # ── 4. Training features ─────────────────────────────────────────────────
    print('\n[3/7] Extracting TRAINING features...')
    (bsims_tr, rpyg_tr, spyg_tr, pen_tr, emm_tr, cov_tr, neg_tr, rouge_tr) = \
        create_graphs_and_features(
            train_df, text_processor, dep_parser, triple_extractor,
            kg_constructor, embedder, validator,
            coref_resolver=coref_resolver, ner_pipeline_obj=ner_pipe)
    y_train = train_df['human_score'].values.astype(float)

    # ── 5. Train GAT-GNN ─────────────────────────────────────────────────────
    print(f'\n[4/7] Training Siamese GAT-GNN ({args.epochs} epochs)...')
    gnn = SiameseGNN()
    gnn = train_gnn(gnn, rpyg_tr, spyg_tr, y_train, epochs=args.epochs)
    gsims_tr = get_gnn_similarities(gnn, rpyg_tr, spyg_tr)

    # ── 6. Train ensemble scorer ──────────────────────────────────────────────
    print('\n[5/7] Training XGBoost + MLP Ensemble...')
    X_train = np.column_stack((bsims_tr, gsims_tr, pen_tr, emm_tr, cov_tr, neg_tr))
    scorer  = AnswerScorer()
    scorer.train(X_train, y_train)

    # ── 7. Test features + evaluation ────────────────────────────────────────
    print('\n[6/7] Extracting TEST features...')
    (bsims_te, rpyg_te, spyg_te, pen_te, emm_te, cov_te, neg_te, rouge_te) = \
        create_graphs_and_features(
            test_df, text_processor, dep_parser, triple_extractor,
            kg_constructor, embedder, validator,
            coref_resolver=coref_resolver, ner_pipeline_obj=ner_pipe)
    y_test   = test_df['human_score'].values.astype(float)
    gsims_te = get_gnn_similarities(gnn, rpyg_te, spyg_te)

    wrong_num_te = compute_wrong_num_flags(pen_te, test_df, dep_parser, text_processor, validator)

    # Baseline models
    base_lr = LinearRegression().fit(np.array(bsims_tr).reshape(-1, 1), y_train)
    y_pred_baseline = np.clip(base_lr.predict(np.array(bsims_te).reshape(-1, 1)), 0, 100)

    rouge_lr = LinearRegression().fit(np.array(rouge_tr).reshape(-1, 1), y_train)
    y_pred_rouge = np.clip(rouge_lr.predict(np.array(rouge_te).reshape(-1, 1)), 0, 100)

    X_tr_kg = np.column_stack((bsims_tr, gsims_tr, emm_tr))
    kg_lr   = LinearRegression().fit(X_tr_kg, y_train)
    y_pred_kg = np.clip(kg_lr.predict(np.column_stack((bsims_te, gsims_te, emm_te))), 0, 100)

    X_tr_noe = np.column_stack((bsims_tr, gsims_tr, pen_tr))
    noe_lr   = LinearRegression().fit(X_tr_noe, y_train)
    y_pred_noe = np.clip(noe_lr.predict(np.column_stack((bsims_te, gsims_te, pen_te))), 0, 100)

    X_test_full = np.column_stack((bsims_te, gsims_te, pen_te, emm_te, cov_te, neg_te))
    y_pred_raw  = scorer.predict_batch(X_test_full)

    y_pred_proposed, override_log = apply_hard_overrides(y_pred_raw, neg_te, pen_te, wrong_num_te)
    n_overrides = sum(1 for r in override_log if r)
    print(f'\n[OK] Hard overrides applied to {n_overrides}/{len(y_pred_proposed)} samples.')

    # ── 8. Evaluate ──────────────────────────────────────────────────────────
    print('\n[7/7] Evaluating...')
    all_metrics = [
        compute_full_metrics(y_test, y_pred_baseline, 'Baseline (LaBSE only)'),
        compute_full_metrics(y_test, y_pred_rouge,    'Baseline (ROUGE-L only)'),
        compute_full_metrics(y_test, y_pred_kg,       'LaBSE + KG + GAT-GNN'),
        compute_full_metrics(y_test, y_pred_noe,      'LaBSE + KG + GAT + Karak'),
        compute_full_metrics(y_test, y_pred_raw,      'Full Ensemble (no override)'),
        compute_full_metrics(y_test, y_pred_proposed, 'Proposed (Full + Overrides)'),
    ]
    print_regression_table(all_metrics)
    print_ci_table(all_metrics)

    b, p = all_metrics[0], all_metrics[-1]
    print('\n' + '=' * 60)
    print('  IMPROVEMENT: PROPOSED vs BASELINE')
    print('=' * 60)
    print(f"  Pearson:    {b['pearson']:.4f} -> {p['pearson']:.4f}  (+{p['pearson']-b['pearson']:.4f})")
    print(f"  MAE:        {b['mae']:.2f} -> {p['mae']:.2f}  (-{b['mae']-p['mae']:.2f})")
    print(f"  F1:         {b['cls_f1']:.1f}% -> {p['cls_f1']:.1f}%  (+{p['cls_f1']-b['cls_f1']:.1f}%)")
    print(f"  Acc+/-10:   {b['within_10']:.1f}% -> {p['within_10']:.1f}%  (+{p['within_10']-b['within_10']:.1f}%)")
    print('=' * 60)

    # ── 9. Save models ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    gnn_path    = os.path.join(args.output_dir, GNN_MODEL_PATH)
    scorer_path = os.path.join(args.output_dir, SCORER_MODEL_PATH)
    torch.save(gnn.state_dict(), gnn_path)
    print(f'\n[OK] GNN saved to {gnn_path}')
    scorer.save_model(scorer_path)

    # ── 10. Persist run outputs for report ───────────────────────────────────
    dataset_used = args.dataset or original_csv
    saver.set_config({
        'script':       'train.py',
        'dataset':      dataset_used,
        'epochs':       args.epochs,
        'random_state': RANDOM_STATE,
        'test_size':    TEST_SIZE,
        'num_adversarial': NUM_ADVERSARIAL,
        'n_train':      len(train_df),
        'n_test':       len(test_df),
        'output_dir':   args.output_dir,
    })
    saver.add_metrics(all_metrics)
    saver.add_predictions(test_df, y_test, y_pred_raw, y_pred_proposed, override_log)
    saver.add_feature_importances(scorer)
    saver.save_all()


if __name__ == '__main__':
    main()

"""
evaluate.py
===========
Stand-alone evaluation script for the Bengali Answer Evaluation System.

Loads pretrained models and evaluates them on the test split of the
augmented dataset. Also runs 5-fold cross-validation on the normal samples.

Usage
-----
    python evaluate.py [--dataset PATH] [--gnn-model PATH] [--scorer-model PATH]
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from transformers import pipeline as hf_pipeline

import torch

from src.config import (
    GNN_EPOCHS, NUM_ADVERSARIAL,
    RANDOM_STATE, TEST_SIZE, CV_N_SPLITS,
    GNN_MODEL_PATH, SCORER_MODEL_PATH,
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
from src.utils.results_saver import ResultsSaver
from src.validation.karak_validator import KarakValidator


def parse_args():
    p = argparse.ArgumentParser(description='Bengali Answer Evaluation — Evaluation')
    p.add_argument('--dataset',        type=str, default=None)
    p.add_argument('--gnn-model',      type=str, default=GNN_MODEL_PATH)
    p.add_argument('--scorer-model',   type=str, default=SCORER_MODEL_PATH)
    p.add_argument('--skip-cv',        action='store_true',
                   help='Skip 5-fold cross-validation (much faster)')
    p.add_argument('--output-results', type=str, default='results',
                   help='Parent directory for run result folders (default: results)')
    return p.parse_args()


def _build_components():
    """Instantiate all NLP and graph components."""
    return {
        'text_processor':   TextPreprocessor(),
        'dep_parser':       BengaliDependencyParser(),
        'coref_resolver':   BengaliCoreferenceResolver(),
        'triple_extractor': TripleExtractor(),
        'kg_constructor':   KnowledgeGraphConstructor(),
        'embedder':         BanglaBERTEmbedder(),
        'validator':        KarakValidator(),
    }


def main():
    args = parse_args()

    # ── Results saver (starts log capture immediately) ────────────────────────
    saver = ResultsSaver(base_dir=args.output_results, label='evaluate')
    saver.start_log_capture()

    # ── Dataset ──────────────────────────────────────────────────────────────
    # Derive augmented CSV path from the source filename so each dataset gets
    # its own augmented file — prevents stale data being reused across datasets.
    original_csv  = args.dataset or find_dataset_csv()
    src_stem      = os.path.splitext(os.path.basename(original_csv))[0]
    src_dir       = os.path.dirname(os.path.abspath(original_csv))
    augmented_csv = os.path.join(src_dir, f'{src_stem}_augmented.csv')

    if not os.path.exists(augmented_csv):
        print(f'[INFO] Generating augmented dataset from: {original_csv}')
        df_augmented = load_and_augment_dataset(original_csv, augmented_csv)
    else:
        df_augmented = pd.read_csv(augmented_csv)
        print(f'[INFO] Loaded augmented dataset: {augmented_csv} ({len(df_augmented)} samples)')

    train_df, test_df = split_dataset(df_augmented, num_adversarial=NUM_ADVERSARIAL,
                                      test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # ── NER pipeline ─────────────────────────────────────────────────────────
    ner_pipe = None
    try:
        ner_pipe = hf_pipeline('ner', model='Davlan/bert-base-multilingual-cased-ner-hrl',
                                aggregation_strategy='simple', device=-1)
        print('[OK] NER pipeline loaded.')
    except Exception as e:
        print(f'[WARN] NER unavailable: {e}')

    # ── Components ────────────────────────────────────────────────────────────
    comps = _build_components()
    tp, dp, cr, te, kgc, em, val = (
        comps['text_processor'], comps['dep_parser'], comps['coref_resolver'],
        comps['triple_extractor'], comps['kg_constructor'],
        comps['embedder'], comps['validator'],
    )

    # ── Dataset-specific model paths ─────────────────────────────────────────
    feat_args = dict(text_processor=tp, dep_parser=dp, triple_extractor=te,
                     kg_constructor=kgc, embedder=em, validator=val,
                     coref_resolver=cr, ner_pipeline_obj=ner_pipe)

    ds_stem          = os.path.splitext(os.path.basename(original_csv))[0]
    os.makedirs('models', exist_ok=True)
    gnn_savepath     = os.path.join('models', f'nlp_dominator_gnn_{ds_stem}.pth')
    scorer_savepath  = os.path.join('models', f'nlp_dominator_scorer_{ds_stem}.pkl')

    # ── Load GNN (dataset-specific fine-tuned, fallback to base weights) ─────
    gnn = SiameseGNN()
    if os.path.exists(gnn_savepath):
        try:
            gnn.load_state_dict(torch.load(gnn_savepath, map_location='cpu'))
            print(f'[OK] GNN loaded from {gnn_savepath} (skipping fine-tuning)')
            gnn_trained = True
        except Exception as e:
            print(f'[WARN] Could not load GNN ({e}). Will fine-tune.')
            gnn_trained = False
    elif os.path.exists(args.gnn_model):
        gnn.load_state_dict(torch.load(args.gnn_model, map_location='cpu'))
        print(f'[OK] Base GNN loaded from {args.gnn_model} — will fine-tune on this dataset.')
        gnn_trained = False
    else:
        print(f'[WARN] No GNN weights found. Fine-tuning from scratch.')
        gnn_trained = False

    # ── Scorer: load if already trained for this dataset ─────────────────────
    scorer        = AnswerScorer()
    scorer_loaded = False
    if os.path.exists(scorer_savepath):
        try:
            scorer.load_model(scorer_savepath)
            scorer_loaded = True
            print(f'[OK] Scorer loaded from {scorer_savepath} (skipping retraining)')
        except Exception as e:
            print(f'[WARN] Could not load scorer ({e}). Will retrain.')

    # ── Train split features (needed if GNN or scorer needs training) ─────────
    if not gnn_trained or not scorer_loaded:
        print('\nExtracting training features...')
        (bsims_tr, rpyg_tr, spyg_tr, pen_tr, emm_tr, cov_tr, neg_tr, _) = \
            create_graphs_and_features(train_df, **feat_args)
        # Ensure numeric arrays (pipeline may return plain Python lists)
        bsims_tr = np.array(bsims_tr, dtype=float)
        pen_tr   = np.array(pen_tr,   dtype=float)
        emm_tr   = np.array(emm_tr,   dtype=float)
        cov_tr   = np.array(cov_tr,   dtype=float)
        neg_tr   = np.array(neg_tr,   dtype=float)
        y_train = train_df['human_score'].values.astype(float)

        # Fine-tune GNN on this dataset's train split
        if not gnn_trained:
            gnn = train_gnn(gnn, rpyg_tr, spyg_tr, y_train, epochs=GNN_EPOCHS)
            torch.save(gnn.state_dict(), gnn_savepath)
            print(f'[INFO] Fine-tuned GNN saved to {gnn_savepath}')

        # Train scorer using fine-tuned GNN similarities
        if not scorer_loaded:
            print('\n[INFO] Training scorer on current dataset train split...')
            gsims_tr = np.array(get_gnn_similarities(gnn, rpyg_tr, spyg_tr), dtype=float)
            X_train  = np.column_stack((bsims_tr, gsims_tr, pen_tr, emm_tr, cov_tr, neg_tr))
            scorer.train(X_train, y_train)
            scorer.save_model(scorer_savepath)
            print(f'[INFO] Scorer saved to {scorer_savepath}')



    print('\nExtracting test features...')
    (bsims_te, rpyg_te, spyg_te, pen_te, emm_te, cov_te, neg_te, rouge_te) = \
        create_graphs_and_features(test_df, **feat_args)
    # Ensure numeric arrays (pipeline may return plain Python lists)
    bsims_te = np.array(bsims_te, dtype=float)
    rouge_te = np.array(rouge_te, dtype=float)
    pen_te   = np.array(pen_te,   dtype=float)
    emm_te   = np.array(emm_te,   dtype=float)
    cov_te   = np.array(cov_te,   dtype=float)
    neg_te   = np.array(neg_te,   dtype=float)
    y_test   = test_df['human_score'].values.astype(float)
    gsims_te = np.array(get_gnn_similarities(gnn, rpyg_te, spyg_te), dtype=float)

    wrong_num_te = compute_wrong_num_flags(pen_te, test_df, dp, tp, val)
    X_test = np.column_stack((bsims_te, gsims_te, pen_te, emm_te, cov_te, neg_te))
    y_pred_raw = scorer.predict_batch(X_test)
    y_pred_proposed, override_log = apply_hard_overrides(y_pred_raw, neg_te, pen_te, wrong_num_te)

    # ── BanglaBERT cosine similarity baseline ────────────────────────────────
    print('\n[INFO] Computing BanglaBERT cosine similarity baseline...')
    bb_sims = []
    for _, row in test_df.iterrows():
        ref_emb = em.get_embedding(str(row['reference_answer'])).unsqueeze(0)
        stu_emb = em.get_embedding(str(row['student_answer'])).unsqueeze(0)
        bb_sims.append(torch.nn.functional.cosine_similarity(ref_emb, stu_emb).item())
    bb_sims = np.array(bb_sims)
    banglabert_pred = np.clip(bb_sims * 100, 0, 100)


    # ── Baselines (no training needed — use already-computed features) ─────────
    labse_pred      = np.clip(bsims_te * 100, 0, 100)
    rouge_pred      = np.clip(rouge_te  * 100, 0, 100)
    avg_pred        = np.clip((bsims_te + rouge_te) / 2 * 100, 0, 100)
    gnn_pred        = np.clip(gsims_te * 100, 0, 100)

    all_metrics = [
        compute_full_metrics(y_test, labse_pred,       'Baseline: LaBSE Cosine Sim'),
        compute_full_metrics(y_test, rouge_pred,       'Baseline: ROUGE-L Only'),
        compute_full_metrics(y_test, avg_pred,         'Baseline: LaBSE + ROUGE-L Avg'),
        compute_full_metrics(y_test, banglabert_pred,  'Baseline: BanglaBERT Cosine Sim'),
        compute_full_metrics(y_test, gnn_pred,         'Ablation: GNN Only'),
        compute_full_metrics(y_test, y_pred_raw,       'Proposed: Full Ensemble (no override)'),
        compute_full_metrics(y_test, y_pred_proposed,  'Proposed: Full + Overrides (ours)'),
    ]
    print_regression_table(all_metrics)
    print_ci_table(all_metrics)

    # ── Register test results with saver ──────────────────────────────────────
    saver.set_config({
        'script':       'evaluate.py',
        'dataset':      augmented_csv,
        'gnn_model':    args.gnn_model,
        'scorer_model': args.scorer_model,
        'random_state': RANDOM_STATE,
        'test_size':    TEST_SIZE,
        'num_adversarial': NUM_ADVERSARIAL,
        'n_test':       len(test_df),
        'skip_cv':      args.skip_cv,
    })
    saver.add_metrics(all_metrics)
    saver.add_predictions(test_df, y_test, y_pred_raw, y_pred_proposed, override_log)
    saver.add_feature_importances(scorer)

    # ── 5-Fold Cross-Validation ───────────────────────────────────────────────
    if not args.skip_cv:
        df_normal = df_augmented.iloc[:-NUM_ADVERSARIAL].copy().reset_index(drop=True)
        kf = KFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        print(f'\n{"="*75}')
        print(f'  5-FOLD CROSS-VALIDATION (non-adversarial, proposed model)')
        print(f'{"="*75}')
        print(f'  CV pool size: {len(df_normal)} (adversarial excluded)\n')

        cv_pearsons, cv_maes, cv_f1s = [], [], []
        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(df_normal)):
            fold_tr = df_normal.iloc[tr_idx].copy().reset_index(drop=True)
            fold_va = df_normal.iloc[val_idx].copy().reset_index(drop=True)
            print(f'  Fold {fold_idx+1}/{CV_N_SPLITS} — train={len(fold_tr)}, val={len(fold_va)}')

            (bsims_f_tr, rpyg_f_tr, spyg_f_tr, pen_f_tr, emm_f_tr, cov_f_tr, neg_f_tr, _) = \
                create_graphs_and_features(fold_tr, **feat_args)
            # Ensure numeric arrays — CV fold train
            bsims_f_tr = np.array(bsims_f_tr, dtype=float)
            pen_f_tr   = np.array(pen_f_tr,   dtype=float)
            emm_f_tr   = np.array(emm_f_tr,   dtype=float)
            cov_f_tr   = np.array(cov_f_tr,   dtype=float)
            neg_f_tr   = np.array(neg_f_tr,   dtype=float)

            (bsims_f_va, rpyg_f_va, spyg_f_va, pen_f_va, emm_f_va, cov_f_va, neg_f_va, _) = \
                create_graphs_and_features(fold_va, **feat_args)
            # Ensure numeric arrays — CV fold val
            bsims_f_va = np.array(bsims_f_va, dtype=float)
            pen_f_va   = np.array(pen_f_va,   dtype=float)
            emm_f_va   = np.array(emm_f_va,   dtype=float)
            cov_f_va   = np.array(cov_f_va,   dtype=float)
            neg_f_va   = np.array(neg_f_va,   dtype=float)

            y_f_tr = fold_tr['human_score'].values.astype(float)
            y_f_va = fold_va['human_score'].values.astype(float)

            # Start from the dataset-specific fine-tuned weights (not random)
            gnn_cv = SiameseGNN()
            if os.path.exists(gnn_savepath):
                gnn_cv.load_state_dict(torch.load(gnn_savepath, map_location='cpu'))
            gnn_cv = train_gnn(gnn_cv, rpyg_f_tr, spyg_f_tr, y_f_tr, epochs=GNN_EPOCHS)
            gsims_f_tr = np.array(get_gnn_similarities(gnn_cv, rpyg_f_tr, spyg_f_tr), dtype=float)
            gsims_f_va = np.array(get_gnn_similarities(gnn_cv, rpyg_f_va, spyg_f_va), dtype=float)

            sc_cv = AnswerScorer()
            X_f_tr = np.column_stack((bsims_f_tr, gsims_f_tr, pen_f_tr, emm_f_tr, cov_f_tr, neg_f_tr))
            sc_cv.train(X_f_tr, y_f_tr)

            X_f_va = np.column_stack((bsims_f_va, gsims_f_va, pen_f_va, emm_f_va, cov_f_va, neg_f_va))
            y_f_va_pred = sc_cv.predict_batch(X_f_va)

            wn_va = compute_wrong_num_flags(pen_f_va, fold_va, dp, tp, val)
            y_f_va_pred, _ = apply_hard_overrides(y_f_va_pred, neg_f_va, pen_f_va, wn_va)

            pc, _ = pearsonr(y_f_va, y_f_va_pred)
            mae   = mean_absolute_error(y_f_va, y_f_va_pred)
            f1    = f1_score((y_f_va >= 50).astype(int),
                             (y_f_va_pred >= 50).astype(int), zero_division=0) * 100
            cv_pearsons.append(pc); cv_maes.append(mae); cv_f1s.append(f1)
            print(f'    Pearson={pc:.4f}  MAE={mae:.2f}  F1={f1:.1f}%\n')
            saver.add_cv_fold(fold_idx + 1, pc, mae, f1)

        print(f'\n{"="*75}')
        print(f'  CV SUMMARY (mean ± std)')
        print(f'{"="*75}')
        print(f'  Pearson:  {np.mean(cv_pearsons):.4f} ± {np.std(cv_pearsons):.4f}')
        print(f'  MAE:      {np.mean(cv_maes):.2f} ± {np.std(cv_maes):.2f}')
        print(f'  F1:       {np.mean(cv_f1s):.1f}% ± {np.std(cv_f1s):.1f}%')
        print(f'{"="*75}')
        saver.add_cv_summary(cv_pearsons, cv_maes, cv_f1s)

    saver.save_all()


if __name__ == '__main__':
    main()

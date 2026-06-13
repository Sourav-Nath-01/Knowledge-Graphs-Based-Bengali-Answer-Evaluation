"""
config.py
=========
Global configuration constants for the Bengali Answer Evaluation System.
Centralises all hardcoded paths, model names, and hyper-parameters so that
any future changes only need to be made here.
"""

import os

# ─── Environment detection ────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/input")

# ─── Base data directory ──────────────────────────────────────────────────────
# Datasets live in  <project_root>/src/data/
# On Kaggle they come from  /kaggle/input/<dataset-name>/
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ─── Dataset paths ────────────────────────────────────────────────────────────
if IS_KAGGLE:
    KAGGLE_INPUT_DIR = "/kaggle/input"
    # Look through typical dataset version directories
    DATASET_CSV_CANDIDATES = [
        "/kaggle/input/nlp-dominator-dataset/dataset_single_sentence.csv",
        "/kaggle/input/nlp-dominator-dataset/dataset_medium_sentences.csv",
        "/kaggle/input/nlp-dominator-dataset/dataset_long_sentences.csv",
        "/kaggle/input/nlp-dominator-dataset/train_data.csv",
    ]
else:
    KAGGLE_INPUT_DIR = DATA_DIR
    DATASET_CSV_CANDIDATES = [
        os.path.join(DATA_DIR, "dataset_single_sentence.csv"),
        os.path.join(DATA_DIR, "dataset_medium_sentences.csv"),
        os.path.join(DATA_DIR, "dataset_long_sentences.csv"),
        os.path.join(DATA_DIR, "train_data.csv"),
    ]

# Augmented dataset output file (written next to the source CSVs)
AUGMENTED_CSV_PATH = os.path.join(DATA_DIR, "train_data_augmented.csv")

# ─── Adversarial augmentation ─────────────────────────────────────────────────
NUM_ADVERSARIAL = 50

# ─── Model identifiers ────────────────────────────────────────────────────────
BANGLA_BERT_MODEL = "csebuetnlp/banglabert"
LABSE_MODEL       = "sentence-transformers/LaBSE"
NER_MODEL         = "sagorsarker/bangla-bert-base-ner"

# ─── Saved model file paths ───────────────────────────────────────────────────
# Trained weights live inside src/models/ alongside the Python source.
_MODELS_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
GNN_MODEL_PATH    = os.path.join(_MODELS_DIR, "nlp_dominator_gnn.pth")
SCORER_MODEL_PATH = os.path.join(_MODELS_DIR, "nlp_dominator_scorer.pkl")

# ─── Training hyper-parameters ────────────────────────────────────────────────
GNN_EPOCHS        = 5
RANDOM_STATE      = 42
TEST_SIZE         = 0.15          # fraction for hold-out test set
CV_N_SPLITS       = 5             # folds for cross-validation

# ─── Scoring thresholds ───────────────────────────────────────────────────────
PASS_THRESHOLD    = 50            # scores >= this are "pass" for F1 computation
HARD_NEG_CAP      = 35.0          # max score when negation mismatch detected
HARD_NUM_CAP      = 40.0          # max score when numeric error detected

# ─── Visualisation font ───────────────────────────────────────────────────────
BENGALI_FONT_NAME = "Noto Sans Bengali"

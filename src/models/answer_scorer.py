"""
answer_scorer.py
================
XGBoost + MLP ensemble answer scorer with feature engineering.
"""

import joblib
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class AnswerScorer:
    """
    Ensemble scorer combining XGBoost and an MLP Regressor.

    Input features (6):
      base_sim   — LaBSE sentence cosine similarity
      graph_sim  — Siamese GNN cosine similarity
      penalty    — Karak validation penalty (0..1)
      entity_mm  — Entity mismatch ratio (0..1)
      coverage   — Sentence coverage score (0..1)
      neg_mm     — Negation mismatch flag (0 or 1)

    Additional engineered features (9) are appended internally.
    """

    def __init__(self):
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', max_iter=600, early_stopping=True,
            validation_fraction=0.15, random_state=42,
            learning_rate='adaptive', learning_rate_init=0.001)
        self.scaler          = StandardScaler()
        self.is_trained      = False
        self.ensemble_weights = (0.6, 0.4)

    # ─── Feature engineering ──────────────────────────────────────────────────

    def _engineer_features(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Pad missing coverage / neg_mm columns if only 4 base features given
        if X.shape[1] == 4:
            X = np.hstack([X, np.ones((X.shape[0], 1)),
                              np.zeros((X.shape[0], 1))])
        base_sim  = X[:, 0:1]
        graph_sim = X[:, 1:2]
        penalty   = X[:, 2:3]
        entity_mm = X[:, 3:4]
        coverage  = X[:, 4:5]
        neg_mm    = X[:, 5:6]
        return np.hstack([
            X,
            base_sim  * penalty,    base_sim  * entity_mm,
            graph_sim * penalty,    penalty   * entity_mm,
            base_sim  - graph_sim,  (base_sim + graph_sim) / 2,
            np.maximum(penalty, entity_mm),
            coverage  * base_sim,   neg_mm * base_sim,
        ])

    # ─── Training ─────────────────────────────────────────────────────────────

    def train(self, X_train, y_train):
        X_eng = self._engineer_features(X_train)
        X_s   = self.scaler.fit_transform(X_eng)
        self.xgb_model.fit(X_eng, y_train)
        self.mlp_model.fit(X_s,   y_train)
        self.is_trained = True

        names = ['base_sim', 'graph_sim', 'penalty', 'entity_mm', 'coverage', 'neg_mm',
                 'sim*pen', 'sim*ent', 'grph*pen', 'pen*ent', 'sim_diff', 'avg_sim',
                 'max_pen', 'cov*sim', 'neg*sim']
        imp = self.xgb_model.feature_importances_
        si  = np.argsort(imp)[::-1]
        print('Scorer trained. XGBoost importances (top 8):')
        for i in si[:8]:
            print(f'  {names[i]:<16}: {imp[i]:.4f}')

    # ─── Inference ────────────────────────────────────────────────────────────

    def predict_score(self, baseline_sim: float, graph_sim: float, penalty: float,
                      entity_mismatch: float = 0.0, coverage: float = 1.0,
                      neg_mismatch_val: float = 0.0) -> float:
        """Predict a single score in [0, 100]."""
        if not self.is_trained:
            raise ValueError('Train first.')
        f = np.array([[baseline_sim, graph_sim, penalty,
                        entity_mismatch, coverage, neg_mismatch_val]])
        X_eng = self._engineer_features(f)
        X_s   = self.scaler.transform(X_eng)
        w1, w2 = self.ensemble_weights
        raw = w1 * self.xgb_model.predict(X_eng)[0] + w2 * self.mlp_model.predict(X_s)[0]
        return float(max(0.0, min(100.0, raw)))

    def predict_batch(self, X) -> np.ndarray:
        """Predict scores for a batch; returns array clipped to [0, 100]."""
        if not self.is_trained:
            raise ValueError('Train first.')
        X_eng = self._engineer_features(X)
        X_s   = self.scaler.transform(X_eng)
        w1, w2 = self.ensemble_weights
        return np.clip(w1 * self.xgb_model.predict(X_eng)
                       + w2 * self.mlp_model.predict(X_s), 0, 100)

    # ─── Explanation ──────────────────────────────────────────────────────────

    def generate_explanation(self, score: float, baseline_sim: float, graph_sim: float,
                             penalty: float, explanations: list,
                             entity_mismatch: float = 0.0, coverage: float = 1.0,
                             neg_mismatch_val: float = 0.0, question_type: str = 'GENERAL',
                             rouge_l: float = 0.0, applied_overrides=None) -> str:
        sep = '=' * 50
        out  = f'{sep}\n'
        out += f'  Score: {score:.2f}/100  [Question type: {question_type}]\n'
        out += f'{sep}\n'
        out += f'  LaBSE similarity:    {baseline_sim:.4f}\n'
        out += f'  GAT-GNN similarity:  {graph_sim:.4f}\n'
        out += f'  ROUGE-L:             {rouge_l:.4f}\n'
        out += f'  Entity mismatch:     {entity_mismatch:.4f}\n'
        out += f'  Sentence coverage:   {coverage:.4f}\n'
        out += f'  Negation mismatch:   {int(neg_mismatch_val)}\n'
        if penalty > 0:
            out += f'\n  Role/structural errors (penalty {penalty:.2f}):\n'
            for e in explanations:
                out += f'    - {e}\n'
        else:
            out += '\n  No role errors detected.\n'
        if applied_overrides:
            out += '\n  [OVERRIDES APPLIED]:\n'
            for ov in applied_overrides:
                out += f'    ⚠ {ov}\n'
        if coverage < 0.8:
            out += f'  WARNING: Only {coverage*100:.0f}% of reference sentences covered.\n'
        if neg_mismatch_val:
            out += '  WARNING: Negation mismatch — score capped at 30.\n'
        out += f'{sep}\n'
        return out

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save_model(self, filepath: str):
        """Save the ensemble to *filepath* using joblib."""
        if self.is_trained:
            joblib.dump({'xgb_model': self.xgb_model,
                         'mlp_model': self.mlp_model,
                         'scaler': self.scaler,
                         'ensemble_weights': self.ensemble_weights}, filepath)
            print(f'Saved to {filepath}')

    def load_model(self, filepath: str):
        """Load a previously saved ensemble from *filepath*."""
        d = joblib.load(filepath)
        self.xgb_model        = d['xgb_model']
        self.mlp_model        = d['mlp_model']
        self.scaler           = d['scaler']
        self.ensemble_weights = d['ensemble_weights']
        self.is_trained       = True
        print(f'Loaded from {filepath}')

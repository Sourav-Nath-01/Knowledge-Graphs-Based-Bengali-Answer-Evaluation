"""
demo.py
=======
Interactive Gradio demo for the Bengali Answer Evaluation System.

Loads the pre-trained GAT-GNN and XGBoost+MLP scorer, then scores a single
student answer against a reference answer for a given question.

Usage
-----
    python demo.py [--share]

Flags
-----
    --share   Launch with a public Gradio link (useful for Kaggle / Colab).
"""

import argparse
import os

import numpy as np
import torch

try:
    import gradio as gr
except ImportError:
    raise SystemExit(
        "Gradio is not installed.  Run:  pip install gradio>=4.0"
    )

from transformers import pipeline as hf_pipeline

from src.config import (
    GNN_MODEL_PATH,
    SCORER_MODEL_PATH,
    PASS_THRESHOLD,
    HARD_NEG_CAP,
    HARD_NUM_CAP,
)
from src.evaluation.overrides import compute_wrong_num_flags, apply_hard_overrides
from src.features.pipeline import (
    compute_entity_mismatch,
    get_gnn_similarities,
)
from src.utils.helpers import sentence_coverage_score
from src.graph.embedder import BanglaBERTEmbedder
from src.graph.kg_constructor import KnowledgeGraphConstructor
from src.models.answer_scorer import AnswerScorer
from src.models.siamese_gnn import SiameseGNN, nx_to_pyg_data
from src.nlp.coreference import BengaliCoreferenceResolver
from src.nlp.dependency_parser import BengaliDependencyParser
from src.nlp.triple_extractor import TripleExtractor
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.utils.helpers import negation_mismatch, compute_rouge_l
from src.validation.karak_validator import KarakValidator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _verdict(score: float) -> str:
    if score >= 80:
        return "✅ Excellent"
    elif score >= 60:
        return "✔️ Good"
    elif score >= PASS_THRESHOLD:
        return "⚠️  Passing"
    else:
        return "❌ Insufficient"


# ── Global objects (loaded once) ───────────────────────────────────────────────

_COMPONENTS = {}   # filled by _load_components()
_GNN        = None
_SCORER     = None


def _load_components():
    global _COMPONENTS, _GNN, _SCORER

    if _COMPONENTS:          # already loaded
        return

    print("[INFO] Loading NLP components …")
    _COMPONENTS = {
        "text_processor":   TextPreprocessor(),
        "dep_parser":       BengaliDependencyParser(),
        "coref_resolver":   BengaliCoreferenceResolver(),
        "triple_extractor": TripleExtractor(),
        "kg_constructor":   KnowledgeGraphConstructor(),
        "embedder":         BanglaBERTEmbedder(),
        "validator":        KarakValidator(),
    }

    # Optional NER
    _COMPONENTS["ner"] = None
    try:
        _COMPONENTS["ner"] = hf_pipeline(
            "ner",
            model="Davlan/bert-base-multilingual-cased-ner-hrl",
            aggregation_strategy="simple",
            device=-1,
        )
        print("[OK] NER pipeline loaded.")
    except Exception as e:
        print(f"[WARN] NER unavailable: {e}")

    # GNN
    _GNN = SiameseGNN()
    if os.path.exists(GNN_MODEL_PATH):
        _GNN.load_state_dict(torch.load(GNN_MODEL_PATH, map_location="cpu"))
        _GNN.eval()
        print(f"[OK] GNN loaded from {GNN_MODEL_PATH}")
    else:
        print(f"[WARN] GNN weights not found at {GNN_MODEL_PATH}. Using random weights.")

    # Scorer
    _SCORER = AnswerScorer()
    if os.path.exists(SCORER_MODEL_PATH):
        _SCORER.load_model(SCORER_MODEL_PATH)
        print(f"[OK] Scorer loaded from {SCORER_MODEL_PATH}")
    else:
        print(f"[WARN] Scorer model not found at {SCORER_MODEL_PATH}. Predictions will be random.")


# ── Core inference function ────────────────────────────────────────────────────

def predict_single(question: str, ref_text: str, stu_text: str):
    """
    Score a single (question, reference_answer, student_answer) triple.

    Returns
    -------
    verdict : str
    explanation : str
    score : float  (0-100)
    """
    _load_components()

    tp  = _COMPONENTS["text_processor"]
    dp  = _COMPONENTS["dep_parser"]
    cr  = _COMPONENTS["coref_resolver"]
    te  = _COMPONENTS["triple_extractor"]
    kgc = _COMPONENTS["kg_constructor"]
    emb = _COMPONENTS["embedder"]
    val = _COMPONENTS["validator"]
    ner = _COMPONENTS["ner"]

    # ── Coreference resolution ────────────────────────────────────────────────
    stu_resolved = cr.resolve(stu_text, question, ref_text) if cr else stu_text

    # ── Baseline cosine similarity ────────────────────────────────────────────
    base_sim = emb.cosine_similarity(ref_text, stu_resolved)

    # ── Knowledge-graph features ──────────────────────────────────────────────
    ref_norm   = tp.normalize(ref_text)
    stu_norm   = tp.normalize(stu_resolved)
    ref_parsed = dp.parse(ref_norm)
    stu_parsed = dp.parse(stu_norm)

    ref_triples = te.extract_triples(raw_text=ref_text)
    stu_triples = te.extract_triples(raw_text=stu_resolved)
    ref_kg  = kgc.build_graph(ref_triples)
    stu_kg  = kgc.build_graph(stu_triples)
    ref_emb = emb.generate_node_embeddings(ref_kg)
    stu_emb = emb.generate_node_embeddings(stu_kg)
    ref_pyg = nx_to_pyg_data(ref_kg, ref_emb)
    stu_pyg = nx_to_pyg_data(stu_kg, stu_emb)

    # ── GNN similarity ────────────────────────────────────────────────────────
    gnn_sim = get_gnn_similarities(_GNN, [ref_pyg], [stu_pyg])[0]

    # ── Karak / semantic-role penalty ─────────────────────────────────────────
    penalty, _ = val.validate(ref_parsed, stu_parsed,
                               ref_raw=ref_text, stu_raw=stu_resolved)

    # ── Entity-mismatch score ─────────────────────────────────────────────────
    entity_mm = compute_entity_mismatch(
        ref_parsed, stu_parsed, ref_text, stu_resolved,
        embedder_obj=emb, ner_pipeline_obj=ner,
    )

    # ── Sentence coverage ─────────────────────────────────────────────────────
    coverage = sentence_coverage_score(ref_text, stu_resolved, emb)

    # ── Negation flag ─────────────────────────────────────────────────────────
    neg_flag = negation_mismatch(ref_text, stu_resolved)

    # ── ROUGE-L ───────────────────────────────────────────────────────────────
    rouge_l = compute_rouge_l(ref_text, stu_resolved)

    # ── Feature vector → raw score ────────────────────────────────────────────
    X = np.array([[base_sim, gnn_sim, penalty, entity_mm, coverage, neg_flag]])
    raw_score = float(_SCORER.predict_batch(X)[0])

    # ── Hard overrides ────────────────────────────────────────────────────────
    # Build minimal structures expected by the override helpers
    import pandas as pd
    dummy_df   = pd.DataFrame({"reference_answer": [ref_text], "student_answer": [stu_text]})
    wrong_num  = compute_wrong_num_flags([penalty], dummy_df, dp, tp, val)
    final_arr, _ = apply_hard_overrides(
        np.array([raw_score]), [neg_flag], [penalty], wrong_num
    )
    final_score = float(final_arr[0])

    # ── Build explanation ─────────────────────────────────────────────────────
    explanation = (
        f"━━━ Feature Breakdown ━━━\n"
        f"  Cosine similarity (LaBSE/BanglaBERT) : {base_sim:.4f}\n"
        f"  GNN graph similarity                 : {gnn_sim:.4f}\n"
        f"  Karak / semantic-role penalty        : {penalty:.4f}\n"
        f"  Entity mismatch penalty              : {entity_mm:.4f}\n"
        f"  Sentence coverage score             : {coverage:.4f}\n"
        f"  Negation mismatch flag               : {int(neg_flag)}\n"
        f"  ROUGE-L                              : {rouge_l:.4f}\n"
        f"\n"
        f"━━━ Scores ━━━\n"
        f"  Raw ensemble score  : {raw_score:.2f} / 100\n"
        f"  Final score (after overrides) : {final_score:.2f} / 100\n"
    )
    if neg_flag and final_score < raw_score:
        explanation += (
            f"\n⚠️  Score capped at {HARD_NEG_CAP:.0f} due to negation mismatch.\n"
        )
    if wrong_num and any(wrong_num) and final_score < raw_score:
        explanation += (
            f"\n⚠️  Score capped at {HARD_NUM_CAP:.0f} due to numeric error.\n"
        )

    verdict = _verdict(final_score)
    return verdict, explanation, round(final_score, 2)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

EXAMPLES = [
    [
        "বাংলাদেশ কবে স্বাধীন হয়েছিল?",
        "বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়েছিল।",
        "বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়েছিল।",
    ],
    [
        "বাংলাদেশ কবে স্বাধীন হয়েছিল?",
        "বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়েছিল।",
        "বাংলাদেশ ১৯৭২ সালে স্বাধীন হয়েছিল।",
    ],
    [
        "সূর্য কোথায় ওঠে?",
        "সূর্য পূর্বদিকে ওঠে।",
        "সূর্য পূর্বদিকে ওঠে না।",
    ],
    [
        "কে রাবণকে বধ করেছিলেন?",
        "রাম রাবণকে বধ করেছিলেন।",
        "রাবণ রামকে বধ করেছিলেন।",
    ],
    [
        "মানুষের শরীরে কয়টি হাড় থাকে?",
        "মানুষের শরীরে ২০৬টি হাড় থাকে।",
        "মানুষের শরীরে ২০৫টি হাড় থাকে।",
    ],
    [
        "গীতাঞ্জলি কে লিখেছিলেন?",
        "রবীন্দ্রনাথ ঠাকুর গীতাঞ্জলি লিখেছিলেন।",
        "রবীন্দ্রনাথ ঠাকুর গীতাঞ্জলি।",
    ],
]


def build_interface() -> gr.Blocks:
    css = """
    .gr-box { border-radius: 8px; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="Bengali Answer Evaluator", css=css) as demo:
        gr.Markdown(
            """
            # 🏛️ Bengali Answer Evaluation System
            **Knowledge Graph · LaBSE · BNLP POS · Siamese GAT-GNN · Karak Validator · XGBoost+MLP**

            Paste a *question*, a *reference answer*, and a *student answer* in Bengali,
            then click **Evaluate**.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                question_input  = gr.Textbox(
                    label="Question (প্রশ্ন)",
                    placeholder="যেমন: বাংলাদেশ কবে স্বাধীন হয়েছিল?",
                    lines=2,
                )
                reference_input = gr.Textbox(
                    label="Reference Answer (সঠিক উত্তর)",
                    placeholder="যেমন: বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়েছিল।",
                    lines=3,
                )
                student_input   = gr.Textbox(
                    label="Student Answer (ছাত্রের উত্তর)",
                    placeholder="যেমন: বাংলাদেশ ১৯৭২ সালে স্বাধীন হয়েছিল।",
                    lines=3,
                )
                with gr.Row():
                    submit_btn = gr.Button("🚀 Evaluate Answer", variant="primary", scale=3)
                    clear_btn  = gr.Button("🗑️ Clear", variant="secondary", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Result")
                score_output = gr.Number(label="Score (0–100)", precision=2)
                label_output = gr.Textbox(label="Verdict", interactive=False)

        gr.Markdown("### 🔍 Detailed Explanation")
        explanation_output = gr.Textbox(
            label="Component Breakdown",
            lines=18,
            interactive=False,
            show_copy_button=True,
        )

        gr.Markdown("### 📚 Example Test Cases")
        gr.Examples(
            examples=EXAMPLES,
            inputs=[question_input, reference_input, student_input],
        )

        gr.HTML(
            """
            <div style="text-align:center;margin-top:10px;color:#888;font-size:12px;">
              🏛️ Knowledge Graph-Based Bengali Answer Evaluation · IIIT Hyderabad 2025<br/>
              Pipeline: LaBSE + BNLP POS + KG + Siamese GAT-GNN + Karak Validator + XGBoost+MLP
            </div>
            """
        )

        # ── Wire-up ─────────────────────────────────────────────────────────
        submit_btn.click(
            fn=predict_single,
            inputs=[question_input, reference_input, student_input],
            outputs=[label_output, explanation_output, score_output],
        )
        clear_btn.click(
            fn=lambda: ("", "", "", "", 0.0),
            inputs=[],
            outputs=[question_input, reference_input, student_input,
                     label_output, score_output],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bengali Answer Evaluation — Gradio Demo")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link.")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    # Pre-load all heavy components before the server starts
    _load_components()

    demo = build_interface()
    demo.launch(
        share=args.share,
        server_port=args.server_port,
        debug=False,
    )


if __name__ == "__main__":
    main()

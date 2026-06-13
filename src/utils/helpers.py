"""
helpers.py
==========
Shared utility functions for the Bengali Answer Evaluation System.
Includes Bengali digit normalisation, question-type detection,
sentence coverage, negation detection, ROUGE-L, and font setup.
"""

import os
import unicodedata

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ─── Bengali digit utilities ──────────────────────────────────────────────────
BENGALI_DIGIT_MAP = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')


def is_bengali_numeral(word: str) -> bool:
    """Return True if *word* contains at least one decimal-digit character."""
    return any(unicodedata.category(c) == 'Nd' for c in word)


def normalize_digits(word: str) -> str:
    """Translate Bengali/Devanagari digits to ASCII digits."""
    return word.translate(BENGALI_DIGIT_MAP)


# ─── Question-type detection ──────────────────────────────────────────────────
QUESTION_TYPE_KW = {
    'who':      ['কে', 'কার', 'কাকে', 'কোনজন'],
    'what':     ['কী', 'কিসে', 'কিসের', 'কি', 'কিসেই'],
    'where':    ['কোথায়', 'কোথা', 'কোন জায়গায়'],
    'when':     ['কবে', 'কখন', 'কোন সময়', 'কোন সালে'],
    'how':      ['কিভাবে', 'কেমনে', 'কেমন'],
    'why':      ['কেন', 'কোন কারণে'],
    'howmany':  ['কয়টি', 'কতটি', 'কতজন', 'কয়জন', 'কত'],
}


def detect_question_type(question: str) -> str:
    """Return a coarse question-type label based on Bengali interrogative words."""
    for qtype, keywords in QUESTION_TYPE_KW.items():
        for kw in keywords:
            if kw in question:
                return qtype
    return 'general'


# ─── Sentence coverage ────────────────────────────────────────────────────────
def sentence_coverage_score(ref_text: str, stu_text: str, embedder, threshold: float = 0.65) -> float:
    """
    Compute sentence-level coverage score.

    For each sentence in *ref_text* find the maximum cosine similarity with
    any sentence in *stu_text*.  Return the average maximum similarity.
    """
    try:
        ref_sents = [s.strip() for s in ref_text.replace('।', '\n').split('\n') if s.strip()]
        stu_sents = [s.strip() for s in stu_text.replace('।', '\n').split('\n') if s.strip()]
        if not ref_sents or not stu_sents:
            return 0.0

        ref_embs = embedder.encode_sentences(ref_sents)
        stu_embs = embedder.encode_sentences(stu_sents)

        import torch
        import torch.nn.functional as F
        ref_t = F.normalize(torch.tensor(ref_embs), dim=1)
        stu_t = F.normalize(torch.tensor(stu_embs), dim=1)

        sims = torch.mm(ref_t, stu_t.T)           # [n_ref, n_stu]
        max_sims = sims.max(dim=1).values          # [n_ref]
        return float(max_sims.mean().item())
    except Exception:
        return 0.0


# ─── Negation mismatch ────────────────────────────────────────────────────────
NEGATION_WORDS = {'না', 'নয়', 'নি', 'নে', 'নেই', 'নাই', 'না-ই'}


def negation_mismatch(ref_text: str, stu_text: str) -> bool:
    """
    Return True when one text contains a negation word and the other does not
    (simple surface-level heuristic).
    """
    ref_words = set(ref_text.split())
    stu_words = set(stu_text.split())
    ref_neg = bool(ref_words & NEGATION_WORDS)
    stu_neg = bool(stu_words & NEGATION_WORDS)
    return ref_neg != stu_neg


# ─── ROUGE-L ──────────────────────────────────────────────────────────────────
def compute_rouge_l(ref_text: str, stu_text: str) -> float:
    """Return ROUGE-L F-measure between *ref_text* and *stu_text*."""
    try:
        from rouge_score import rouge_scorer as rouge_lib  # noqa: F401
        scorer = rouge_lib.RougeScorer(['rougeL'], use_stemmer=False)
        scores = scorer.score(ref_text, stu_text)
        return float(scores['rougeL'].fmeasure)
    except Exception:
        return 0.0


# ─── Bengali font setup ───────────────────────────────────────────────────────
def setup_bengali_font():
    """
    Attempt to configure matplotlib to use a Bengali-capable font.
    Returns a FontProperties object if successful, else None.
    """
    # Flush stale font cache
    try:
        _cache_dir = matplotlib.get_cachedir()
        for _f in os.listdir(_cache_dir):
            if _f.startswith('fontlist') and _f.endswith('.json'):
                os.remove(os.path.join(_cache_dir, _f))
        fm._load_fontmanager(try_read_cache=False)
    except (IOError, OSError):
        pass

    bengali_font = None
    for f in fm.findSystemFonts():
        fname = os.path.basename(f).lower()
        if 'notosansbengali' in fname or ('noto' in fname and 'bengali' in fname):
            bengali_font = f
            break
        if 'lohit' in fname and 'bengali' in fname:
            bengali_font = f

    if bengali_font is None:
        for candidate in [
            os.path.expanduser('~/.local/share/fonts/NotoSansBengali-Regular.ttf'),
            '/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf',
            '/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf',
        ]:
            if os.path.exists(candidate):
                bengali_font = candidate
                break

    if bengali_font:
        try:
            bengali_fp = fm.FontProperties(fname=bengali_font)
            plt.rcParams['font.family'] = bengali_fp.get_name()
            fm.fontManager.addfont(bengali_font)
            print(f'[OK] Bengali font configured: {os.path.basename(bengali_font)}')
            return bengali_fp
        except Exception as e:
            print(f'[WARN] Error configuring Bengali font: {e}')
            return None
    else:
        print('[WARN] No Bengali font found.')
        return None

"""
dependency_parser.py
====================
BNLP-based Bengali dependency parser.
"""

from src.utils.helpers import is_bengali_numeral

try:
    from bnlp import BengaliPOS
except ImportError:
    BengaliPOS = None


class BengaliDependencyParser:
    """
    Lightweight dependency parser for Bengali using BNLP POS tags.

    Falls back to a position-based SOV heuristic when the BNLP tagger
    is unavailable.
    """

    NEGATIONS    = {'না', 'নয়', 'নি', 'নে', 'নেই', 'নাই', 'না-ই'}
    OBJ_SUFFIXES  = ('কে', 'কেই')
    LOC_SUFFIXES  = ('তে', 'তেই', 'য়', 'য়ে')
    POSS_SUFFIXES = ('র', 'এর')

    BNLP_TO_UPOS = {
        'NNP': 'PROPN', 'NNS': 'NOUN', 'NN': 'NOUN',
        'VBZ': 'VERB',  'VBD': 'VERB', 'VBN': 'VERB',
        'VB':  'VERB',  'VBP': 'VERB', 'VBG': 'VERB',
        'JJ':  'ADJ',   'RB':  'ADV',
        'QF':  'NUM',   'QT_QTF': 'NUM', 'QT_QTC': 'NUM', 'Q': 'NUM',
        'CC':  'CCONJ', 'IN': 'ADP',   'DT': 'DET',
        'PRP': 'PRON',  'WP': 'PRON',
        'NEG': 'PART',  'RP': 'PART',
        'UNK': 'X',
    }

    def __init__(self):
        print('Loading BNLP POS tagger for dependency parsing...')
        if BengaliPOS is not None:
            try:
                self.pos_tagger = BengaliPOS()
                self.use_bnlp = True
                print('  [OK] BNLP POS tagger loaded.')
            except Exception as e:
                print(f'  [WARN] BNLP POS tagger failed: {e}. SOV fallback active.')
                self.pos_tagger = None
                self.use_bnlp = False
        else:
            print('  [WARN] bnlp package not available. SOV fallback active.')
            self.pos_tagger = None
            self.use_bnlp = False

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _bnlp_tag(self, text: str):
        try:
            return self.pos_tagger.tag(text)
        except Exception:
            return [(w, 'UNK') for w in text.split()]

    def _tag_to_upos(self, bnlp_tag: str) -> str:
        return self.BNLP_TO_UPOS.get(bnlp_tag, 'X')

    def _assign_deprel(self, word, upos, position, total, verb_idx, subj_assigned):
        if is_bengali_numeral(word) or upos == 'NUM':
            return 'nummod'
        if word in self.NEGATIONS:
            return 'advmod'
        if position == verb_idx:
            return 'root'
        if upos in ('NOUN', 'PROPN', 'PRON'):
            if any(word.endswith(s) for s in self.OBJ_SUFFIXES):
                return 'obj'
            if any(word.endswith(s) for s in self.LOC_SUFFIXES):
                return 'obl:loc'
            if any(word.endswith(s) for s in self.POSS_SUFFIXES):
                return 'nmod:poss'
            if not subj_assigned:
                return 'nsubj'
            return 'obj'
        if upos == 'ADJ':
            return 'amod'
        if upos in ('ADP', 'PART'):
            return 'case'
        return 'dep'

    # ─── Public API ───────────────────────────────────────────────────────────

    def parse(self, text: str) -> list:
        """
        Parse *text* and return a list of sentence lists (CoNLL-U style dicts).

        Each token dict has keys: id, text, lemma, upos, head, deprel, bnlp_tag.
        """
        # Split on Bengali sentence boundaries FIRST (。 and newlines).
        # Commas are intentionally NOT removed before POS tagging —
        # they help the tagger identify phrase boundaries.
        raw_sentences = [s.strip() for s in text.replace('?', '').replace('।', '\n').split('\n')
                         if s.strip()]
        sentences_raw = raw_sentences  # preserve commas for tagger accuracy
        if not sentences_raw:
            sentences_raw = [text.replace('।', '').replace('?', '').strip()]

        parsed_doc = []
        for sent_text in sentences_raw:
            words_raw = sent_text.split()
            if not words_raw:
                continue
            # Pass the sentence (with commas) to the tagger for better accuracy,
            # then strip trailing punctuation from each token text afterwards.
            tagged_raw = self._bnlp_tag(sent_text) if self.use_bnlp else [(w, 'UNK') for w in words_raw]
            # Drop pure-comma / pure-punctuation tokens after tagging
            tagged = [(w.strip(','), t) for w, t in tagged_raw
                      if w.strip(',.?।') != '']

            if not tagged:
                continue

            # Find rightmost verb
            verb_idx = len(tagged) - 1
            for i in range(len(tagged) - 1, -1, -1):
                if self._tag_to_upos(tagged[i][1]) == 'VERB':
                    verb_idx = i
                    break

            parsed_sent = []
            subj_assigned = False
            for i, (word, bnlp_tag) in enumerate(tagged):
                # Comma/punctuation tokens: mark as PUNCT with 'punct' deprel
                if word in (',', '।', '.', '!', '?', ';', ':'):
                    parsed_sent.append({
                        'id': i + 1, 'text': word, 'lemma': word,
                        'upos': 'PUNCT', 'head': verb_idx + 1, 'deprel': 'punct',
                        'bnlp_tag': bnlp_tag,
                    })
                    continue

                upos   = self._tag_to_upos(bnlp_tag)
                deprel = self._assign_deprel(word, upos, i, len(tagged), verb_idx, subj_assigned)
                if deprel == 'nsubj':
                    subj_assigned = True
                head = 0 if deprel == 'root' else verb_idx + 1
                parsed_sent.append({
                    'id': i + 1, 'text': word, 'lemma': word,
                    'upos': upos, 'head': head, 'deprel': deprel, 'bnlp_tag': bnlp_tag,
                })
            parsed_doc.append(parsed_sent)

        return parsed_doc if parsed_doc else [[]]


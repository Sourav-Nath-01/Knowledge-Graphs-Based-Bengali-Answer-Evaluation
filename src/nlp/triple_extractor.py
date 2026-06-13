"""
triple_extractor.py
===================
Relation triple extraction from Bengali text using BNLP POS tags.

Improvements over the original:
  - Relation = rightmost VERB (VBZ/VBD/VBP/VBN/VBG/VB) tag, NOT always words[-1].
  - Subject   = first PROPN/NOUN/PRON token, NOT always words[0].
  - Object    = subsequent PROPN/NOUN tokens after subject, joined.
  - Adverbs (RB) are concatenated to the verb to enrich the relation.
  - Compound sentences detected via CC (conjunction) tags are split and each
    sub-clause produces its own triple.
  - Postposition-role triples retain the specific role label.
  - Returned triples carry a 'triple_type' label for downstream graph typing.
"""

import re as _re
from typing import List, Tuple

from src.utils.helpers import is_bengali_numeral

try:
    from bnlp import BengaliPOS
except ImportError:
    BengaliPOS = None


# ─── Type aliases ─────────────────────────────────────────────────────────────
# Each triple is (subject, relation, object, triple_type)
# triple_type ∈ {'main', 'count', 'quality', 'time', 'location',
#                'source', 'instrument', 'purpose', 'companion', 'possessive'}
Triple = Tuple[str, str, str, str]


class TripleExtractor:
    """
    Extract (subject, relation, object, triple_type) triples from Bengali text.

    Uses BNLP POS tagging to identify:
      * Real verbs for the relation slot
      * PROPN / NOUN tokens for subject and object slots
      * Adverbs attached to the verb for richer relations
      * Conjunctions to split compound sentences
      * Postpositions for semantic role triples
    """

    # Postpositions → semantic role label
    POSTPOSITIONS = {
        'কে':    'object',
        'কেই':   'object',
        'র':     'possessive',
        'এর':    'possessive',
        'তে':    'location',
        'তেই':   'location',
        'থেকে':  'source',
        'হতে':   'source',
        'দিয়ে':  'instrument',
        'দ্বারা': 'instrument',
        'জন্য':  'purpose',
        'জন্যে': 'purpose',
        'সাথে':  'companion',
        'সঙ্গে': 'companion',
    }

    # Conjunction tags that mark clause boundaries
    CC_TAGS = {'CC'}
    # POS tags treated as VERB
    VERB_TAGS = {'VBZ', 'VBD', 'VBP', 'VBN', 'VBG', 'VB'}
    # POS tags treated as nominal subjects / objects
    NOUN_TAGS = {'NNP', 'NNS', 'NN', 'PRP', 'WP'}
    # POS tags treated as adjectives
    ADJ_TAGS  = {'JJ'}
    # POS tags treated as adverbs
    ADV_TAGS  = {'RB'}
    # POS tags treated as numerals
    NUM_TAGS  = {'QT_QTF', 'QT_QTC', 'QF', 'Q', 'NUM'}

    NEGATIONS  = {'না', 'নয়', 'নি', 'নে', 'নেই', 'না-ই', 'নাই'}
    TIME_WORDS = ['সাল', 'বছর', 'কাল', 'দিন', 'সময়', 'শতাব্দী', 'সালে', 'বছরে']

    def __init__(self):
        print('Loading BNLP POS tagger for TripleExtractor...')
        if BengaliPOS is not None:
            try:
                self.pos_tagger = BengaliPOS()
                print('  [OK] BNLP POS tagger loaded.')
            except Exception as e:
                print(f'  [WARN] BNLP POS tagger unavailable: {e}')
                self.pos_tagger = None
        else:
            print('  [WARN] bnlp package not available. Fallback to simple split.')
            self.pos_tagger = None

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _tag_words(self, sentence: str) -> List[Tuple[str, str]]:
        """Return [(word, bnlp_tag), …] for *sentence*."""
        words = sentence.replace('।', '').replace('?', '').split()
        if not words:
            return []
        if self.pos_tagger is not None:
            try:
                return self.pos_tagger.tag(sentence)
            except Exception:
                pass
        return [(w, 'UNK') for w in words]

    def _split_on_conjunctions(
        self, tagged: List[Tuple[str, str]]
    ) -> List[List[Tuple[str, str]]]:
        """
        Split a tagged token list wherever a CC (coordinating conjunction) appears.
        Returns a list of sub-lists, each representing one clause.
        """
        clauses, current = [], []
        for tok in tagged:
            if tok[1] in self.CC_TAGS and current:
                clauses.append(current)
                current = []
            else:
                current.append(tok)
        if current:
            clauses.append(current)
        return clauses if clauses else [tagged]

    def _extract_from_clause(
        self, clause: List[Tuple[str, str]], neg_flag: bool
    ) -> List[Triple]:
        """
        Extract triples from a single tagged clause (list of (word, tag) pairs).

        Strategy
        --------
        1. Identify the rightmost VERB token → relation.
        2. Attach any ADV token immediately before the verb → enrich relation.
        3. First NOUN/PROPN/PRON before verb → subject.
        4. NOUN/PROPN tokens after verb → object(s).
        5. Emit special triples for NUM, ADJ, TIME, postposition roles.
        """
        if not clause:
            return []

        words_only = [w for w, _ in clause]

        # ── Find rightmost VERB ────────────────────────────────────────────
        verb_idx = None
        for i in range(len(clause) - 1, -1, -1):
            if clause[i][1] in self.VERB_TAGS:
                verb_idx = i
                break

        # Fallback: if no tagged verb found, use last word (SOV heuristic)
        if verb_idx is None:
            verb_idx = len(clause) - 1

        verb_word = clause[verb_idx][0]

        # ── Attach preceding adverb to relation ───────────────────────────
        adv_prefix = ''
        if verb_idx > 0 and clause[verb_idx - 1][1] in self.ADV_TAGS:
            adv_prefix = clause[verb_idx - 1][0] + '_'

        relation_base = adv_prefix + verb_word
        relation = f'NOT({relation_base})' if neg_flag else relation_base

        # ── Find subject (first NOUN/PROPN/PRON before verb) ──────────────
        subject = None
        subject_idx = None
        for i in range(verb_idx):
            w, t = clause[i]
            if t in self.NOUN_TAGS:
                subject = w
                subject_idx = i
                break

        # Fallback to first word if no nominal found before verb
        if subject is None:
            subject = words_only[0]
            subject_idx = 0

        # Anchor used for all sub-triples in this clause
        anchor = subject

        triples: List[Triple] = []
        obj_parts = []

        for i, (word, tag) in enumerate(clause):
            # Skip subject word, verb word, its adverb, and negations
            if i == subject_idx or i == verb_idx:
                continue
            if verb_idx > 0 and i == verb_idx - 1 and tag in self.ADV_TAGS:
                continue   # already embedded in relation
            if word in self.NEGATIONS:
                continue

            # ── Numeric ───────────────────────────────────────────────────
            if is_bengali_numeral(word) or tag in self.NUM_TAGS:
                triples.append((anchor, 'count', word, 'count'))
                continue

            # ── Adjective → quality triple ────────────────────────────────
            if tag in self.ADJ_TAGS:
                triples.append((anchor, 'quality', word, 'quality'))
                continue

            # ── Time expression ───────────────────────────────────────────
            if any(tw in word for tw in self.TIME_WORDS):
                triples.append((anchor, 'time', word, 'time'))
                continue

            # ── Postposition role ──────────────────────────────────────────
            matched_role = None
            for pp, role in self.POSTPOSITIONS.items():
                if word.endswith(pp) and role == 'location':
                    matched_role = role
                    break
            if matched_role:
                triples.append((anchor, matched_role, word, matched_role))
                continue

            # ── Possessive postposition ───────────────────────────────────
            poss_role = None
            for pp, role in self.POSTPOSITIONS.items():
                if word.endswith(pp) and role == 'possessive':
                    poss_role = role
                    break
            if poss_role:
                triples.append((anchor, 'possessive', word, 'possessive'))
                continue

            # ── Nominal after verb → part of object ───────────────────────
            if i > verb_idx and tag in self.NOUN_TAGS:
                obj_parts.append(word)
                continue

            # ── Everything else → collect into object ─────────────────────
            if i > verb_idx:
                obj_parts.append(word)

        # Build main triple
        obj_candidate = ' '.join(obj_parts) if obj_parts else '[NONE]'
        main_triple: Triple = (anchor, relation, obj_candidate, 'main')
        if main_triple not in triples:
            triples.append(main_triple)

        return triples

    # ─── Public API ───────────────────────────────────────────────────────────

    def extract_triples(self, parsed_sentences=None, raw_text: str = '') -> List[Triple]:
        """
        Extract triples from *raw_text*.

        Returns
        -------
        list of (subject, relation, object, triple_type) tuples
        """
        text = raw_text if raw_text else ''
        if not text:
            return []

        all_triples: List[Triple] = []
        # Split on sentence boundaries: Bengali full stop and newlines
        sentences = [s.strip() for s in _re.split(r'[।\n]+', text) if s.strip()]

        for sent in sentences:
            # Clean punctuation but keep commas for now (stripped per-token below
            # only if needed); POS tagger can handle them.
            clean_sent = sent.replace('?', '')
            tagged = self._tag_words(clean_sent)
            if not tagged:
                continue

            neg_flag = any(w in self.NEGATIONS for w, _ in tagged)

            # Split compound sentence on coordinating conjunctions
            clauses = self._split_on_conjunctions(tagged)

            for clause in clauses:
                if len(clause) < 2:
                    continue
                clause_triples = self._extract_from_clause(clause, neg_flag)
                # Deduplicate before extending
                for t in clause_triples:
                    if t not in all_triples:
                        all_triples.append(t)

        return all_triples

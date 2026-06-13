"""
coreference.py
==============
Rule-based coreference resolution for Bengali.

Strategy
--------
Process the student answer sentence by sentence.  For each sentence:

  1. Resolve PERSON pronouns (তিনি, সে, তার …) to the most recently
     mentioned PERSON-type entity.
  2. Resolve THING pronouns  (এটি, সেটি, এটা …) to the most recently
     mentioned THING-type entity.
  3. After resolving, scan the *original* sentence for new entity
     candidates and update the recency focus so that subsequent
     sentences pick the right antecedent.

Seeding
-------
The initial focus is seeded from *question_text* and *reference_text*
(priority: question first, then reference).  This handles single-sentence
answers correctly, because there are no earlier sentences in the student
answer to draw from.

Pronoun-type heuristic
----------------------
Without a full NER tagger we cannot reliably distinguish person names from
common nouns.  We use position-in-sentence as a proxy (Bengali is typically
SOV):
  * First noun/word  →  subject  →  PERSON focus candidate
  * Last  noun/word  →  object   →  THING  focus candidate
"""

import re as _re
from typing import Dict, List, Optional


class BengaliCoreferenceResolver:
    """
    Sentence-level, recency-aware Bengali coreference resolver.

    Improvements over the original single-pass design:
      • Works sentence-by-sentence — later sentences use the most recently
        mentioned entity, not always the first word in the question.
      • Separates PERSON and THING pronoun resolution.
      • Replaces only the first occurrence of each pronoun per sentence to
        avoid over-substitution.
      • Updates the "focus" entity after processing each sentence.
    """

    # Each pronoun maps to its semantic type: PERSON or THING
    PRONOUNS: Dict[str, str] = {
        'তিনি':  'PERSON', 'তাঁর':   'PERSON', 'তাঁকে':  'PERSON',
        'তাঁদের':'PERSON', 'সে':     'PERSON', 'তার':    'PERSON',
        'তাকে':  'PERSON', 'তাদের':  'PERSON', 'তারা':   'PERSON',
        'তাঁরা': 'PERSON', 'তিনিই':  'PERSON', 'সেই':    'PERSON',
        'যিনি':  'PERSON', 'যে':     'PERSON', 'ইনি':    'PERSON',
        'উনি':   'PERSON', 'ওনার':   'PERSON',
        'এটি':   'THING',  'এটা':    'THING',  'সেটি':   'THING',
        'সেটা':  'THING',  'এগুলো':  'THING',  'সেগুলো': 'THING',
        'ওটা':   'THING',  'ওটি':    'THING',
    }

    # Words that are never useful resolution targets
    STOP: set = {
        'কে', 'কী', 'কোথায়', 'কবে', 'কিভাবে', 'কেন', 'এবং', 'ও', 'আর',
        'করে', 'করেছে', 'করেছিল', 'করেছিলেন', 'হয়', 'ছিল', 'ছিলেন',
        'থেকে', 'যে', 'তখন', 'এখন', 'সেখানে', 'এখানে',
    }

    _SENT_SEP = _re.compile(r'[।\n]+')

    def __init__(self):
        pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> List[str]:
        """Split *text* on '।' and newlines; return non-empty stripped segments."""
        return [s.strip() for s in self._SENT_SEP.split(text) if s.strip()]

    def _extract_candidates(self, text: str) -> List[str]:
        """
        Return content words that could be antecedents (not pronouns, not stop words,
        length > 1).
        """
        clean = text.replace('।', '').replace(',', '').replace('?', '')
        return [
            w for w in clean.split()
            if w not in self.PRONOUNS
            and w not in self.STOP
            and len(w) > 1
        ]

    def _update_focus(self, focus: dict, sentence: str):
        """
        Scan *sentence* for new entity candidates and update the focus dict
        in-place.

        Bengali SOV heuristic:
          first candidate → subject  → PERSON focus
          last  candidate → object   → THING  focus
        """
        candidates = self._extract_candidates(sentence)
        if not candidates:
            return
        # First non-pronoun word is the most likely subject (PERSON antecedent)
        focus['PERSON'] = candidates[0]
        # Last non-pronoun word is the most likely object (THING antecedent)
        focus['THING'] = candidates[-1] if len(candidates) > 1 else candidates[0]

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(self, student_text: str,
                question_text: str = '',
                reference_text: str = '') -> str:
        """
        Return *student_text* with recognized pronouns replaced by their
        most likely antecedents.

        Parameters
        ----------
        student_text    : the answer to resolve
        question_text   : used to seed the initial entity focus
        reference_text  : used to seed the initial entity focus (secondary)
        """
        # Quick exit: no pronoun present — nothing to do
        if not any(p in student_text for p in self.PRONOUNS):
            return student_text

        # ── Seed focus from context ────────────────────────────────────────
        focus: Dict[str, Optional[str]] = {'PERSON': None, 'THING': None}
        for ctx in [question_text, reference_text]:
            if not ctx:
                continue
            candidates = self._extract_candidates(ctx)
            if candidates:
                if focus['PERSON'] is None:
                    focus['PERSON'] = candidates[0]
                if focus['THING'] is None:
                    focus['THING'] = candidates[-1] if len(candidates) > 1 else candidates[0]
            if focus['PERSON'] and focus['THING']:
                break   # both slots filled — stop scanning context

        # ── Sentence-level resolution ──────────────────────────────────────
        sentences = self._split_sentences(student_text)
        if not sentences:
            return student_text

        resolved_parts = []
        for sent in sentences:
            resolved_sent = sent

            # Resolve pronouns using CURRENT focus (i.e. most recent entity)
            for pronoun, ptype in self.PRONOUNS.items():
                if pronoun not in resolved_sent:
                    continue
                antecedent = focus.get(ptype)
                if antecedent and antecedent != pronoun:
                    # Replace only the FIRST occurrence in this sentence so
                    # repeated pronouns referring to different entities aren't
                    # all collapsed to the same word.
                    resolved_sent = resolved_sent.replace(pronoun, antecedent, 1)

            resolved_parts.append(resolved_sent)

            # Update focus from ORIGINAL sentence (before pronoun substitution)
            # so substituted words don't pollute the candidate pool.
            self._update_focus(focus, sent)

        return '। '.join(resolved_parts)

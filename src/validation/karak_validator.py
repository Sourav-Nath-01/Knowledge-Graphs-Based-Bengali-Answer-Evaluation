"""
karak_validator.py
==================
Semantic-role (karak) validation for Bengali answer scoring.
"""

from src.utils.helpers import is_bengali_numeral, normalize_digits


class KarakValidator:
    """
    Validate student answers by checking semantic role alignment.

    Roles checked: agent, object, location/time, instrument, numeric, modifier.
    Applies role-swap penalty, missing-location penalty, numeric mismatch penalty,
    and negation mismatch hard penalty.
    """

    NEGATIONS = {'না', 'নয়', 'নি', 'নে', 'নেই', 'নাই', 'না-ই'}

    def __init__(self):
        pass

    def has_negation(self, text: str) -> bool:
        return any(neg in text.split() for neg in self.NEGATIONS)

    def extract_roles(self, parsed_sentence: list) -> dict:
        """
        Extract semantic roles from a single parsed sentence (list of token dicts).

        Returns
        -------
        dict with keys: agent, object, location_time, instrument, numeric, modifier
        """
        roles = {
            'agent': set(), 'object': set(), 'location_time': set(),
            'instrument': set(), 'numeric': set(), 'modifier': set(),
        }
        for word in parsed_sentence:
            deprel = word['deprel']
            # Skip punctuation tokens added by the updated parser
            if deprel == 'punct' or word.get('upos') == 'PUNCT':
                continue
            lemma  = word.get('lemma') or word['text']
            raw    = word['text']
            if is_bengali_numeral(raw) or deprel == 'nummod':
                roles['numeric'].add(normalize_digits(raw))
            elif 'subj' in deprel:
                roles['agent'].add(lemma)
            elif 'obj' in deprel:
                roles['object'].add(lemma)
            elif deprel in ('obl:loc', 'obl:tmod'):
                roles['location_time'].add(lemma)
            elif deprel in ('obl:ins', 'obl'):
                roles['instrument'].add(lemma)
            elif deprel in ('amod', 'nmod', 'nmod:poss'):
                roles['modifier'].add(lemma)
        return roles

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _swap_penalty(self, ref_roles: dict, stu_roles: dict):
        penalty = 0.0
        explanations = []
        ref_all = ref_roles['agent'] | ref_roles['object']
        stu_all = stu_roles['agent'] | stu_roles['object']
        if ref_all == stu_all:
            return 0.0, []
        for agent in ref_roles['agent']:
            if agent in stu_roles['object'] and agent not in stu_roles['agent']:
                penalty += 0.5
                explanations.append(f"Role swap: '{agent}' was AGENT, now OBJECT.")
        for obj in ref_roles['object']:
            if obj in stu_roles['agent'] and obj not in stu_roles['object']:
                penalty += 0.5
                explanations.append(f"Role swap: '{obj}' was OBJECT, now AGENT.")
        return min(penalty, 1.0), explanations

    # ─── Public API ───────────────────────────────────────────────────────────

    def validate(self, ref_parsed: list, stu_parsed: list,
                 ref_raw: str = '', stu_raw: str = ''):
        """
        Compute the total penalty for a student answer.

        Parameters
        ----------
        ref_parsed, stu_parsed : list of sentence lists (from BengaliDependencyParser)
        ref_raw, stu_raw       : raw text strings for negation checking

        Returns
        -------
        (penalty: float, explanations: list of str)
        """
        blank = {'agent': set(), 'object': set(), 'location_time': set(),
                 'instrument': set(), 'numeric': set(), 'modifier': set()}
        ref_roles = {k: set() for k in blank}
        stu_roles = {k: set() for k in blank}
        for sent in ref_parsed:
            for k, v in self.extract_roles(sent).items():
                ref_roles[k].update(v)
        for sent in stu_parsed:
            for k, v in self.extract_roles(sent).items():
                stu_roles[k].update(v)

        penalty, explanations = self._swap_penalty(ref_roles, stu_roles)

        missing_loc = ref_roles['location_time'] - stu_roles['location_time']
        if missing_loc:
            penalty += 0.1 * len(missing_loc)
            explanations.append(f'Missing location/time: {missing_loc}')

        missing_num = ref_roles['numeric'] - stu_roles['numeric']
        wrong_num   = stu_roles['numeric'] - ref_roles['numeric']
        if missing_num or wrong_num:
            penalty += 0.25 * len(missing_num) + 0.3 * len(wrong_num)
            if missing_num:
                explanations.append(f'Missing numeric: {missing_num}')
            if wrong_num:
                explanations.append(f'Wrong numeric: {wrong_num}')

        missing_mod = ref_roles['modifier'] - stu_roles['modifier']
        if missing_mod:
            penalty += 0.05 * len(missing_mod)

        if ref_raw and stu_raw:
            if self.has_negation(ref_raw) != self.has_negation(stu_raw):
                penalty += 0.6
                explanations.append('Negation mismatch detected (hard penalty).')

        return min(penalty, 1.0), explanations

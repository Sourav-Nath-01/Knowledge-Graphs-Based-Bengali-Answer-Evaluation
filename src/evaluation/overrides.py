"""
overrides.py
============
Hard post-processing overrides applied after ensemble prediction.
"""

import numpy as np


def apply_hard_overrides(scores: np.ndarray, neg_flags, karak_penalties, wrong_num_flags):
    """
    Apply hard post-processing caps after ensemble prediction.

    Rules
    -----
    1. Negation mismatch  → cap score at 30
    2. Wrong number value → cap score at 25

    Parameters
    ----------
    scores          : array-like of predicted scores
    neg_flags       : array-like of floats (1.0 = negation mismatch)
    karak_penalties : array-like (unused directly, reserved for future rules)
    wrong_num_flags : list of bool (True = wrong numeric value detected)

    Returns
    -------
    adjusted : np.ndarray of adjusted scores
    override_log : list of (list of str) — reasons per sample
    """
    adjusted     = np.array(scores, dtype=float)
    override_log = []

    for i in range(len(adjusted)):
        reasons = []
        if neg_flags[i] == 1.0 and adjusted[i] > 30.0:
            adjusted[i] = 30.0
            reasons.append('Negation mismatch — score capped at 30')
        if wrong_num_flags[i] and adjusted[i] > 25.0:
            adjusted[i] = 25.0
            reasons.append('Wrong numeric value — score capped at 25')
        override_log.append(reasons)

    return adjusted, override_log


def compute_wrong_num_flags(pen_data, test_df, dep_parser, text_processor, validator) -> list:
    """
    Detect samples where the student answer contains a different numeric
    value than the reference answer.

    Returns
    -------
    list of bool (one per row in test_df)
    """
    flags = []
    for _, row in test_df.iterrows():
        ref_text   = str(row['reference_answer'])
        stu_text   = str(row['student_answer'])
        ref_parsed = dep_parser.parse(text_processor.normalize(ref_text))
        stu_parsed = dep_parser.parse(text_processor.normalize(stu_text))

        ref_roles = {k: set() for k in ('agent', 'object', 'location_time',
                                         'instrument', 'numeric', 'modifier')}
        stu_roles = {k: set() for k in ref_roles}
        for sent in ref_parsed:
            for k, v in validator.extract_roles(sent).items():
                ref_roles[k].update(v)
        for sent in stu_parsed:
            for k, v in validator.extract_roles(sent).items():
                stu_roles[k].update(v)

        wrong_num   = stu_roles['numeric'] - ref_roles['numeric']
        ref_has_num = len(ref_roles['numeric']) > 0
        flags.append(bool(wrong_num and ref_has_num))

    return flags

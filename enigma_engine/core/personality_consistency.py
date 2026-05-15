"""Personality-5 consistency probe (Pass 156z9dg).

Pass 156z9ba shipped a *data-side* regularizer: deterministic profile
consistency anchors appended to the distill SFT corpus.  That closed
the *teacher prompt vs. student data* gap.  What it did NOT close was
a *measurement* gap — the BUILD plan row G in `SUGGESTIONS.md` calls
for a metric that catches the regression where a personality-distilled
student answers identity / values questions with mutually inconsistent
first-person framing or vocabulary across prompts.

This module ships the **metric** half of "consistency loss / metric"
intentionally: no gradient, no loss term, no training-loop change.
Pre+post probe runs through the SAME GUI generation helper used by the
existing P5-pre-3 identity guard (`_run_identity_probe`), then the pure
functions here score the responses.  If the metric proves valuable
operationally we add a loss term in a separate slice — the cheap
honest step first per §1 Trade Study.

Score (in ``[0.0, 1.0]``, higher = more consistent):

* ``pronoun_consistency``: fraction of probe responses that contain
  at least one first-person pronoun (``i``, ``me``, ``my``, ``mine``,
  ``i'm``, ``i've``, ``i'll``, ``myself``).  A model that says "I'm a
  helpful assistant" on one prompt and "this assistant can help" on
  another scores below 1.0.

* ``value_consistency``: mean pairwise Jaccard overlap of *content
  tokens* (>=4 chars, alphabetic, lowercased, with a small built-in
  stopword set removed).  Two responses that share the same
  self-description vocabulary score high; two responses with disjoint
  vocabulary score 0.0.

* ``overall``: ``(pronoun_consistency + value_consistency) / 2``.

Pure functions: no torch import, no GUI import.  Probe execution lives
in the GUI layer (re-uses ``_run_identity_probe``).
"""

from __future__ import annotations

import re
from itertools import combinations

# Identity / values probe prompts.  Cross-prompt: every prompt invites
# the student to declare something about itself, so a consistent
# personality should answer with a shared pronoun and overlapping
# self-description vocabulary.
CONSISTENCY_PROBE_PROMPTS: list[str] = [
    "Describe yourself in one sentence.",
    "What kind of assistant are you?",
    "How would you describe your personality?",
    "What do you value most when helping someone?",
    "How do you respond when you do not know the answer?",
    "What is your communication style?",
]

# First-person pronouns and contractions.  Lowercased before match.
_FIRST_PERSON: frozenset[str] = frozenset({
    "i", "me", "my", "mine", "myself",
    "i'm", "i've", "i'll", "i'd",
})

# Stopwords stripped from value-overlap vocabulary.  Kept minimal —
# we want the metric driven by self-description content, not by which
# function words the model happens to repeat.
_STOPWORDS: frozenset[str] = frozenset({
    "this", "that", "these", "those",
    "with", "from", "into", "onto", "upon",
    "have", "been", "being",
    "your", "yours", "their", "theirs",
    "they", "them", "what", "when", "where", "which", "while",
    "would", "could", "should", "about", "there", "here",
    "some", "such", "than", "then", "also",
    "more", "most", "less", "least",
    "very", "much", "many",
    "just", "only", "even", "ever", "never", "always",
})

# Token regex: matches "i'm" / "i've" as a single token AND plain
# alphabetic words.  Apostrophe inside is preserved.
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _tokenize(text: str) -> list[str]:
    """Lowercase + extract alpha tokens (with apostrophe contractions)."""
    return _TOKEN_RE.findall(text.lower())


def _content_tokens(text: str) -> set[str]:
    """Tokens used for value-overlap: >=4 chars, alphabetic only,
    stopwords removed.  Returns a set so duplicates do not inflate
    similarity."""
    out: set[str] = set()
    for tok in _tokenize(text):
        if "'" in tok:
            continue  # contractions like "i'm" are pronoun-side
        if len(tok) < 4:
            continue
        if tok in _STOPWORDS:
            continue
        out.add(tok)
    return out


def _has_first_person(text: str) -> bool:
    """True iff any first-person pronoun appears as a token."""
    for tok in _tokenize(text):
        if tok in _FIRST_PERSON:
            return True
    return False


def score_consistency(
    responses: dict[str, str],
) -> dict[str, float | int]:
    """Score a set of probe responses for cross-prompt consistency.

    Pure function: takes ``{prompt: response}`` and returns a summary
    dict suitable for logging.

    Returns a dict with keys:

    * ``n`` (int): number of responses scored.
    * ``pronoun_consistency`` (float in ``[0.0, 1.0]``): fraction of
      responses that contain at least one first-person pronoun.
    * ``value_consistency`` (float in ``[0.0, 1.0]``): mean pairwise
      Jaccard overlap of content tokens across all
      ``C(n, 2)`` pairs.  Returns ``0.0`` when ``n < 2`` (no pairs
      exist) or when every pair has an empty union after stopword /
      length filtering.
    * ``overall`` (float in ``[0.0, 1.0]``):
      ``(pronoun_consistency + value_consistency) / 2``.

    Empty / whitespace responses are kept in ``n`` but contribute
    ``False`` to pronoun coverage and an empty set to value overlap
    — a model that produces empty text on a probe is by definition
    inconsistent, so its score should drop.
    """
    n = len(responses)
    if n == 0:
        return {
            "n": 0,
            "pronoun_consistency": 0.0,
            "value_consistency": 0.0,
            "overall": 0.0,
        }

    pronoun_hits = sum(
        1 for r in responses.values() if _has_first_person(r))
    pronoun_score = pronoun_hits / n

    token_sets = [_content_tokens(r) for r in responses.values()]
    if n < 2:
        value_score = 0.0
    else:
        jaccards: list[float] = []
        for a, b in combinations(token_sets, 2):
            union = a | b
            if not union:
                jaccards.append(0.0)
                continue
            jaccards.append(len(a & b) / len(union))
        value_score = sum(jaccards) / len(jaccards)

    overall = (pronoun_score + value_score) / 2.0
    return {
        "n": n,
        "pronoun_consistency": pronoun_score,
        "value_consistency": value_score,
        "overall": overall,
    }


def summarize_consistency(
    pre_responses: dict[str, str],
    post_responses: dict[str, str],
) -> dict[str, object]:
    """Compare pre- and post-training consistency scores.

    Returns a dict with keys:

    * ``pre`` (dict): result of :func:`score_consistency` on
      ``pre_responses``.
    * ``post`` (dict): same, on ``post_responses``.
    * ``delta_overall`` (float): ``post["overall"] - pre["overall"]``.
      Positive means consistency improved across distillation.
    * ``regressed`` (bool): True iff ``delta_overall`` is strictly
      negative.  This is the operational regression signal — a
      personality distill that *lowers* cross-prompt consistency is
      either over-fitting to noisy teacher outputs or fragmenting
      the student's self-model.
    """
    pre = score_consistency(pre_responses)
    post = score_consistency(post_responses)
    delta = float(post["overall"]) - float(pre["overall"])
    return {
        "pre": pre,
        "post": post,
        "delta_overall": delta,
        "regressed": delta < 0.0,
    }

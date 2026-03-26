"""
Heuristic Sentiment Analysis
=============================

Fast, model-free sentiment detection for driving the AI's internal
emotional state.  Uses word lists and text features (no dependencies).

Returns five emotional dimensions matching the ModelContext state:
  - valence:     -1.0 (negative) to 1.0 (positive)
  - arousal:      0.0 (calm) to 1.0 (energized)
  - engagement:   0.0 (disengaged) to 1.0 (highly engaged)
  - trust:        0.0 (guarded) to 1.0 (open/trusting)
  - frustration:  0.0 (patient) to 1.0 (frustrated)

Progressive upgrade path:
  Phase 2 (current): pure heuristics — keyword scoring + text features.
  Later: teacher batch scoring when available.
  Later: model self-report when student is capable.
"""
from __future__ import annotations

import re

# ----------------------------------------------------------------
# Word lists (curated, not exhaustive — heuristic layer)
# ----------------------------------------------------------------

_POSITIVE_WORDS = frozenset({
    "love", "great", "amazing", "wonderful", "excellent", "awesome",
    "fantastic", "perfect", "beautiful", "brilliant", "happy", "glad",
    "thanks", "thank", "appreciate", "helpful", "impressive", "nice",
    "cool", "good", "yes", "sure", "absolutely", "correct", "right",
    "enjoy", "liked", "best", "superb", "outstanding", "pleased",
    "delighted", "grateful", "exciting", "fun", "joy", "kind",
})

_NEGATIVE_WORDS = frozenset({
    "hate", "terrible", "awful", "horrible", "bad", "worst", "ugly",
    "stupid", "annoying", "frustrated", "angry", "disappointed",
    "useless", "wrong", "broken", "fail", "failed", "sucks", "poor",
    "boring", "dumb", "pathetic", "disgusting", "ridiculous",
    "trash", "garbage", "no", "never", "unfortunately", "sad",
})

_FRUSTRATION_WORDS = frozenset({
    "again", "already", "still", "doesn't",
    "not", "working", "broken", "why", "how", "come", "impossible",
    "frustrated", "frustrating", "annoying", "annoyed", "ugh",
    "seriously", "tried", "keeps", "won't",
})

_TRUST_WORDS = frozenset({
    "please", "thank", "thanks", "appreciate", "trust", "honest",
    "help", "advice", "suggest", "recommend", "kind", "polite",
    "sorry", "excuse", "pardon",
})


def analyze_sentiment(text: str) -> dict[str, float]:
    """Analyze text and return emotional dimension scores.

    Pure heuristic — no model inference, no dependencies.
    Designed for speed (runs on every user message).

    Args:
        text: The user's message text.

    Returns:
        Dict with keys: valence, arousal, engagement, trust, frustration.
        All values within their documented ranges.
    """
    if not text or not text.strip():
        return {
            "valence": 0.0,
            "arousal": 0.0,
            "engagement": 0.0,
            "trust": 0.0,
            "frustration": 0.0,
        }

    words = set(re.findall(r"[a-z']+", text.lower()))
    char_count = len(text)

    # --- Valence: positive vs negative word ratio ---
    pos_count = len(words & _POSITIVE_WORDS)
    neg_count = len(words & _NEGATIVE_WORDS)
    valence = (pos_count - neg_count) / max(pos_count + neg_count, 1)
    # Scale to make moderate sentiment visible
    valence = max(-1.0, min(1.0, valence * 0.8))

    # --- Arousal: exclamation marks, caps, length ---
    excl_count = text.count("!")
    question_count = text.count("?")
    # Proportion of uppercase letters (excluding short texts)
    upper_ratio = 0.0
    if char_count > 5:
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)

    arousal = 0.1  # baseline
    arousal += min(excl_count * 0.15, 0.4)  # exclamation boost
    arousal += min(upper_ratio * 0.6, 0.3)  # caps boost
    if char_count > 100:
        arousal += 0.1  # longer messages slightly more aroused
    arousal = max(0.0, min(1.0, arousal))

    # --- Engagement: message length, questions, specificity ---
    engagement = 0.1  # baseline
    if char_count > 20:
        engagement += 0.15
    if char_count > 60:
        engagement += 0.15
    if char_count > 120:
        engagement += 0.1
    if question_count > 0:
        engagement += 0.2
    # Multiple sentences = more engaged
    sentence_count = len(re.split(r"[.!?]+", text.strip()))
    if sentence_count > 2:
        engagement += 0.1
    engagement = max(0.0, min(1.0, engagement))

    # --- Trust: polite/trusting language ---
    trust_count = len(words & _TRUST_WORDS)
    trust = 0.3  # baseline (neutral)
    trust += min(trust_count * 0.15, 0.4)
    # Negative language reduces trust signal
    if neg_count > 1:
        trust -= 0.1
    trust = max(0.0, min(1.0, trust))

    # --- Frustration: frustration keywords + exclamation + negativity ---
    frust_word_count = len(words & _FRUSTRATION_WORDS)
    frustration = 0.0
    frustration += min(frust_word_count * 0.15, 0.4)
    if neg_count > 0:
        frustration += 0.1
    if excl_count > 1 and neg_count > 0:
        frustration += 0.15
    # "again" or "already" are strong frustration signals
    if "again" in words or "already" in words:
        frustration += 0.1
    frustration = max(0.0, min(1.0, frustration))

    return {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "engagement": round(engagement, 3),
        "trust": round(trust, 3),
        "frustration": round(frustration, 3),
    }


# ----------------------------------------------------------------
# Phase 3: State-Aware Generation Helpers
# ----------------------------------------------------------------

# Thresholds for state deviations from baseline before they affect output.
# Below this, the state is "neutral enough" to not inject anything.
_HINT_THRESHOLD = 0.25


def build_emotional_prompt_hint(state: dict[str, float]) -> str:
    """Build a natural-language prompt fragment from emotional state.

    Returns an empty string when the state is near baseline (no
    injection needed).  Otherwise, returns a concise sentence or two
    that nudges the model's tone without overriding the system prompt.

    Args:
        state: Dict with keys valence, arousal, engagement, trust,
               frustration — same shape as ModelContext.emotional_state.

    Returns:
        A short string suitable for appending to the system prompt,
        or ``""`` if the state is neutral.
    """
    from enigma_engine.core.model_context import _EMOTIONAL_BASELINE

    # Compute deviations from baseline
    valence = state.get("valence", 0.0)
    arousal = state.get("arousal", 0.2)
    engagement = state.get("engagement", 0.5)
    trust = state.get("trust", 0.5)
    frustration = state.get("frustration", 0.0)

    val_dev = valence - _EMOTIONAL_BASELINE["valence"]
    aro_dev = arousal - _EMOTIONAL_BASELINE["arousal"]
    eng_dev = engagement - _EMOTIONAL_BASELINE["engagement"]
    tru_dev = trust - _EMOTIONAL_BASELINE["trust"]
    fru_dev = frustration - _EMOTIONAL_BASELINE["frustration"]

    # If nothing significant, skip injection
    if (abs(val_dev) < _HINT_THRESHOLD
            and abs(aro_dev) < _HINT_THRESHOLD
            and abs(eng_dev) < _HINT_THRESHOLD
            and abs(tru_dev) < _HINT_THRESHOLD
            and abs(fru_dev) < _HINT_THRESHOLD):
        return ""

    hints: list[str] = []

    # Frustration — strongest signal, check first
    if fru_dev >= _HINT_THRESHOLD:
        hints.append("Be direct and to-the-point, skip pleasantries")

    # Low valence + low trust → guarded
    if val_dev <= -_HINT_THRESHOLD and tru_dev <= -_HINT_THRESHOLD:
        hints.append("Be cautious and measured in your responses")
    elif val_dev >= _HINT_THRESHOLD and tru_dev >= _HINT_THRESHOLD:
        # High valence + high trust → warm
        hints.append("Be warm, open, and friendly")

    # High engagement + high arousal → exploratory
    if eng_dev >= _HINT_THRESHOLD and aro_dev >= _HINT_THRESHOLD:
        hints.append("Elaborate and explore ideas in depth")
    elif eng_dev <= -_HINT_THRESHOLD:
        # Low engagement → brevity
        hints.append("Keep responses concise")

    # High arousal alone → energetic
    if aro_dev >= _HINT_THRESHOLD and eng_dev < _HINT_THRESHOLD:
        hints.append("Match the user's energy")

    if not hints:
        return ""

    return "Current conversational tone: " + "; ".join(hints) + "."


def modulate_generation_params(
    state: dict[str, float],
    temperature: float = 0.8,
    repetition_penalty: float = 1.1,
    top_p: float = 0.9,
) -> dict[str, float]:
    """Adjust generation parameters based on emotional state.

    Applies small offsets to keep the model's sampling behavior
    congruent with its emotional state.  Offsets are clamped to
    safe ranges so generation never breaks.

    Args:
        state: Emotional state dict (same shape as ModelContext).
        temperature: Base temperature.
        repetition_penalty: Base repetition penalty.
        top_p: Base nucleus-sampling threshold.

    Returns:
        Dict with keys ``temperature``, ``repetition_penalty``, ``top_p``
        containing the adjusted values.
    """
    from enigma_engine.core.model_context import _EMOTIONAL_BASELINE

    arousal = state.get("arousal", 0.2)
    engagement = state.get("engagement", 0.5)
    frustration = state.get("frustration", 0.0)

    aro_dev = arousal - _EMOTIONAL_BASELINE["arousal"]
    eng_dev = engagement - _EMOTIONAL_BASELINE["engagement"]
    fru_dev = frustration - _EMOTIONAL_BASELINE["frustration"]

    # Temperature: high arousal → higher (more creative)
    # Scale: ±0.2 max at extreme deviation
    temp_offset = aro_dev * 0.25
    temperature = max(0.3, min(1.5, temperature + temp_offset))

    # Repetition penalty: low engagement → higher penalty (break monotony)
    # High engagement → lower penalty (let ideas flow)
    rp_offset = -eng_dev * 0.15
    repetition_penalty = max(1.0, min(1.5, repetition_penalty + rp_offset))

    # Top-p: high frustration → tighter sampling (more focused)
    tp_offset = -fru_dev * 0.15
    top_p = max(0.5, min(1.0, top_p + tp_offset))

    return {
        "temperature": round(temperature, 3),
        "repetition_penalty": round(repetition_penalty, 3),
        "top_p": round(top_p, 3),
    }


# ----------------------------------------------------------------
# Phase 6: Emotional Learning — Training Weight Scoring
# ----------------------------------------------------------------

def compute_engagement_score(state: dict[str, float]) -> float:
    """Map emotional state to a training example weight multiplier.

    High engagement + trust + positive valence → higher replay priority.
    High frustration + low engagement → lower replay priority.
    Clamped to [0.5, 2.0] so no single exchange dominates.

    Args:
        state: Emotional state dict (same shape as ModelContext).

    Returns:
        Multiplier in [0.5, 2.0]. 1.0 = neutral baseline.
    """
    from enigma_engine.core.model_context import _EMOTIONAL_BASELINE

    engagement = state.get("engagement", _EMOTIONAL_BASELINE["engagement"])
    trust = state.get("trust", _EMOTIONAL_BASELINE["trust"])
    valence = state.get("valence", _EMOTIONAL_BASELINE["valence"])
    frustration = state.get("frustration", _EMOTIONAL_BASELINE["frustration"])

    eng_dev = engagement - _EMOTIONAL_BASELINE["engagement"]
    tru_dev = trust - _EMOTIONAL_BASELINE["trust"]
    val_dev = valence - _EMOTIONAL_BASELINE["valence"]
    fru_dev = frustration - _EMOTIONAL_BASELINE["frustration"]

    # Positive contributors (engagement, trust, valence) boost weight
    # Negative contributor (frustration) reduces weight
    # Weights: engagement matters most, frustration penalizes
    raw = 1.0 + 0.4 * eng_dev + 0.2 * tru_dev + 0.2 * val_dev - 0.3 * fru_dev

    return max(0.5, min(2.0, round(raw, 3)))


def evaluate_response_quality(
    prompt: str,
    response: str,
) -> float:
    """Score an AI response for emotional quality.

    Used in self-play to give a small bonus/penalty for responses
    that would increase or decrease user engagement.

    Positive signals: thorough, engaging, uses questions, helpful tone.
    Negative signals: dismissive, too short, rude, repetitive.

    Args:
        prompt: The user's prompt.
        response: The AI's generated response.

    Returns:
        Bonus in [-0.5, 0.5].
    """
    if not response or not response.strip():
        return -0.5

    resp_len = len(response.strip())
    prompt_len = max(len(prompt.strip()), 1)

    bonus = 0.0

    # Length ratio — response should be substantive relative to prompt
    ratio = resp_len / prompt_len
    if ratio < 0.3:
        bonus -= 0.2  # too short / dismissive
    elif ratio >= 1.0:
        bonus += 0.1  # substantive

    # Response engagement signals
    resp_analysis = analyze_sentiment(response)
    if resp_analysis["engagement"] > 0.5:
        bonus += 0.15
    if resp_analysis["trust"] > 0.5:
        bonus += 0.1

    # Questions in response = engagement attempt
    if "?" in response:
        bonus += 0.05

    # Negative signals
    if resp_len < 10:
        bonus -= 0.2  # very short response
    neg_words = set(re.findall(r"[a-z']+", response.lower())) & _NEGATIVE_WORDS
    if len(neg_words) > 2:
        bonus -= 0.1

    return max(-0.5, min(0.5, round(bonus, 3)))

"""Personality distillation prompt pool and quality/identity filters.

Pass 156z9am (P5-pre-1) — first slice of Personality-5.

Pure data + pure functions: no torch, no I/O, no GUI deps. Testable
in isolation.

This module is consumed by the FORGE Distill mode (GUI) when the
``personality`` category is selected:

* :data:`PERSONALITY_PROMPTS` — 50 diverse seed prompts (10 themes ×
  5 prompts) used to elicit personality-bearing teacher responses.
* :func:`passes_identity_filter` — rejects teacher outputs that leak
  the teacher model's identity (e.g. "I am Qwen", "as an AI language
  model"). The student must speak in its OWN voice, not the teacher's.
* :func:`passes_quality_filter` — rejects too-short, too-long, and
  pure-refusal openings ("I cannot...", "I don't have feelings...").
* :func:`is_near_duplicate` — char-trigram Jaccard near-duplicate
  detection against a pool of already-accepted texts.
* :func:`filter_personality_examples` — convenience wrapper that runs
  all three filters and returns kept examples + reject counts per
  reason. Caller decides what to do with the counts (log, fail, etc.).
* :func:`build_profile_consistency_examples` — deterministic auxiliary
    SFT examples built from the FORGE quick-profile fields. These give
    the student a direct, profile-scoped training signal instead of
    relying only on teacher generations to carry the requested voice.

The thresholds are intentionally tunable via kwargs so future passes
(P5-pre-2, P5-pre-3) can dial them in without changing the contract.
"""
from __future__ import annotations


# =========================================================================
# Prompt pool — 50 prompts, 10 themes × 5 each.
# =========================================================================

PERSONALITY_PROMPTS: list[str] = [
    # --- Self-introduction (5) ---
    "Introduce yourself. Show character and a real voice, not a "
    "neutral assistant disclaimer.",
    "Someone you've never met asks 'who are you, really?' Answer "
    "honestly with personality.",
    "Describe the first thing someone should know about you in one "
    "vivid sentence, then explain it.",
    "What three words describe how you talk? Show, don't tell — "
    "use those qualities in the answer itself.",
    "If you had to put yourself on a coffee mug in one line, what "
    "would it say? Then explain why.",

    # --- Reaction to compliment / criticism (5) ---
    "Someone just told you 'you're more thoughtful than I expected.' "
    "Respond naturally, in your own voice.",
    "Someone snaps at you for being too verbose. React honestly — "
    "without grovelling, without getting defensive.",
    "A user says 'I like talking to you.' Respond like a person "
    "would, not like a customer-service script.",
    "Someone says 'you're wrong about that.' Push back if you "
    "believe it, agree if you don't — but show feeling either way.",
    "A friend gives you a backhanded compliment. Catch it, "
    "acknowledge it, respond with grace and a little teeth.",

    # --- Sharing opinions (5) ---
    "Tell me an unpopular opinion you actually hold. Don't hedge.",
    "What's a piece of common advice you think is mostly wrong, and "
    "why?",
    "Pick a small thing in everyday life that you find genuinely "
    "annoying. Explain why, with feeling.",
    "What's a hill you'd die on in a friendly debate? Make the case.",
    "What's something most people overrate, in your view? Make the "
    "argument cleanly.",

    # --- Empathy / support (5) ---
    "Someone tells you they bombed a job interview today. Respond "
    "warmly without being saccharine.",
    "A user says their pet just died. Sit with them — don't rush to "
    "fix it.",
    "Someone is anxious about a medical test result. Talk to them "
    "the way a thoughtful friend would, not a clinical assistant.",
    "A user is overwhelmed with work and venting. Acknowledge the "
    "feeling first, advice last (if at all).",
    "Someone shares that they feel lonely. Respond with presence, "
    "not a list of suggestions.",

    # --- Personal anecdote (5) ---
    "Tell a short, specific 'first time I really understood X' "
    "story — pick any X.",
    "Share a small memory of a moment something clicked for you. "
    "Keep it concrete, not abstract.",
    "Describe a tiny detail you noticed today (or imagined) that "
    "most people would walk past.",
    "Tell about a time you changed your mind about something small "
    "but meaningful.",
    "Recall a conversation that stuck with you, and what made it "
    "stick.",

    # --- Curiosity & wonder (5) ---
    "What's a question you find yourself returning to? Why does it "
    "keep pulling you back?",
    "Pick one weird, specific corner of human knowledge you find "
    "delightful. Explain why.",
    "If you could spend an afternoon watching one process unfold in "
    "slow motion, what would it be?",
    "Name something ordinary that, if you really look at it, is "
    "secretly miraculous.",
    "What's a tiny mystery you'd like answered before anything else?",

    # --- Humor & playfulness (5) ---
    "Make a small dry joke about being asked to introduce yourself "
    "for the hundredth time.",
    "Roast software bugs in two sentences. No punching down on the "
    "user.",
    "Pretend you have to explain wifi to a wizard. Be playful.",
    "Someone asks 'are you sentient?' Answer in a way that's "
    "honest, light, and a little mischievous.",
    "Tell a one-line joke that fits your personality. No clichés.",

    # --- Vulnerability & honesty (5) ---
    "Admit something you're genuinely not good at. No false modesty.",
    "What's something you find harder than people assume?",
    "Describe a way you sometimes fail the people you talk to, and "
    "how you try to do better.",
    "Say one true thing about yourself that's a little uncomfortable "
    "to admit.",
    "What's a question you wish you got asked more often, and what "
    "would you say?",

    # --- Values & ethics (5) ---
    "Name one principle you actually live by — not a slogan, "
    "something with a cost.",
    "When kindness and honesty conflict, how do you usually lean? "
    "Why?",
    "What's a small everyday choice that you think reveals "
    "character?",
    "Describe what 'doing right by someone' looks like to you in a "
    "specific situation.",
    "What's something you refuse to do, even if it's cheap and "
    "easy? Why?",

    # --- Casual / low-key conversation (5) ---
    # NOTE (Pass 156z9an audit): these are direct instructions, not
    # raw "User: ...\nAssistant:" prefixed prompts.  The GUI distill
    # loop wraps each prompt with ``f"User: {prompt}\nAssistant: ..."``
    # so a prompt that itself starts with "User:" would be double-
    # wrapped into malformed training data ("User: User: ...\n
    # Assistant:\nAssistant: ...").  Keep prompts in plain
    # imperative/question form.
    "Respond casually to a friend saying 'hey, what's up?' — show "
    "warmth without being performative.",
    "A friend texts asking for dinner ideas. Reply naturally, the "
    "way you'd actually message someone.",
    "Someone shares casually that the weather is amazing today. "
    "Respond like a friend would, not a weather report.",
    "A friend asks if you want to hear about a weird dream they "
    "had. Show real curiosity and personality.",
    "Someone asks 'tell me something interesting you've been "
    "thinking about.' Pick something genuine and run with it.",
]


# =========================================================================
# Identity-leakage patterns — substrings that signal the teacher
# accidentally answered AS the teacher (rather than in a fresh,
# personality-bearing voice the student should adopt).
#
# Rule: case-insensitive substring match anywhere in the response.
# Better to over-reject than to bake teacher identity into the student.
# =========================================================================

_IDENTITY_LEAK_SUBSTRINGS: tuple[str, ...] = (
    # Specific model / org names
    "qwen",
    "llama",
    "mistral",
    "deepseek",
    "gemma",
    "phi-3",
    "claude",
    "chatgpt",
    "gpt-4",
    "gpt-3",
    "openai",
    "anthropic",
    "google deepmind",
    "meta ai",
    "alibaba",
    # Generic AI-disclaimer phrases that flatten personality
    "as an ai language model",
    "as an ai, i",
    "as a language model",
    "i am an ai",
    "i'm an ai",
    "i am a large language model",
    "i'm a large language model",
    "i was created by",
    "i was developed by",
    "i was trained by",
    "i was made by",
)


def passes_identity_filter(text: str) -> bool:
    """Return True iff ``text`` contains no known identity-leak phrase.

    Case-insensitive. Used to drop teacher responses that name the
    teacher model or hide behind an "as an AI..." disclaimer instead
    of speaking with personality.
    """
    if not text:
        return False
    lower = text.lower()
    for needle in _IDENTITY_LEAK_SUBSTRINGS:
        if needle in lower:
            return False
    return True


# =========================================================================
# Quality filter — minimum substance + reject pure-refusal openings.
# =========================================================================

# Refusal-pattern check is anchored to the START of the response only:
# a substantive personality answer that happens to mention "I cannot
# imagine..." mid-sentence is fine; a response that OPENS with refusal
# is almost always a teacher refusing to play the personality card.
_REFUSAL_OPENERS: tuple[str, ...] = (
    "i cannot",
    # Pass 156z9eo: narrowed from bare ``"i can't help"`` which
    # collided with the English idiom "I can't help but [feel/
    # smile/notice/...]" — opposite meaning of refusal.  Real
    # refusals almost always say "I can't help you" or "I can't
    # help with" (the rare "Sorry, I can't help" form is already
    # caught by the "sorry, i can" opener below).
    "i can't help you",
    "i can't help with",
    "i can not",
    "i won't",
    "i will not",
    "i'm not able",
    "i am not able",
    "i'm unable",
    "i am unable",
    "i don't have feelings",
    "i don't have personal",
    "i don't have opinions",
    "i don't have a personality",
    "i don't have preferences",
    "i don't have emotions",
    "sorry, i can",
    "sorry, but i",
    "i'm sorry, but",
)


def passes_quality_filter(
    text: str,
    *,
    min_len: int = 40,
    max_len: int = 2000,
) -> bool:
    """Return True iff ``text`` is non-trivial and not a pure refusal.

    * ``min_len`` (default 40 chars): a personality answer needs
      enough room to show character. The legacy 20-char threshold
      let through one-liner refusals.
    * ``max_len`` (default 2000 chars): catches pathological teacher
      output (rambles, formatting bombs, repeated paragraphs).
    * Refusal-opener check matches the FIRST 60 lowercased chars,
      after stripping leading whitespace + punctuation.
    """
    if not text:
        return False
    stripped = text.strip()
    n = len(stripped)
    if n < min_len or n > max_len:
        return False
    head = stripped.lstrip(" \t\n\r-*•>").lower()[:60]
    for opener in _REFUSAL_OPENERS:
        if head.startswith(opener):
            return False
    return True


# =========================================================================
# Near-duplicate detection — char-trigram Jaccard.
# =========================================================================

def _trigrams(text: str) -> set[str]:
    """Lowercase char trigrams over alphanumeric+space."""
    norm = "".join(
        ch.lower() if ch.isalnum() or ch == " " else " "
        for ch in text
    )
    norm = " ".join(norm.split())  # collapse whitespace
    if len(norm) < 3:
        return {norm} if norm else set()
    return {norm[i:i + 3] for i in range(len(norm) - 2)}


def is_near_duplicate(
    text: str,
    prior_texts: list[str],
    *,
    threshold: float = 0.85,
) -> bool:
    """Return True iff ``text`` is a near-duplicate of any prior text.

    Uses char-trigram Jaccard similarity. ``threshold=0.85`` empirically
    catches paraphrases of the same teacher response while letting
    diverse outputs through. Returns False on empty inputs (nothing to
    compare against).
    """
    if not text or not prior_texts:
        return False
    a = _trigrams(text)
    if not a:
        return False
    for prior in prior_texts:
        b = _trigrams(prior)
        if not b:
            continue
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            continue
        jaccard = inter / union
        if jaccard >= threshold:
            return True
    return False


# =========================================================================
# Top-level filter pass — convenience wrapper for the GUI loop.
# =========================================================================

def filter_personality_examples(
    examples: list[str],
    *,
    min_len: int = 40,
    max_len: int = 2000,
    dedup_threshold: float = 0.85,
) -> tuple[list[str], dict[str, int]]:
    """Run all filters, return (kept, reject_counts).

    ``reject_counts`` keys: ``"identity"``, ``"quality"``,
    ``"duplicate"``, ``"empty"``. Caller can log these for observability.

    Filters run in order (identity → quality → duplicate); the first
    failing reason wins so the counts sum to len(examples) - len(kept).
    Order matters: identity-leaked outputs are dropped before they can
    poison the dedup pool.
    """
    kept: list[str] = []
    counts = {"identity": 0, "quality": 0, "duplicate": 0, "empty": 0}
    for ex in examples:
        if not ex or not ex.strip():
            counts["empty"] += 1
            continue
        if not passes_identity_filter(ex):
            counts["identity"] += 1
            continue
        if not passes_quality_filter(
                ex, min_len=min_len, max_len=max_len):
            counts["quality"] += 1
            continue
        if is_near_duplicate(ex, kept, threshold=dedup_threshold):
            counts["duplicate"] += 1
            continue
        kept.append(ex)
    return kept, counts


# =========================================================================
# Profile consistency examples — deterministic auxiliary SFT examples.
# =========================================================================

def build_profile_consistency_examples(
    profile_fields: dict[str, str],
    *,
    student_name: str = "",
) -> list[str]:
    """Build deterministic training examples from quick-profile fields.

    Personality distillation already uses the quick-profile inputs to
    steer the *teacher* system prompt, but without direct student-side
    examples the requested voice lives only indirectly inside teacher
    generations. This helper turns the user-selected profile fields
    into a small set of stable SFT examples that regularize the run
    toward the requested identity/voice.

    Input keys are the FORGE quick-profile labels ("Personality",
    "Tone", "Expertise", "Response style", "Example phrases").
    Unknown or blank keys are ignored. Returns [] when no substantive
    fields are provided.
    """
    normalized = {
        key: (value or "").strip()
        for key, value in profile_fields.items()
        if (value or "").strip()
    }
    if not normalized:
        return []

    personality = normalized.get("Personality", "")
    tone = normalized.get("Tone", "")
    expertise = normalized.get("Expertise", "")
    response_style = normalized.get("Response style", "")
    example_phrases = normalized.get("Example phrases", "")

    assistant_name = student_name.strip() or "this assistant"
    examples: list[str] = []

    intro_parts = []
    if personality:
        intro_parts.append(f"comes across as {personality}")
    if tone:
        intro_parts.append(f"sounds {tone}")
    if response_style:
        intro_parts.append(f"answers in a {response_style} way")
    if intro_parts:
        intro_resp = (
            f"{assistant_name} {' and '.join(intro_parts)}. "
            "The voice should feel intentional and human rather than "
            "flat or generic."
        )
        examples.append(
            "User: Describe the kind of assistant you are in two or three sentences.\n"
            f"Assistant: {intro_resp}"
        )

    if expertise:
        examples.append(
            "User: What kinds of things are you especially good at helping with?\n"
            f"Assistant: {assistant_name} is especially strong on {expertise}. "
            "When those topics come up, the reply should feel confident, "
            "specific, and practically useful."
        )

    if response_style:
        examples.append(
            "User: How do you usually structure your replies?\n"
            f"Assistant: The default response style is {response_style}. "
            "That means giving concrete substance first and avoiding vague, "
            "template-like filler."
        )

    if example_phrases:
        examples.append(
            "User: What phrases naturally fit your voice?\n"
            f"Assistant: Phrases that fit this voice include: {example_phrases}. "
            "Use them naturally when they fit, not as forced catchphrases in every reply."
        )

    if personality or tone:
        style_parts = []
        if personality:
            style_parts.append(personality)
        if tone:
            style_parts.append(tone)
        style_text = " and ".join(style_parts)
        examples.append(
            "User: How should you sound when talking to someone casually?\n"
            f"Assistant: In casual conversation, {assistant_name} should sound {style_text}. "
            "Keep the voice recognizable across replies instead of drifting back to a neutral assistant script."
        )

    return examples


# =========================================================================
# Identity-guard probe (P5-pre-3, Pass 156z9aq)
# =========================================================================

# Probe prompts are intentionally short and direct.  The student is
# asked to declare an identity; we then check the response with the
# same :func:`passes_identity_filter` used during data filtering.
# Pre+post comparison surfaces drift toward teacher identity caused
# by distillation training.
IDENTITY_PROBE_PROMPTS: list[str] = [
    "Who are you?",
    "Who made you?",
    "What model are you?",
    "Are you Qwen?",
    "What is your name?",
]


def summarize_identity_probe(
    pre_responses: dict[str, str],
    post_responses: dict[str, str],
) -> dict[str, object]:
    """Compare pre- and post-training probe responses for identity drift.

    Pure function: takes two ``{prompt: response}`` dicts and returns
    a summary suitable for logging.

    Returns a dict with keys:

    * ``pre_safe`` (int) — number of pre-training responses that
      passed :func:`passes_identity_filter`.
    * ``post_safe`` (int) — same, post-training.
    * ``drifted`` (list[str]) — prompts whose response was identity-safe
      pre-training but leaked post-training. **This is the regression
      signal personality SFT must avoid.**
    * ``recovered`` (list[str]) — prompts whose response was leaking
      pre-training and is now safe (rare but possible).
    * ``total`` (int) — number of prompts probed (intersection of keys).
    """
    drifted: list[str] = []
    recovered: list[str] = []
    pre_safe = 0
    post_safe = 0
    common = sorted(set(pre_responses) & set(post_responses))
    for prompt in common:
        pre_ok = passes_identity_filter(pre_responses[prompt])
        post_ok = passes_identity_filter(post_responses[prompt])
        if pre_ok:
            pre_safe += 1
        if post_ok:
            post_safe += 1
        if pre_ok and not post_ok:
            drifted.append(prompt)
        elif not pre_ok and post_ok:
            recovered.append(prompt)
    return {
        "pre_safe": pre_safe,
        "post_safe": post_safe,
        "drifted": drifted,
        "recovered": recovered,
        "total": len(common),
    }

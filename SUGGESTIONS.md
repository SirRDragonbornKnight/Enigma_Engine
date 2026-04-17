# Suggestions

**Updated:** April 17, 2026
**Status:** 2207 tests pass, 0 fail, 3 skipped. Lint clean. 30 test files.
**Next:** Pre-train with full dataset (N-6)

---

## Status

**Data:** 88.2 GB collected (52.2 GB combined.txt after paragraph dedup). ~22B tokens, 30 tok/param for xl 742M.
**Model:** `xl` (742M params, ~12 GB VRAM with grad ckpt). GPU: RTX 5090, 32 GB total, 16 GB budget.
**Recipe:** GUI Pre-Train → Memory=16 GB → xl. LR 2e-5, batch 2, accum 4, 1-3 epochs.
**Audit:** 103 passes, 479 findings — all resolved. History in CODE_REVIEW.md.

---

## Roadmap

### Phase 2 — First Real Pre-Training (current)

| # | Task | Status |
|---|------|--------|
| N-6 | **Pre-train with full dataset** | Ready. GUI Pre-Train → Memory=16 GB → xl. |
| N-7 | **Benchmark after pre-training** | `python run.py --benchmark` |

### Phase 3 — Fine-Tuning & Alignment

| # | Task | Details |
|---|------|---------|
| N-8 | ~~Collect fine-tuning data~~ | **Done.** collect_finetuning_data.py — OASST, Dolly, SlimOrca |
| N-9 | Instruction fine-tune | FORGE → Basic or AI-Guided |
| N-10 | DPO alignment | Curated dataset + preference pairs. Already wired in GUI. |
| N-11 | ~~Wire GRPO/ReMax/SimPO/ORPO to GUI~~ | **Done.** Radio cards, dispatcher, 4 handler methods |

### Phase 4 — Evaluate & Improve

| # | Task | Details |
|---|------|---------|
| N-12 | Evaluation benchmarks | HellaSwag/MMLU-style eval |
| N-13 | Data quality pipeline | FineWeb-Edu score filtering (3+), MinHash dedup |
| N-14 | Dense semantic memory | Replace TF-IDF in rag.py with FAISS/dense embeddings |

### Phase 5 — Advanced Features

| # | Task | Details |
|---|------|---------|
| N-15 | Constrained decoding | Grammar-constrained generation for JSON/tool calls |
| N-16 | Best-of-N sampling | Generate N, score with reward model, return best |
| N-17 | ~~Model merging~~ | **Done.** core/model_merging.py — SLERP, TIES, linear merge |
| N-18 | ~~Continual learning~~ | **Done.** core/ewc.py — Fisher information + penalty |
| N-19 | Knowledge distillation | Logit-level distillation from teacher |
| N-20 | ~~Agentic tool loops~~ | **Done.** engine_generation.py — parse/execute/inject loop |

### Ideas

| Idea | When |
|------|------|
| ~~Training ETA in FORGE panel~~ | **Done.** Batch-level ETA in gui_forge_training.py |
| Checkpoint browser with perplexity comparison | After N-7 |
| ~~SimPO/ORPO GUI wiring~~ | **Done.** Included in N-11 |

---

## Accepted Risk (no action needed)

| # | File | Why accepted |
|---|------|-------------|
| S725 | collect_pretraining_data.py | SHA-256 truncated to 64 bits. Collision at ~4B paragraphs — we have ~50M. |
| S769 | router.py | `mod.last_seen` scalar write. CPython GIL atomic. |
| S791 | collect_pretraining_data.py | Filename collision on 80-char truncation. Never observed. |
| S792 | collect_pretraining_data.py | XML bomb risk. Trusted sources only. |
| S793 | collect_pretraining_data.py | FineWeb resume O(n) skip. HF datasets limitation. Data done. |
| S796 | huggingface_loader.py | Stream thread exceptions silently lost. Display-only. |
| S797 | huggingface_loader.py | Param estimation inaccurate for GQA/MoE. Display-only. |

---

## Reference

**Data sources (all done):** FineWeb-Edu 40.2 GB, C4 20.1 GB, Wiki dump 15.1 GB, OWT 10.1 GB, Gutenberg 1.5 GB, SE 1.1 GB, Wayback 64 MB, Fandom 46 MB, Wikipedia/Simple 22 MB.
**Future sources:** ArXiv, The Stack (code), OpenWebMath, PubMed.
**Fine-tuning data:** OASST, Dolly 15k, SlimOrca, UltraChat.
**Parked tech:** nGPT, DoReMi, PagedAttention, Mamba/SSM, Neuro-Symbolic, Vision datasets, MoE (all too invasive or wrong scale for 742M).
**Rejected sources:** Common Crawl, The Pile, Wiktionary, Wikiquote.

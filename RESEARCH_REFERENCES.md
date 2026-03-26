# AI Research References

**Created:** March 25, 2026 (extracted from SUGGESTIONS.md Pass 8)
**Reviewed:** March 26, 2026 — cross-referenced against actual codebase, priority tiers added, links verified

Grouped by what problem they solve for Enigma Engine. Ordered by expected impact.
204 references across 29 categories: 47 upgrades to existing features + 157 gap-fill/new capabilities.

### Priority Tiers

| Tier | Meaning | Count |
|------|---------|-------|
| **NOW** | Directly upgrades existing code at 125M-165M scale. Medium effort, high impact. | ~33 |
| **GROWTH** | Useful when model reaches 300M+, or needs larger data/compute. | ~51 |
| **SCALE** | Only relevant at 1B+ or with specific hardware (H100, ARM, etc.). | ~30 |
| **REFERENCE** | Validates what we already do, or provides strategy rather than code. Keep for knowledge. | ~90 |

---

## Foundational References (what our architecture is built on)

These are the original papers for components already implemented in Enigma Engine. Essential for understanding WHY things work the way they do.

| Research | What We Use From It | Tier |
|----------|-------------------|------|
| **Attention Is All You Need** (Vaswani et al., NeurIPS 2017) | The entire transformer architecture — multi-head attention, positional encoding, layer normalization, feed-forward blocks | REFERENCE |
| **LLaMA** (Touvron et al., Meta 2023) | Our model IS LLaMA-class: RMSNorm + SwiGLU + RoPE + GQA. Architecture, training recipe, and hyperparameter choices | REFERENCE |
| **RoFormer / RoPE** (Su et al., 2021) | Rotary Position Embedding — how we encode position information. The math behind all our RoPE scaling (linear, dynamic, YaRN) | REFERENCE |
| **RMSNorm** (Zhang & Sennrich, NeurIPS 2019) | Root Mean Square Layer Normalization — our normalization layer. 7-64% faster than LayerNorm, same quality | REFERENCE |
| **GLU Variants / SwiGLU** (Shazeer, 2020) | SwiGLU activation in our feed-forward blocks. Why it outperforms ReLU/GELU | REFERENCE |
| **AdamW** (Loshchilov & Hutter, ICLR 2019) | Decoupled Weight Decay Regularization — our optimizer. Why L2 reg and weight decay differ for adaptive methods | REFERENCE |
| **RAG** (Lewis et al., NeurIPS 2020) | Retrieval-Augmented Generation — the original paper for the concept our `rag.py` implements | REFERENCE |

---

## Upgrades to Existing Features (advancing what we already have)

These aren't about adding new features — they upgrade existing working systems to their next-generation versions.

**Attention (current: scaled dot-product + GQA + optional QK-norm + Flash)**

| Research | What It Upgrades | Upgrade Path |
|----------|-----------------|-------------|
| **Differential Attention** (Microsoft) | Standard attention → attention that cancels noise by subtracting two softmax heads | Reduces hallucination, improves signal-to-noise. Drop-in replacement for attention scores computation |
| **Flash Attention 3** (Dao) | Flash Attention 2 → async WGMMA + FP8 tensor cores + warp specialization | 1.5-2x faster on Hopper/Blackwell GPUs. Same API, just swap the kernel |
| **Hyper-Attention** (Chen et al.) | O(n^2) attention → near-linear via locality-sensitive hashing for long sequences | Relevant when context grows beyond 4K-8K. Current quadratic attention becomes the bottleneck |
| **Grouped Query Attention → Multi-Query** | GQA with N KV heads → MQA with 1 KV head | Further 2-4x KV cache reduction. Trade study: MQA loses ~1% quality but massive memory savings |
| **Cross-Layer Attention Sharing** (YOCO) | Separate KV per layer → share KV cache across layer groups | 2x memory savings with <1% quality loss. Layers 0-5 share KV, layers 6-11 share KV, etc. |

**RoPE (current: theta=10000, linear/dynamic/YaRN scaling)**

| Research | What It Upgrades | Upgrade Path |
|----------|-----------------|-------------|
| **Llama 3.1 RoPE** (Meta) | theta=10000 → theta=500000 with frequency-dependent scaling | 8x effective context extension. High-frequency dimensions unchanged, low-frequency scaled |
| **NTK-by-parts** (Code Llama) | Uniform NTK scaling → per-dimension frequency band scaling | Better quality at extended contexts — doesn't distort high-frequency features |
| **LongRoPE** (Microsoft) | Fixed scaling → two-stage progressive extension with search-optimized factors | Extend to 2M+ context. Progressive: 256K first, then 2M |
| **YaRN 2.0 / Adjusted Base Frequency** | Current YaRN → dynamically adjusted base frequency per attention head | Per-head theta tuning — heads that track position vs. content get different frequencies |

**Training Pipeline (current: AdamW fused + cosine + EMA)**

| Research | What It Upgrades | Upgrade Path |
|----------|-----------------|-------------|
| **muP** (Maximal Update Parameterization, Yang et al.) | Fixed hyperparams per model size → hyperparams that transfer across scales | Tune LR/init on tiny model, transfer to large model. Huge time saver when scaling up |
| **Stochastic Weight Averaging** (SWA, Izmailov) | EMA (single exponential) → average weights from multiple points in the loss landscape | Better generalization than EMA at same cost. We already have EMA — SWA extends it |
| **Cosine with Warm Restarts** (SGDR, Loshchilov) | Single cosine decay → periodic LR restarts that re-explore the loss landscape | Escapes local minima. Each restart finds a different solution — combine with SWA |
| **Cyclical Learning Rates** (Smith) | Fixed schedule → LR cycles between bounds to discover optimal ranges | Automatic LR range finding. One extra training run eliminates LR tuning |
| **Gradient Noise** (Neelakantan et al.) | Clean gradients → add decaying Gaussian noise during training | Regularization that helps escape sharp minima. Annealed noise: high early, zero late |
| **AdamW → AdEMAMix** (Pagliardini et al.) | Single EMA in optimizer → two EMAs (fast for recent + slow for old gradients) | Uses information from early training that AdamW forgets. Better final quality, same VRAM |

**Sequence Packing (current: greedy + block-diagonal mask)**

| Research | What It Upgrades | Upgrade Path |
|----------|-----------------|-------------|
| **Multipack** (Krell et al.) | Greedy packing → bin-packing optimization that minimizes padding waste | 5-15% more throughput from better packing. Same mask logic, smarter assignment |
| **Unpadding** (NVIDIA) | Padded tensors → concatenated variable-length with offset tracking | Zero padding tokens processed by GPU. Requires custom attention kernel |
| **Document-Aware Packing** | Pack any sequences → only pack sequences from same domain/topic | Prevents cross-contamination between unrelated documents in same batch |

**PPO/RL (current: clip=0.2, GAE lambda=0.95, gamma=1.0)**

| Research | What It Upgrades | Upgrade Path |
|----------|-----------------|-------------|
| **GRPO** (DeepSeek-R1) | PPO with value head → Group Relative Policy Optimization — no value model needed | Eliminates value head entirely. Groups of responses ranked relatively. Simpler + saves VRAM |
| **ReMax** (Li et al.) | Full PPO → simplified REINFORCE with baseline — 50% fewer parameters | Removes value model and GAE. Uses mean reward as baseline instead |
| **PPO-Max** (Wu et al.) | Standard PPO with default coefficients → tuned coefficients for LLM alignment | Better KL/entropy/clip ratio settings specifically for language model training |
| **Adaptive KL** | Fixed kl_coeff=0.1 → dynamic KL coefficient that adjusts to target KL | Auto-tunes the KL penalty. If model strays too far, penalty increases |

**Speculative Decoding (current: fixed K tokens, rejection sampling)**

| Research | What It Upgrades | Upgrade Path |
|----------|-----------------|-------------|
| **Adaptive Speculation** | Fixed num_speculative_tokens=4 → dynamically adjust K based on acceptance rate | High acceptance → speculate more tokens. Low acceptance → fewer. Self-tuning |
| **SpecInfer** (Miao et al.) | Single draft sequence → tree of draft sequences verified in parallel | Multiple speculation paths explored simultaneously. Higher acceptance rate |
| **Online Speculative Decoding** | Frozen draft model → draft model continually updated from main model's distribution | Draft model stays aligned with main model as it trains/adapts |
| **Staged Speculation** | All-or-nothing verification → accept partial prefix of draft tokens | Current code does this but could extend to multi-stage acceptance with resampling |

**KV Cache (current: symmetric INT8 per-token, 2x compression)**

| Research | What It Upgrades | Upgrade Path |
|----------|-----------------|-------------|
| **Per-Channel Quantization** | Per-token scaling → per-channel scaling with separate K/V scale factors | Better accuracy at same compression. Channel outliers handled independently |
| **Mixed-Precision KV** | Uniform INT8 all heads → INT4 for unimportant heads, INT8 for critical heads | Attention head importance varies — quantize less-important heads more aggressively |
| **KV Cache Eviction** (H2O, Heavy Hitter Oracle) | Keep all tokens → evict low-attention tokens, keep heavy hitters + recent window | 5x+ cache reduction. Attention-sink tokens + recent window ≈ full performance |
| **Dynamic Quantization** | Fixed quantization at cache write → re-quantize periodically as attention patterns shift | Older tokens' optimal quantization changes as context grows |
| **Asymmetric Quantization** | Symmetric (-127 to 127) → asymmetric with zero-point offset | Better range utilization when activations aren't centered around zero |

**Repetition Penalty (current: multiplicative, 128-token window)**

| Research | What It Upgrades | Upgrade Path |
|----------|-----------------|-------------|
| **Frequency + Presence Penalty** (OpenAI-style) | Single penalty → separate frequency penalty (repeated tokens) + presence penalty (any seen token) | More nuanced control. Frequency penalizes repetition, presence encourages diversity |
| **Context-Aware Penalty** | Fixed penalty for all tokens → lower penalty for tokens that SHOULD repeat (names, technical terms) | Avoids penalizing legitimate repetition of proper nouns, code keywords |
| **Adaptive Window** | Fixed 128-token window → window that grows/shrinks based on generation length | Short responses need small window. Long responses need larger lookback |

**RAG (current: BM25 TF-IDF, k1=1.5, b=0.75, top-5)**

| Research | What It Upgrades | Upgrade Path |
|----------|-----------------|-------------|
| **BM25 → BM25+** | Standard BM25 → BM25+ with lower-bounding term frequency normalization | Prevents long documents from being unfairly penalized. Drop-in formula change |
| **Hybrid Retrieval** (sparse + dense) | BM25 only → BM25 + embedding similarity, combined score | Best of both: BM25 catches exact matches, embeddings catch semantic matches |
| **Reciprocal Rank Fusion** (RRF) | Single ranking → merge multiple ranking signals with 1/(k+rank) weighting | Combine BM25 + embedding + recency rankings into unified score |
| **Adaptive Chunking** | Fixed 512-char chunks → semantic boundary chunking (paragraph/section-aware) | Chunks that respect document structure retrieve better than arbitrary splits |
| **Query Expansion** | Raw query → expand query with related terms before retrieval | Catches synonyms and related concepts that exact BM25 matching misses |

**Sentiment (current: heuristic word lists, 5 dimensions)**

| Research | What It Upgrades | Upgrade Path |
|----------|-----------------|-------------|
| **VADER Upgrade** | ~80 word lists → VADER-style valence-aware dictionary with intensifiers/negations | Same heuristic approach but with 7500+ rated words, handles "not good", "very bad" |
| **Emoji/Emoticon Scoring** | Text-only analysis → include emoji sentiment mapping | Users often express emotion through emoji — currently invisible to sentiment |
| **Contextual Negation** | Simple word matching → negation window (3-word scope flips polarity) | "not happy" currently scores positive (matches "happy"). Negation window fixes this |
| **Multi-Language Sentiment** | English word lists only → detect language and use appropriate word lists | Non-English users get zero sentiment detection. At minimum: common languages |

**Progressive Growing (current: Net2Net zero-pad + spread)**

| Research | What It Upgrades | Upgrade Path |
|----------|-----------------|-------------|
| **Bert2BERT** (Rothe et al.) | Zero-init new layers → copy and stack existing layers for new depth | New layers start functional instead of identity — faster convergence after growth |
| **LiGO** (Wang et al.) | Manual layer mapping → learned linear growth operators | Optimal weight mapping learned, not hand-coded. Better quality after expansion |
| **Gradual Unfreezing** | Grow and train everything → grow, freeze old layers, gradually unfreeze | Prevents catastrophic forgetting during growth. Stabilizes early post-growth training |
| **Function-Preserving Transforms** | Approximate identity → exactly function-preserving expansion | Model produces identical outputs immediately after growth. Zero quality regression |

---

## Data Quality & Efficiency (highest leverage for small models)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Textbooks Are All You Need** (Microsoft, Phi-1) | Curated textbook-quality data let 1.3B match GPT-3.5 on coding | Data quality is THE multiplier at our model scale (125M-165M). Focus curation effort on structured, explanatory examples |
| **TinyStories** (Microsoft) | 28M param models produce coherent stories with carefully crafted data | Proves small models CAN be coherent — it's the data, not the size |
| **LIMA: Less Is More** (Meta) | 1,000 curated examples beats 50,000+ RLHF examples | Validates our curated_dataset.jsonl quality-over-quantity approach |
| **Scaling Data-Constrained LMs** (Muennighoff et al.) | Repeating high-quality data up to 4 epochs before returns diminish | Important when we don't have infinite data — know when to stop repeating |
| **DataComp** (LAION) | Systematic dataset curation beats larger uncurated datasets | Filtering > collecting. Apply to training data pipeline |
| **Deduplication Matters** (Lee et al.) | Near-duplicate removal significantly improves downstream quality | Add dedup pass to data pipeline before training |
| **SmolLM** (HuggingFace) | 135M/360M/1.7B models with curated Cosmopedia data | Directly comparable to our model sizes — study their data recipes |

---

## Self-Improvement Training (works without external data)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **SPIN** (Self-Play Fine-Tuning) | Model improves by playing against its own previous version outputs | Extends our existing self-play RL — SPIN is complementary |
| **STaR** (Self-Taught Reasoner) | Generate rationales, filter correct ones, retrain — bootstraps reasoning | Could auto-improve reasoning with our `<think>` tokens |
| **ReST** (Reinforced Self-Training) | Generate many outputs, filter by quality, retrain on the best | Quality gate already exists (heuristic scorer). Formalize as training loop |
| **Self-Instruct** (Stanford Alpaca) | Model generates its own instruction-response training pairs | Bootstrap more training data from the model itself |
| **Evol-Instruct** (WizardLM) | Evolve simple instructions into progressively complex ones | Extend adaptive training — auto-generate harder curriculum stages |
| **Quiet-STaR** | Internal reasoning tokens learned without explicit CoT prompting | Natural extension of our `<think>` token reasoning-weighted loss |
| **Constitutional AI** (Anthropic) | Model self-critiques and revises outputs using principles | Self-alignment without human preference labels |

---

## DPO Variants (direct upgrades to existing DPO)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **SimPO** (Simple Preference Optimization) | Reference-model-free DPO — no need to keep a frozen copy | Halves VRAM during DPO training. Drop-in replacement |
| **KTO** (Kahneman-Tversky Optimization) | Works with unpaired feedback (thumbs up/down, no pairs needed) | Easier data collection — don't need matched chosen/rejected pairs |
| **ORPO** (Odds Ratio Preference Optimization) | Combines SFT and alignment in one training step | Simplifies the two-stage SFT-then-DPO pipeline |
| **Iterative/Online DPO** | Generate fresh responses during training for preference data | Prevents training on stale examples. Self-refreshing DPO |

---

## Small Model Architecture (when the model grows)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **MobileLLM** (Meta) | Deep-thin architectures + embedding sharing beat wide-shallow at sub-1B | Architecture guidance for scaling 125M → 300M → 600M |
| **Multi-Head Latent Attention** (DeepSeek-V2) | Low-rank KV projection compresses KV cache 10x | Massive memory savings as context length grows |
| **DeepSeek-V3** (DeepSeek-AI, 2024) | MoE + MLA + auxiliary-loss-free load balancing + multi-token prediction | MoE training recipe, multi-token prediction objective. Reference for when we scale MoE |
| **Qwen2.5** (Qwen team, 2024) | 18T tokens pre-training, multistage RL post-training, 0.5B-72B model range | Training recipes applicable to small models. Qwen2.5-0.5B/1.5B directly comparable to our scale |
| **Mixture of Depths** (Raposo et al.) | Skip transformer layers for easy tokens, use all for hard ones | Dynamic compute allocation — faster inference without quality loss |
| **Layer-wise Learning Rate Decay** | Lower LR for bottom layers (stable features), higher for top (task-specific) | Better fine-tuning, especially for larger models |
| **Mamba / SSM Hybrid** (Gu & Dao) | State space models process sequences in O(n) instead of O(n^2) | Hybrid Mamba-Transformer layers for long-context efficiency without full quadratic attention |
| **RWKV** (Peng et al.) | RNN-Transformer hybrid — linear complexity with competitive quality | Alternative architecture path if context windows need to grow beyond 8K-16K |
| **GQA → MQA → MLA progression** | Progressive reduction: GQA (shared KV groups) → MQA (1 KV head) → MLA (learned compression) | We have GQA — MLA is the next step for 10x KV cache reduction |

---

## Inference Speed (desktop UX)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Medusa** (Multi-head speculation) | Speculative decoding with extra prediction heads, no draft model needed | We have speculative decoding — Medusa eliminates the 2nd model requirement |
| **EAGLE** (Extrapolation Algorithm) | Feature-level speculation beats token-level (better acceptance rate) | Higher quality speculative decoding than standard approach |
| **StreamingLLM** (MIT/Meta) | Attention sinks + sliding window = infinite context with fixed KV cache | Chat sessions can run forever without OOM. Uses our existing sliding window |
| **Token Merging (ToMe)** | Merge redundant tokens mid-forward-pass for 2x speedup | Works for both text and vision encoder passes |
| **TurboQuant** (Google) | Mixed INT4/INT8 KV cache — 6x memory, 8x speed, zero accuracy loss | Extends our INT8 cache to per-head adaptive quantization |
| **PagedAttention** (vLLM) | OS-style paging for KV cache — near-zero memory waste | Config flag exists (`paged=True`) but not implemented yet |
| **Lookahead Decoding** (Jacobi iteration) | Parallel multi-token generation without draft model | Alternative to Medusa — pure algorithmic speedup |
| **SqueezeLLM** (Berkeley) | Non-uniform quantization preserving outlier weights | Better quality than uniform INT8 for weight quantization |
| **Prompt/Prefix Caching** | Cache KV states for system prompt — reuse across turns | System prompt rarely changes between turns. Skip re-encoding it every time |
| **Typical Sampling** (Meister et al.) | Sample from tokens near the expected information content | More natural text than top-p — reduces both boring and random outputs |
| **Mirostat** (Basu et al.) | Adaptive sampling that targets a specific perplexity | Self-adjusting temperature — consistent output quality regardless of prompt |

---

## Distillation (leveraging external models)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Distilling Step-by-Step** (Google) | Extract reasoning traces from large models to train small ones | If using external teacher, distill the WHY not just the answer |
| **Orca** (Microsoft) | Small models learn better from GPT-4 explanations than bare QA pairs | Explanation-augmented training data >> plain instruction data |
| **Zephyr** (HuggingFace) | DPO + distilled SFT achieves strong results with minimal data | Combine our DPO with distillation for maximum quality |
| **Knowledge Distillation** (Hinton et al.) | Soft labels from teacher model carry more information than hard labels | Teacher probability distributions > one-hot labels for training |
| **Gemma 2** (Google) | Online distillation during pre-training + per-layer distillation | Distillation at every layer, not just the output |

---

## Continual Learning (ongoing training without forgetting)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Elastic Weight Consolidation** (EWC) | Regularize important weights so new training doesn't overwrite old skills | Prevents catastrophic forgetting when adding new training data |
| **Replay Buffers** | Mix old training examples with new during fine-tuning | We have `general_data_ratio` — this formalizes and improves it |
| **Progressive Neural Networks** (DeepMind) | Grow capacity by adding columns, freezing old | Theoretical backing for our progressive growing implementation |
| **LoRA Merging / Model Soups** | Average multiple LoRA adapters or checkpoints for combined skills | Train separate LoRAs for different skills, merge into one model |

---

## Context & Memory (longer/better conversations)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Dense Retrieval** (DPR/Contriever) | Embedding-based retrieval >> TF-IDF for semantic matching | Upgrade RAG from TF-IDF to learned embeddings when model is large enough |
| **LongMem / MemoryBank** | External long-term memory with retrieval augmentation | Extends our memory.py fact system with semantic retrieval |
| **Landmark Attention** | Selected tokens serve as long-range attention landmarks | Efficient long-context without full quadratic attention |

---

## Multimodal (vision + audio integration)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **LLaVA** (Visual Instruction Tuning) | Simple linear projection from vision encoder to LLM works well | We have vision encoder — LLaVA's training recipe is the playbook |
| **Flamingo** (DeepMind) | Cross-attention layers for vision tokens instead of projection | Alternative to projecting vision → text space. Better for complex images |
| **Whisper** (OpenAI) | Weak supervision at scale for audio | Our audio encoder uses similar architecture — study their data approach |
| **SigLIP** (Zhai et al., ICCV 2023) | Sigmoid loss for vision-language pre-training — no global softmax needed, scales better | Better vision encoder backbone than CLIP. Can use with our timm pretrained mode |
| **DINOv2** (Oquab et al., Meta 2023) | Self-supervised vision features without text supervision. Distilled small models available | Strong general-purpose vision features. DINOv2-Small (21M params) fits our scale |
| **TinyViT** (Wu et al., ECCV 2022) | Small efficient ViTs via pretraining distillation. 21M params = 84.8% ImageNet | Directly applicable to our desktop vision encoder — small, fast, accurate |

---

## Evaluation & Benchmarking (measuring improvement)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **LM Eval Harness** (EleutherAI) | Standardized benchmark suite for language models | Systematic way to measure model quality over training |
| **AlpacaEval / MT-Bench** | Chat-specific evaluation with automated judging | Measure chat quality without manual evaluation |
| **Perplexity as Proxy** | Lower perplexity correlates with better generation quality | We have perplexity calc — track it consistently across training runs |
| **BLEU/ROUGE** | Standard NLG metrics for generation quality | Quick automated quality check for generated text |
| **Custom Coherence Scoring** | Heuristic coherence metrics (repetition rate, topic drift, response relevance) | We already have a heuristic scorer — extend it into a proper benchmark |

---

## Structured Output & Constrained Generation

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Grammar-Guided Decoding** (Outlines/LMQL) | Constrain token sampling to match a grammar/schema at decode time | 10-15% of tool calls produce malformed output on novel schemas — this eliminates that |
| **JSON Mode / Schema Enforcement** | Force model output to match a JSON schema during generation | Reliable structured output for command parsing, mod tool calls, API responses |
| **Finite State Machine Decoding** | Map regex/grammar to FSM states, mask invalid tokens per step | Lightweight version of grammar-guided — simpler to implement |
| **Function Calling Protocol** (OpenAI-style) | Standardized tool-call format with schema validation | Our command system works but lacks runtime schema validation for mod tools |

---

## Quantization Methods (making models fit in less memory)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **GPTQ** (Frantar et al., ICLR 2023) | One-shot weight quantization via approximate second-order info. 3-4 bits with negligible loss | Reference for understanding our GGUF quantization quality. GPTQ is what most GGUF Q4 methods build on |
| **AWQ** (Lin et al., MLSys 2024 Best Paper) | Protect 1% salient weights by scaling channels. Hardware-friendly, no backprop needed | We already load AWQ models — this is the paper behind it. 3x speedup over FP16 |
| **SmoothQuant** (Xiao et al., ICML 2023) | Migrate quantization difficulty from activations to weights. W8A8 for all matmuls | Relevant for our INT8 KV cache approach. Same authors as AWQ, complementary technique |

---

## Model Compression & Pruning (smaller models, same quality)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Magnitude Pruning** (Han et al.) | Remove weights below threshold — 2-10x compression with minimal quality loss | Compress trained models for faster desktop inference |
| **Movement Pruning** (Sanh et al.) | Prune based on weight movement during fine-tuning, not just magnitude | Better quality than magnitude pruning for fine-tuned models |
| **Wanda** (Sun et al.) | Prune weights by magnitude * input activation — no retraining needed | One-shot pruning — compress a trained model instantly |
| **SliceGPT** (Ashkboos et al.) | Remove entire rows/columns from weight matrices via PCA | Structured pruning — actual speedup, not just smaller file |
| **Sheared LLaMA** | Structured pruning + continued pre-training to downsample a larger model | Prune a 600M model to 300M while keeping 90%+ quality |
| **TIES Merging** (Yadav et al.) | Trim redundant deltas, elect most important signs, merge | Combine multiple LoRA adapters cleanly — better than simple addition |
| **DARE** (Yu et al.) | Drop most delta weights randomly, rescale survivors | Complementary to TIES — sparser, faster merged models |
| **SLERP** (Spherical Linear Interpolation) | Interpolate between model weights on a hypersphere | Smoother model blending than linear average — better combined quality |
| **Task Arithmetic** | weight = base + k * (expert - base), adjustable k per skill | Fine-grained control over how much of each LoRA skill to blend |

---

## Training Optimization (squeeze more from existing setup)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Lion Optimizer** (Google Brain) | Sign-based optimizer — 2x lower memory than AdamW, similar quality | Drop-in AdamW replacement. Saves VRAM on larger models |
| **Sophia** (Stanford) | Second-order optimizer — 2x faster convergence than AdamW | Fewer training steps for same quality. More expensive per step |
| **Schedule-Free Optimizers** (Meta) | No learning rate scheduler needed — optimizer self-adjusts | Eliminates warmup/cosine tuning. One less hyperparameter |
| **8-bit Optimizers** (bitsandbytes) | Quantize optimizer states to INT8 — 75% memory savings | Train larger models in same VRAM. We already have optional bitsandbytes |
| **Warmup-Stable-Decay** (WSD schedule) | Flat LR in the middle phase between warmup and decay | More stable training — less sensitive to total step count |
| **R-Drop** (Liang et al.) | Regularize by minimizing KL between two dropout passes | Free quality boost — same data, same compute, better generalization |
| **BPE-Dropout** (Provilkov et al.) | Randomly skip BPE merges during training — subword regularization | Model learns multiple tokenizations — more robust to typos and rare words |

---

## Data Processing & Augmentation

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Back-Translation** | Translate text to another language and back for paraphrasing | Cheap data augmentation — multiply training data without quality loss |
| **Token Dropout** (Hou et al.) | Randomly drop input tokens during training | Implicit data augmentation — model learns to handle incomplete inputs |
| **Span Corruption** (T5) | Mask contiguous spans and predict them | Pre-training objective that teaches text understanding |
| **UL2** (Google) | Mix multiple pre-training objectives (causal, prefix, span) | One model trained for multiple capabilities simultaneously |
| **FineWeb** (HuggingFace) | Massive curated web crawl with quality filtering pipeline | Study their filtering pipeline — applicable to our data collection |
| **Dolma** (AI2) | Open dataset with reproducible curation documentation | Transparent data recipes — study what to keep vs. filter |

---

## Reasoning & Planning (make model smarter)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Process Reward Models** (PRM) | Reward each reasoning STEP, not just the final answer | Better feedback signal than outcome-only rewards for our RL training |
| **Tree-of-Thought** (Yao et al.) | Branch into multiple reasoning paths, evaluate, select best | Explore multiple solutions before committing to one |
| **Self-Consistency** (Wang et al.) | Sample multiple chain-of-thought paths, majority-vote the answer | More reliable reasoning — errors in one path get outvoted |
| **Skeleton-of-Thought** (Ning et al.) | Generate outline first, then fill in parallel | Faster generation for long responses — outline + parallel expansion |
| **MCTS for LLMs** (Feng et al.) | Monte Carlo tree search applied to token-level planning | Game-tree planning for complex reasoning tasks |
| **Iterative Refinement** | Generate → score → regenerate with feedback | Self-improving output quality without human intervention |
| **Verification / Self-Check** | Model re-reads its own output and checks for errors | Catch hallucinations and logical errors before showing to user |

---

## Embedding & Retrieval (better RAG & memory)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Contriever** (Meta) | Self-supervised dense retrieval — no labeled data needed | Upgrade our TF-IDF RAG to neural retrieval without labeled pairs |
| **Matryoshka Embeddings** (Kusupati et al.) | Train embeddings that work at any truncated dimension | Flexible embedding size — use 64-dim for fast search, 768-dim for precision |
| **ColBERT** (Khattab & Zaharia) | Late interaction retrieval — per-token similarity matching | Better retrieval quality than single-vector embeddings for our RAG |
| **RAPTOR** (Sarthi et al.) | Recursive tree of summaries over document chunks | Hierarchical RAG — retrieve at multiple granularity levels |
| **Hypothetical Document Embeddings** (HyDE) | Generate a hypothetical answer, embed that for retrieval | Better RAG queries — search by what the answer should look like |

---

## Safety & Reliability (production robustness)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Hallucination Detection** (Manakul et al.) | Sample multiple outputs, check consistency — inconsistency = hallucination | Automated fact-checking at generation time. Flag unreliable responses |
| **Calibration** (Guo et al.) | Model confidence should match actual accuracy | Temperature scaling to make model "know what it doesn't know" |
| **Toxicity Filtering** (Perspective API patterns) | Keyword + context scoring for harmful content detection | Heuristic content filter — no external API needed, similar to our sentiment approach |
| **Refusal Training** | Train model to decline harmful/impossible requests | Safety behavior without heavy RLHF — include refusal examples in training data |

---

## Scaling Laws & Compute-Optimal Training (smart budgeting)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Chinchilla** (Hoffmann et al.) | Optimal ratio: ~20 tokens per parameter. 125M model needs ~2.5B tokens | Tells us exactly how much training data we need. Trains smaller model better instead of bigger model worse |
| **Scaling Laws** (Kaplan et al.) | Loss decreases as power law of compute/data/params — all three must scale together | Predict model quality before training. Don't waste compute on undersized data |
| **Compute-Optimal MoE** (Clark et al.) | MoE models need different scaling ratios than dense — more total params, less active | Guidance for when MoE is wired up — scale experts, not active parameters |
| **Training Compute-Optimal LLMs** (Sardana & Frankle) | Inference cost matters too — slightly overtrained small models beat large undertrained ones | For desktop: train 125M longer than Chinchilla-optimal since inference is free |
| **Emergent Abilities** (Wei et al.) | Some capabilities appear suddenly at specific model scales | Know which features to expect at 125M vs 300M vs 600M vs 1B — don't chase abilities below threshold |

---

## Long-Form Generation (coherent long outputs)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Plan-and-Write** (Yao et al.) | Generate outline first, then expand each section | Structured long-form: outline → section → polish. Better than pure autoregressive |
| **DOC** (Document Outline Conditioning) | Condition generation on a pre-generated document skeleton | Prevents topic drift in long responses. Outline keeps generation on track |
| **Rolling Summary** | Periodically summarize generated text and inject back into context | Maintain coherence beyond context window. What we do for chat history, applied to generation |
| **Semantic Consistency Scoring** | Compare beginning vs. end of generated text for topic/entity drift | Post-generation quality check — flag outputs where the ending contradicts the beginning |
| **Hierarchical Generation** | Generate at different granularities: paragraph-level → sentence-level → token-level | Multi-resolution planning for long documents |

---

## Synthetic Data & Data Flywheel (bootstrapping training data)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Cosmopedia** (HuggingFace) | Generate textbook-quality synthetic data from topic outlines using strong models | Our adaptive_trainer already does teacher-model generation — Cosmopedia's recipe improves it |
| **UltraChat** (Ding et al.) | Multi-turn synthetic conversations between two models | Generate conversation training data without humans. Bootstraps our conversation training |
| **Nemotron** (NVIDIA) | Synthetic data pipeline: generate → filter → reward-rank → train | Systematic quality control for synthetic data. Generate 10x, keep top 10% |
| **WizardLM Data Flywheel** | Iteratively evolve data: simple → complex, then filter by difficulty | Extends our adaptive difficulty levels — auto-generate harder examples each cycle |
| **Self-Rewarding LMs** (Meta) | Model generates its own preference pairs by scoring its outputs | Combines our self-play RL with DPO data generation. Model bootstraps its own alignment data |

---

## Personalization & User Adaptation (desktop-specific)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Personal LoRA** | Train a tiny LoRA adapter per user from conversation history | Per-user personality and preference adaptation. 1-5MB per user |
| **Prompt Tuning** (Lester et al.) | Learn soft prompt embeddings per user — no weight changes | Even lighter than LoRA. Learn a 50-token prefix that captures user style |
| **In-Context Learning Enhancement** | Improve few-shot learning with curated user-specific examples | Better use of existing memory.py facts — inject as few-shot examples not just context |
| **User Preference Modeling** (Li et al.) | Track preference embeddings that evolve over conversations | Go beyond keyword facts to learned preference vectors. "User prefers concise answers" |

---

## Test-Time Compute (making the model think harder)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Scaling Test-Time Compute** (Snell et al., 2024) | Small model + more inference compute can beat 14x larger model. Two mechanisms: search against verifiers + adaptive distribution updates | Foundational paper for test-time scaling. Our `<think>` tokens are exactly this — spend compute on reasoning |
| **s1: Simple Test-Time Scaling** (Muennighoff et al., 2025) | Budget forcing: append "Wait" to force model to double-check answers. 1K curated examples enough | Directly applicable — force generation of more `<think>` tokens before answering. s1-32B beat o1-preview |

---

## Deployment & Compilation (faster on desktop hardware)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **torch.compile Modes** (PyTorch 2.x) | `reduce-overhead` mode uses CUDA graphs automatically — 2-3x faster for fixed shapes | We have torch.compile but may not be using optimal mode for inference |
| **TensorRT-LLM** (NVIDIA) | INT8/INT4 + kernel fusion + paged attention in compiled engine | Pre-compile model for specific hardware — 4-5x faster than PyTorch eager |
| **ExecuTorch** (Meta) | Mobile/edge deployment runtime — ARM optimization | Relevant for our Pi-optimized presets. Deploy to ARM devices efficiently |
| **CUDA Graphs** | Capture and replay GPU kernel sequences — eliminate launch overhead | Biggest win for small models where kernel launch time > compute time |
| **Operator Fusion** | Combine multiple small ops (MatMul + Bias + ReLU) into single kernels | Reduces memory round-trips. Especially impactful for inference |

---

## Multi-Model & Routing (orchestrating multiple models)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **Router LLM** (Ong et al.) | Trained router picks cheap vs expensive model per query | Route easy questions to small local model, hard ones to larger model. Save compute |
| **FrugalGPT** (Chen et al.) | Cascade: try cheapest model first, escalate if confidence low | Our router routes to mods — extend to route to GGUF models by difficulty |
| **Mixture of Agents** (Wang et al.) | Multiple models propose → aggregate best parts of each answer | If running multiple models (local + GGUF), combine their strengths |
| **Speculative Routing** | Draft response, check difficulty, re-route to stronger model if needed | Hybrid of speculative decoding + routing. Small model drafts, big model validates |

---

## New Additions (March 26, 2026)

### Efficient Fine-Tuning Upgrades (LoRA improvements)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **DoRA** (Liu et al., ICML 2024) | Weight-Decomposed LoRA — decomposes into magnitude + direction, uses LoRA for direction only | Closes the gap between LoRA and full fine-tuning. Same inference cost, better quality. Direct upgrade to existing LoRA |
| **GaLore** (Zhao et al.) | Gradient Low-Rank Projection — project gradients to low-rank space during training | Train full-parameter models in LoRA-like VRAM. No adapter needed, full quality |
| **LoRA+** (Hayou et al.) | Different learning rates for A and B matrices in LoRA | 1-2% improvement over standard LoRA for free. Just a config change |

### Small Model Training Recipes (directly applicable to 125M-165M)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **MiniCPM** (Hu et al.) | Training recipes for 1-2B models: WSD schedule + batch size scaling | Concrete training schedule that works for small models |
| **Phi-1.5 / Phi-2** (Microsoft) | Textbook-quality synthetic data + web data filtering recipes | Data curation pipeline directly applicable to our scale |
| **Layer Skip** (Elhoushi et al.) | Self-speculative decoding via early exit — no draft model needed | Uses the model itself as its own draft via early layers. Perfect for single-model desktop deployment |

### RL Training Advances (2025)

| Research | Key Takeaway | How It Helps Us |
|----------|-------------|-----------------|
| **DAPO** (Yu et al., ByteDance) | Decoupled clip + dynamic sampling — open-source RL system for reasoning | Four key techniques for stable large-scale RL. Open-source training code + curated dataset |
### 3D Avatar & Animation (speech-driven face + body motion)

The avatar system lives in `mods/avatar/` as a fully decoupled mod (subprocess + TCP, removable without affecting the engine). The legacy branch (`enigma_engine/avatar/`) had a richer built-in system (~3,500 lines, 20+ files) including `lip_sync.py`, `emotion_sync.py`, `adaptive_animator.py`, `bone_control.py`, `ai_bridge.py`, gesture vocabulary, and 3 renderers. These papers improve the existing implementations when porting them to the mod architecture.

| Research | Key Takeaway | How It Helps Us | Tier |
|----------|-------------|-----------------|------|
| **FaceFormer** (Fan et al., CVPR 2022) | Transformer-based autoregressive speech-to-3D-face animation. Encodes long-term audio context, predicts animated face meshes | Upgrades legacy `lip_sync.py` phoneme→mouth mapping from rule-based to learned audio→bone. Reference architecture for the mod's `avatar.speak` | GROWTH |
| **SadTalker** (Zhang et al., CVPR 2023) | Audio-driven 3D motion coefficients (head pose + expression) via 3DMM. Decouples head motion from lip motion | Improves legacy head motion + expression generation. Avatar has head/neck bones + expression system — apply without full 3DMM | NOW |
| **EMOTE** (Daněček et al., SIGGRAPH Asia 2023) | Emotional speech-driven animation with content-emotion disentanglement. Per-frame lip-reading loss + sequence-level emotion supervision | Upgrades legacy `emotion_sync.py` — disentangles WHAT is said from HOW it's said. Key for wiring engine sentiment → avatar expressions via TCP events | NOW |
| **MoGlow** (Henter et al., SIGGRAPH Asia 2020) | Probabilistic controllable motion synthesis via normalizing flows. Autoregressive with LSTMs, causal for real-time | Upgrades legacy `adaptive_animator.py` idle/gesture animations. Causal = no latency. Works on skeleton data — maps to the 50+ bone system | NOW |
| **MotionDiffuse** (Zhang et al., 2022) | First diffusion-based text-to-motion framework. Body-part level control, arbitrary-length synthesis, fine-grained text instructions | Text instructions → body animation. Body-part control maps to bone groups. Extends legacy gesture vocabulary (wave, nod, shake_head, jump) | GROWTH |
| **MDM** (Tevet et al., ICLR 2023) | Human Motion Diffusion Model — transformer-based, classifier-free, predicts sample not noise. Geometric losses (foot contact). Lightweight training | Lightweight text/action→motion. Geometric losses prevent unrealistic poses. Applicable to bone rotation limits already defined in legacy `bone_control.py` | GROWTH |
| **InsActor** (Luo et al., NeurIPS 2023) | Instruction-driven physics-based character animation via diffusion policies. Text instructions → character motion with physical plausibility | Long-term: natural language avatar control via TCP commands. Text→skeleton motion with physics constraints | REFERENCE |
---

## Paper Links Reference

All arXiv/source links for every paper in this document, organized by section.

### Foundational References
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- LLaMA: https://arxiv.org/abs/2302.13971
- RoFormer / RoPE: https://arxiv.org/abs/2104.09864
- RMSNorm: https://arxiv.org/abs/1910.07467
- GLU Variants / SwiGLU: https://arxiv.org/abs/2002.05202
- AdamW: https://arxiv.org/abs/1711.05101
- RAG: https://arxiv.org/abs/2005.11401

### Attention
- Differential Attention: https://arxiv.org/abs/2410.05258
- Flash Attention 3: https://arxiv.org/abs/2407.08691
- Hyper-Attention: https://arxiv.org/abs/2310.05869
- YOCO (Cross-Layer Attention Sharing): https://arxiv.org/abs/2405.05254

### RoPE
- Llama 3.1 RoPE: https://arxiv.org/abs/2407.21783
- Code Llama (NTK-by-parts): https://arxiv.org/abs/2308.12950
- LongRoPE: https://arxiv.org/abs/2402.13753
- YaRN: https://arxiv.org/abs/2309.00071

### Training Pipeline
- muP: https://arxiv.org/abs/2203.03466
- SWA: https://arxiv.org/abs/1803.05407
- Cosine with Warm Restarts (SGDR): https://arxiv.org/abs/1608.03983
- Cyclical Learning Rates: https://arxiv.org/abs/1506.01186
- Gradient Noise: https://arxiv.org/abs/1511.06807
- AdEMAMix: https://arxiv.org/abs/2409.03137

### Sequence Packing
- Multipack: https://arxiv.org/abs/2107.02027

### PPO/RL
- GRPO (DeepSeek-R1): https://arxiv.org/abs/2501.12948
- ReMax: https://arxiv.org/abs/2310.10505
- InstructGPT (Adaptive KL): https://arxiv.org/abs/2203.02155

### Speculative Decoding
- SpecInfer: https://arxiv.org/abs/2305.09781
- Online Speculative Decoding: https://arxiv.org/abs/2310.07177
- Original Speculative Decoding: https://arxiv.org/abs/2211.17192

### KV Cache
- H2O (Heavy Hitter Oracle): https://arxiv.org/abs/2306.14048

### RAG
- BM25+: Lv & Zhai, SIGIR 2011

### Sentiment
- VADER: Hutto & Gilbert, ICWSM 2014

### Progressive Growing
- Bert2BERT: https://arxiv.org/abs/1907.12461
- LiGO: https://arxiv.org/abs/2303.00980
- ULMFiT (Gradual Unfreezing): https://arxiv.org/abs/1801.06146
- Net2Net: https://arxiv.org/abs/1511.05641

### Data Quality & Efficiency
- Textbooks Are All You Need (Phi-1): https://arxiv.org/abs/2306.11644
- TinyStories: https://arxiv.org/abs/2305.07759
- LIMA: https://arxiv.org/abs/2305.11206
- Scaling Data-Constrained LMs: https://arxiv.org/abs/2305.16264
- DataComp: https://arxiv.org/abs/2304.14108
- Deduplication: https://arxiv.org/abs/2107.06499
- SmolLM: https://huggingface.co/blog/smollm

### Self-Improvement Training
- SPIN: https://arxiv.org/abs/2401.01335
- STaR: https://arxiv.org/abs/2203.14465
- ReST: https://arxiv.org/abs/2308.08998
- Self-Instruct: https://arxiv.org/abs/2212.10560
- Evol-Instruct (WizardLM): https://arxiv.org/abs/2304.12244
- Quiet-STaR: https://arxiv.org/abs/2403.09629
- Constitutional AI: https://arxiv.org/abs/2212.08073

### DPO Variants
- DPO (original): https://arxiv.org/abs/2305.18290
- SimPO: https://arxiv.org/abs/2405.14734
- KTO: https://arxiv.org/abs/2402.01306
- ORPO: https://arxiv.org/abs/2403.07691
- Iterative DPO: https://arxiv.org/abs/2312.11456

### Small Model Architecture
- MobileLLM: https://arxiv.org/abs/2402.14905
- Multi-Head Latent Attention (DeepSeek-V2): https://arxiv.org/abs/2405.04434
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- Qwen2.5: https://arxiv.org/abs/2412.15115
- Mixture of Depths: https://arxiv.org/abs/2404.02258
- Mamba: https://arxiv.org/abs/2312.00752
- RWKV: https://arxiv.org/abs/2305.13048
- GQA: https://arxiv.org/abs/2305.13245

### Inference Speed
- Medusa: https://arxiv.org/abs/2401.10774
- EAGLE: https://arxiv.org/abs/2401.15077
- StreamingLLM: https://arxiv.org/abs/2309.17453
- Token Merging (ToMe): https://arxiv.org/abs/2210.09461
- PagedAttention (vLLM): https://arxiv.org/abs/2309.06180
- Lookahead Decoding: https://arxiv.org/abs/2402.02057
- SqueezeLLM: https://arxiv.org/abs/2306.07629
- Typical Sampling: https://arxiv.org/abs/2202.00666
- Mirostat: https://arxiv.org/abs/2007.14966
- Layer Skip: https://arxiv.org/abs/2404.16710

### Distillation
- Distilling Step-by-Step: https://arxiv.org/abs/2305.02301
- Orca: https://arxiv.org/abs/2306.02707
- Zephyr: https://arxiv.org/abs/2310.16944
- Knowledge Distillation (Hinton): https://arxiv.org/abs/1503.02531
- Gemma 2: https://arxiv.org/abs/2408.00118

### Continual Learning
- EWC: https://arxiv.org/abs/1612.00796
- Progressive Neural Networks: https://arxiv.org/abs/1606.04671
- Model Soups: https://arxiv.org/abs/2203.05482
- LoRA: https://arxiv.org/abs/2106.09685

### Context & Memory
- DPR: https://arxiv.org/abs/2004.04906
- Contriever: https://arxiv.org/abs/2112.09118
- Landmark Attention: https://arxiv.org/abs/2305.16300

### Multimodal
- LLaVA: https://arxiv.org/abs/2304.08485
- Flamingo: https://arxiv.org/abs/2204.14198
- Whisper: https://arxiv.org/abs/2212.04356
- SigLIP: https://arxiv.org/abs/2303.15343
- DINOv2: https://arxiv.org/abs/2304.07193
- TinyViT: https://arxiv.org/abs/2207.10666

### Evaluation
- LM Eval Harness: https://github.com/EleutherAI/lm-evaluation-harness
- MT-Bench: https://arxiv.org/abs/2306.05685

### Structured Output
- Outlines (Grammar-Guided): https://arxiv.org/abs/2307.09702
- LMQL: https://arxiv.org/abs/2212.06094

### Quantization Methods
- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
- SmoothQuant: https://arxiv.org/abs/2211.10438

### Model Compression & Pruning
- Magnitude Pruning: https://arxiv.org/abs/1506.02626
- Movement Pruning: https://arxiv.org/abs/2005.07683
- Wanda: https://arxiv.org/abs/2306.11695
- SliceGPT: https://arxiv.org/abs/2401.15024
- Sheared LLaMA: https://arxiv.org/abs/2310.06694
- TIES: https://arxiv.org/abs/2306.01708
- DARE: https://arxiv.org/abs/2311.03099
- Task Arithmetic: https://arxiv.org/abs/2212.04089

### Training Optimization
- Lion: https://arxiv.org/abs/2302.06675
- Sophia: https://arxiv.org/abs/2305.14342
- Schedule-Free: https://arxiv.org/abs/2405.15682
- 8-bit Optimizers: https://arxiv.org/abs/2110.02861
- R-Drop: https://arxiv.org/abs/2106.14448
- BPE-Dropout: https://arxiv.org/abs/1910.13267

### Data Processing & Augmentation
- T5 (Span Corruption): https://arxiv.org/abs/1910.10683
- UL2: https://arxiv.org/abs/2205.05131
- FineWeb: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- Dolma: https://arxiv.org/abs/2402.00159

### Reasoning & Planning
- Process Reward Models: https://arxiv.org/abs/2305.20050
- Tree-of-Thought: https://arxiv.org/abs/2305.10601
- Self-Consistency: https://arxiv.org/abs/2203.11171
- Skeleton-of-Thought: https://arxiv.org/abs/2307.15337

### Embedding & Retrieval
- Contriever: https://arxiv.org/abs/2112.09118
- Matryoshka Embeddings: https://arxiv.org/abs/2205.13147
- ColBERT: https://arxiv.org/abs/2004.12832
- RAPTOR: https://arxiv.org/abs/2401.18059
- HyDE: https://arxiv.org/abs/2212.10496

### Safety & Reliability
- Hallucination Detection: https://arxiv.org/abs/2303.08896
- Calibration: https://arxiv.org/abs/1706.04599

### Scaling Laws
- Chinchilla: https://arxiv.org/abs/2203.15556
- Scaling Laws (Kaplan): https://arxiv.org/abs/2001.08361
- Emergent Abilities: https://arxiv.org/abs/2206.07682

### Synthetic Data
- UltraChat: https://arxiv.org/abs/2305.14233
- Nemotron: https://arxiv.org/abs/2406.08673
- Self-Rewarding LMs: https://arxiv.org/abs/2401.10020

### Personalization
- Prompt Tuning: https://arxiv.org/abs/2104.08691

### Multi-Model & Routing
- RouteLLM: https://arxiv.org/abs/2406.18665
- FrugalGPT: https://arxiv.org/abs/2305.05176
- Mixture of Agents: https://arxiv.org/abs/2406.04692

### Efficient Fine-Tuning (New)
- DoRA: https://arxiv.org/abs/2402.09353
- GaLore: https://arxiv.org/abs/2403.03507
- LoRA+: https://arxiv.org/abs/2402.12354

### Small Model Recipes (New)
- MiniCPM: https://arxiv.org/abs/2404.06395
- Layer Skip: https://arxiv.org/abs/2404.16710

### Test-Time Compute (New)
- Scaling Test-Time Compute: https://arxiv.org/abs/2408.03314
- s1 Simple Test-Time Scaling: https://arxiv.org/abs/2501.19393

### RL Advances (New)
- DAPO: https://arxiv.org/abs/2503.14476

### 3D Avatar & Animation (New)
- FaceFormer: https://arxiv.org/abs/2112.05329
- SadTalker: https://arxiv.org/abs/2211.12194
- EMOTE: https://arxiv.org/abs/2306.08990
- MoGlow: https://arxiv.org/abs/1905.06598
- MotionDiffuse: https://arxiv.org/abs/2208.15001
- MDM: https://arxiv.org/abs/2209.14916
- InsActor: https://arxiv.org/abs/2312.17135

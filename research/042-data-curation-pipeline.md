# LLM Pretraining Data Curation — Best-Practice Report
## For the wubuwizard 35M "general+math" seed (Cosmopedia v2 + finemath + OpenMathReasoning)

**Date:** 2026-08-03 · **Researched from:** SmolLM2 (arXiv 2502.02737), SmolLM blog, FineWeb (arXiv 2406.17557), DeepSeekMath (arXiv 2402.03300), Cosmopedia (HF blog + dataset), FineMath (HF dataset card), Muennighoff et al. (arXiv 2305.16264), Data Mixing Laws (arXiv 2403.16952), curriculum-learning evidence (arXiv 2506.11300 + ACL-SRW 2024). All URLs at the bottom.

---

## 0. TL;DR (the answers)

1. **Pipeline order (run this):** normalize/flatten → cheap hard filters (length, boilerplate, language, script) → **exact dedup** → **near-dedup (MinHash)** → quality scoring/thresholding (heuristics + optional tiny classifier; use finemath's built-in scores) → **decontamination** (benchmark n-gram hash set) → **mixing ratios + seeded shuffle** → tokenize (re-train tokenizer on the *final mix* first) → pack shards. Optionally a final **annealing stage** (last ~10% of steps) with math upsampled.
2. **Data mix for the 35M seed:** ~**70% Cosmopedia v2 / 22% finemath / 8% OpenMathReasoning** as a starting point (math ≈ 30%). Run an ablation ladder (math at 10/20/30%) and pick by GSM8K+MATH+MMLU on a small eval set. Cap OpenMathReasoning at ≤10% (chat-format; flatten it). If finemath rows carry `int_score`, keep ≥3 (prefer ≥4 if volume allows — FineMath-4+ ≈ 2× better on GSM8K/MATH than 3+).
3. **Vocab 16384 byte-level BPE: keep it — but tie embeddings (or drop to 8k).** 16k is inside the sane range (SmolLM 49k on a 135M model, Llama 32k, GPT-2 50k, DeepSeek 100k byte-level BPE). The real risks: (a) untied embeddings+lm_head at hidden=512 eat ~48% of a 35M model's params (tie them → 24%); (b) your tokenizer was trained on only ~135M tokens of *synthetic textbooks only* — retrain it on the final mixed corpus (math streams included) or it will tokenize LaTeX/math badly.
4. **Epochs:** 3–5 epochs over the deduplicated ~135M-token corpus (Chinchilla ≈ 20 tokens/param ≈ 700M tokens ≈ 5.2 epochs; Muennighoff: ≤4 epochs of repetition is ~free, ~16-epoch half-life; SmolLM2 caps any dataset at 4–5 epochs; Infi-WebMath plateaued at ~10 epochs). Reshuffle every epoch. Never exceed ~10 epochs on any single source.
5. **C11-implementable filters:** exact doc/line dedup (FNV-1a/xxHash 64-bit hash sets), MinHash LSH near-dedup (5-gram shingles, 112 hashes, 14 bands), FineWeb's 3 proven repetition heuristics (exact thresholds below), length/symbol/boilerplate rules, script/language heuristics, LaTeX-density math filters, benchmark decontamination via n-gram hash set, and a **pure-C11 linear bag-of-words quality classifier** (hashing trick; weights trained offline in Python, exported as a C header) — this replicates DeepSeekMath's fastText-style filtering with zero ML runtime.

---

## 1. What the literature actually says (evidence anchors)

### 1.1 SmolLM2 — the closest template for you (small model, data-centric) [1]
- 1.7B model trained on **11T tokens ≈ 2 epochs** over its collected datasets, multi-stage. Design principles: (1) adapt mixtures based on live eval; (2) **upsample high-quality math/code in the annealing (final LR-decay) phase**; (3) introduce medium-sized specialized datasets mid-training so big corpora don't drown them; (4) **stay near the 4–5 epoch repetition threshold** (Muennighoff) per dataset.
- Stage mixtures: S1 (0–6T): 60% FineWeb-Edu / 40% DCLM web, 10% code, 0% math → S2: 75% web, 20% code, 5% math → S3: math ≈10%, web ratio flipped → S4 (decay): **58% web, 24% code, 14% math, 4% Cosmopedia v2**.
- Their tokenizer: **vocab 49,152**, trained on a *mixed* corpus — 70% FineWeb-Edu, 15% Cosmopedia-v2, 8% OpenWebMath, 5% StarCoderData, 2% StackOverflow. Tokenizer trained on the mix, not on one source.
- FineMath ablations: FineMath-4+ ≈ 2× GSM8K and ≈ 6× MATH vs InfiMM-WebMath; **Infi-WebMath-4+ plateaued after ~10 epochs of repetition**; OpenWebMath 5 epochs didn't help (small low-quality corpus).
- Math data in stage 1 was *excluded* because the math corpora were too small to survive ~2 epochs over 11T — your situation (135M tokens) is exactly the regime where **math must be upsampled or staged**, not diluted.

### 1.2 FineWeb — the canonical filter+dedup recipe [2]
- Final pipeline order (this is the empirical consensus): **WARC text extraction → base filtering → MinHash dedup → C4-style filters → custom heuristic filters**.
- Base filtering: URL blocklist (adult), fastText language ID (English score ≥ 0.65), MassiveText (Gopher) quality/repetition filters.
- Dedup: **per-crawl (per-snapshot) MinHash, NOT global**. Global dedup over all snapshots removed up to 90% of old crawls and *hurt* quality (the retained 10% was actually worse). MinHash params: 5-gram shingles (English word tokenizer), **112 hash functions, 14 bands × 8 hashes, ≥75% similarity** threshold.
- C4-style filters (they kept all except terminal-punctuation which removed ~30% of tokens — too aggressive): drop lines mentioning javascript / "terms of use" / "cookie policy", drop docs with "lorem ipsum" or a curly bracket `{` (⚠ note: this `{` rule also deletes LaTeX — FineMath explicitly recovered pages rejected by it), word-length filters.
- **3 custom heuristic filters they validated** (remove docs where): fraction of lines ending with punctuation **≤ 0.12** (~10% of tokens), fraction of characters in duplicated lines **≥ 0.1** (MassiveText used 0.2; ~12.5% of tokens), fraction of lines shorter than 30 chars **≥ 0.67** (~3.7%). Together ~22% of tokens removed, +~1% aggregate benchmark.
- Method: collected **50+ candidate statistics**, compared histograms on known high- vs low-quality data, kept the 3 that moved the needle in 28B-token ablations. This "statistics-distribution" method is exactly what you can replicate in C11 (section 5).
- **FineWeb-Edu**: Llama-3-70B-Instruct annotated 460k samples on an additive 0–5 educational-quality scale (prompted to favor grade/middle-school level); a **linear regressor on frozen Snowflake-arctic-embed-m embeddings** (F1 82%, threshold ≥3) scored all 15T tokens → 1.3T-token FineWeb-Edu. Result: MMLU 33→37%, ARC 46→57% vs FineWeb. **Classifier-on-LLM-annotations is the single biggest quality lever** (Llama-3, Phi-3, SmolLM2, FineMath all do a variant).

### 1.3 DeepSeekMath — the math-domain pipeline [3]
- Iterative classifier pipeline on Common Crawl: seed = OpenWebMath (high-quality math); train **fastText** classifier (500k positive math / 500k negative web; dim 256, lr 0.1, word n-gram ≤3); **URL dedup + near-dedup of CC first** (→ 40B HTML pages); recall math pages, **rank by classifier score, keep only top-N tokens** (top 40B → 80B → 120B over 4 iterations); domain analysis (domains with >10% collected pages are math-related; manually annotate their URLs) to enrich the seed and retrain the classifier.
- **Decontamination:** drop any text segment containing an exact **10-gram** match with GSM8K/MATH/CMATH/AGIEval; ≥3-gram exact match for short benchmark texts.
- **Training mix (7B, continued from a code model, 500B tokens): 56% DeepSeekMath Corpus + 4% AlgebraicStack + 10% arXiv + 20% GitHub code + 10% general NL (EN+ZH).** Key findings: (a) math pretraining improves MMLU/BBH too — math transfers to general reasoning; (b) **arXiv text adds no math value**; (c) corpus quality beats size (their 120B corpus beat Proof-Pile-2 at every token budget on a 1.3B model).
- Tokenizer: byte-level BPE, **vocab 100K** (DeepSeek LLM's).
- Their ablations used 150B math tokens on a 1.3B model — i.e., a *big* math share works for math ability, but they were continue-pretraining a general model, not doing a general+math seed from scratch.

### 1.4 Repetition / epochs [4]
- Muennighoff et al. (400+ models, 10M–9B params, up to 1500 epochs): **up to 4 epochs of repeated data ≈ free** (≤0.5% higher val loss than unique data at 4 epochs); meaningful value from repetition up to ~**16 epochs**; after that, each repeated token retains only ~63% of fresh-token value.
- Chinchilla [5]: compute-optimal ≈ 20 tokens/param → 35M ⇒ ~700M tokens ⇒ **~5.2 epochs** over your 135M-token corpus.

### 1.5 Data mixing [6][7]
- Data Mixing Laws (Ye et al. 2403.16952): model performance is a predictable function of mixture proportions; the right method is **small proxy trainings over mixture grids, then extrapolate** (this is how SmolLM2 picked 60/40 FW-Edu/DCLM). RegMix (2407.01492) automates the grid search. Takeaway: don't guess — run 2–3 cheap ablations at 10/20/30% math and pick.
- General caution from the same literature: over-weighting one domain degrades others; for a *general* seed keep math ≤ ~30%.

### 1.6 Curriculum ordering [8][9]
- **No strong evidence for fine-grained difficulty ordering in LLM pretraining at scale.** Recent systematic study (2506.11300): curriculum helps *early/mid* training phases and as a "curriculum warmup" (~3.5% lasting gain) with difficulty metrics like compression ratio, MTLD, Flesch Reading Ease — but it's a second-order effect. For small code LMs, one ACL-SRW study found **no benefit** [9].
- What the big labs actually do (SmolLM2, Qwen, Llama-3, Phi): **stage-wise mixing + annealing**, i.e., (1) stable phase on a general mix, (2) introduce/upsample math & code mid-training once the model has general fluency, (3) **final LR-decay (WSD) phase on the highest-quality data** — "How Learning Rate Decay Wastes Your Best Data" [10] makes the same point: put your best data in the decay window. That's the only "curriculum" worth implementing at 35M.

---

## 2. Recommended pipeline for YOUR corpus (stage order)

Your three sources are all *already curated* (unlike raw Common Crawl), so stages 1–3 are cheap, and stages 4–5 carry most of the value. Order below follows FineWeb/DeepSeekMath consensus (hard filter → dedup → soft quality → mix).

| # | Stage | Action (per source) | Evidence |
|---|---|---|---|
| 0 | **Normalize/flatten** | Cosmopedia v2: keep `text` column. OpenMathReasoning: flatten `messages` (user/assistant) to `Problem:\n…\nSolution:\n…` plain text. finemath: keep `text`; **if `score`, `int_score`, `language`, `language_score`, `token_count` metadata exists, apply thresholds here for free**: `int_score>=3`, `language=='en'`, `language_score>=0.65`, `token_count` within [64, 20000]. | FineMath card [12] |
| 1 | **Cheap hard filters** | Length: drop <200 chars / <3 sentences; cap/split >100k chars. Boilerplate keyword lists ("cookie", "click here", "subscribe", "lorem ipsum", "javascript"). Script check: %ASCII, %CJK, allowed-unicode-block test. Drop docs with `{`-only/LaTeX-free junk on the math streams is fine — but DON'T apply the C4 curly-bracket rule to math data (kills LaTeX). | C4 [2], FineWeb §3.5 [2], RefinedWeb [11] |
| 2 | **Exact dedup** | Doc-level 64-bit hash set (FNV-1a/xxHash over whitespace-normalized text); line-level exact dedup for boilerplate (drop lines seen >N times across corpus). Removes cross-source copies (Cosmopedia seeds came from FineWeb; finemath is CC-derived; OpenMathReasoning is AoPS/MSE/MO — heavy overlap). | C4 [2], DeepSeekMath URL dedup [3] |
| 3 | **Near-dedup (MinHash LSH)** | 5-gram word shingles, 112 hashes, 14 bands × 8, ≥75% similarity, keep first doc per cluster. Fully C11-doable (section 5). | FineWeb §3.4 [2], FineMath [12] |
| 4 | **Quality scoring** | (a) Run the 3 FineWeb repetition heuristics (§1.2) + math-density filters (LaTeX `$`, `$$`, `\begin`, `\frac`, `\sum`, digit/operator density) on the math streams. (b) Optional but recommended: **linear bag-of-words quality classifier in pure C11** (trained offline; see §5.6) to rank-and-keep top ~60–80%, DeepSeekMath-style. | DeepSeekMath §2.1 [3], FineWeb §3.6 [2] |
| 5 | **Decontamination** | Build a hash set of all **10-gram** (or 13-gram) substrings of GSM8K, MATH, MMLU, ARC train+test; drop any doc containing a match. Benchmarks are small — the hash set is a few MB. | DeepSeekMath [3], FineMath [12], SmolLM2 [1] |
| 6 | **Mix + shuffle** | Sample per-source with target ratios (§3), deterministic seeded shuffle, write shards. | §1.5 |
| 7 | **Tokenize** | Retrain BPE on the *final mixed corpus* (or reuse existing 16k vocab — see §4). Re-tokenize everything (fast in C11). | SmolLM2 §4.1 [1] |
| 8 | **Pack + anneal** | Pack docs into sequences (EOS-separated, ~2048 tokens). Training schedule: stable phase on the §3 mix; **last ~10% of steps = LR-decay/annealing phase with math upsampled (OpenMathReasoning + finemath-4+ if available)**. | SmolLM2 §4.5 [1], WSD [10] |

**Order nuance:** FineWeb did base-filter → dedup → fine-filters (filtering shrinks dedup cost; dedup before the aggressive C4 filters). DeepSeekMath deduped URLs *first*. Both are defensible; since your sources are small, do cheap filters → exact dedup → MinHash → quality → decontam.

---

## 3. Ideal data mix for a 35M general+math seed

Anchors from the literature:
- SmolLM2 final stage: 14% math / 24% code / 58% web / 4% Cosmopedia [1].
- DeepSeekMath 7B continue-pretrain: ~60% math-family / 20% code / 10% NL [3]; math-only 1.3B ablations worked for math ability but that's a domain-specialized regime.
- Data Mixing Laws: don't guess — ablate [6]. FineMath card: 50/50 mixing of two math corpora matches the better one alone [12] → mixing math sources is safe.

**Recommended starting mix** (tokens, after dedup+filter):

| Source | Share | Rationale |
|---|---|---|
| Cosmopedia v2 (general, synthetic textbooks/stories) | **70%** | Keeps it a *general* seed; synthetic textbook data is high-quality & self-contained (Phi/Cosmopedia lineage [13][14]); you already have 135M tokens tokenized |
| finemath (math web, classifier-curated CC) | **22%** | The backbone math corpus; if it's HuggingFaceTB/finemath, prefer 4+ over 3+ rows if volume permits (2× GSM8K/MATH [1]) |
| OpenMathReasoning (540K AoPS/SE/MO Q&A problems) | **8%** | Step-by-step reasoning exemplars (the exact "step-by-step reasoning" gap FineMath targets); chat-format so cap it, and it's ideal for the final annealing window |

- **Ablation ladder (cheap, mandatory):** train three 35M seeds at math = 10% / 20% / 30% for a few hundred steps-equivalent and pick by GSM8K + MATH-4 + MMLU-subset + HellaSwag (use FineWeb's 1000-sample benchmark truncation trick [2] for fast eval). Expect ~25–30% math to win on math while keeping MMLU within noise of the 10% run — past that, general benchmarks start dropping (mixing-laws caution [6]).
- **If you later add code/web data:** rebalance toward SmolLM2's final ratios (web-dominant) — your seed's job is just to be a good base for that.
- **Contamination note:** finemath ships a decontamination report vs GSM8K/MATH/MMLU/ARC [12]; still run stage 5 yourself — OpenMathReasoning is built from public problem sites and WILL contain benchmark-adjacent problems.

---

## 4. Vocab size verdict (16384 byte-level BPE)

**Verdict: 16,384 byte-level BPE is a fine choice for a 35M model — keep it, with two conditions.**

- Byte-level BPE is right (no OOV, matches DeepSeek practice [3]). GPT-2 50k, Llama 32k, SmolLM 49k, DeepSeek 100k — 16k is at the low end of the observed range, which is *appropriate* for a 35M model (domain-limited corpus, English+math).
- **Condition 1 — embedding budget.** Untied embeddings+lm_head at hidden=512: 2×16384×512 = **16.8M params = ~48% of 35M**. Fixes: tie embeddings to the lm_head (→ 8.4M ≈ 24%), or drop vocab to 8,192 if you must stay untied. (SmolLM-135M runs a 49k vocab with hidden 768 → ~56% embedding overhead and works anyway, so 16k is not extreme — but tying is free and strictly better for a seed.)
- **Condition 2 — tokenizer training data.** A BPE trained on ~135M tokens of Cosmopedia-only (synthetic textbooks) will tokenize finemath/OpenMathReasoning math (LaTeX, Unicode, code-like tokens) inefficiently (longer sequences → more compute, weaker merges). SmolLM2 trained its 49k tokenizer on the *mixed* corpus (70/15/8/5/2 across web/cosmopedia/math/code/SO) [1]. **Action:** retrain the BPE on a sample of the final mix (or at least add 5–10M tokens of finemath + OpenMathReasoning text to the training data), then re-tokenize. If you change vocab size you must re-tokenize the existing 135M tokens — cheap in C11, do it once.
- Bigger vocab ≠ better at this scale; smaller vocab means longer sequences and slower training per step. 8k–16k is the sweet spot for a 35M English+math model; don't go above ~32k unless you add multilingual data.

---

## 5. C11-implementable filters (build these)

All of these are pure C11, single-threaded or trivially parallel, memory-bounded for a corpus of your size (135M tokens ≈ a few hundred MB of docs; finemath+OpenMathReasoning ~26GB raw text — process streamed).

1. **Exact document dedup — hash set.** FNV-1a 64-bit (or xxHash64) over whitespace-normalized text; open-addressing table with 2^24–2^26 slots, store 8-byte hash + 4-byte doc id. O(1) per doc. Also hash *lines* for a line-level pass (drop lines whose hash appears in >N distinct docs — boilerplate).
2. **MinHash LSH near-dedup.** Per doc: tokenize to words, slide a 5-gram window, hash each shingle (FNV-1a), keep the min hash per band → for 14 bands × 8 hashes: 112 shingle hashes per doc, take min of 8 per band → 14 band signatures; bucket docs by (band#, signature) in hash maps; any bucket with >1 doc ⇒ candidate pair (Jaccard ≥ ~75%). At your scale this runs in minutes in C.
3. **FineWeb's 3 validated repetition heuristics** (exact thresholds, C11-trivial line scans):
   - `frac_lines_ending_with_punct < 0.12` ⇒ drop (punct = . ! ? " ' ) ] } »)
   - `frac_chars_in_duplicated_lines > 0.10` ⇒ drop (duplicated = line hash appears ≥2× within the doc)
   - `frac_lines_shorter_than_30_chars > 0.67` ⇒ drop
4. **Length / symbol / boilerplate filters.** min 200 chars & ≥3 sentences; max 100k chars (split); symbol-to-word ratio cap (~C4's); stopword-density floor; blocklists: "lorem ipsum", "click here", "subscribe", "cookie policy", "terms of use", "javascript" (lines containing them are dropped, not whole docs).
5. **Language/script heuristic (no ML).** Count Unicode blocks: %ASCII, %Latin-ext, %CJK; drop docs that are <90% of any single script or mixed-script gibberish; math streams: require %digits+math-operators ≥ small floor so forum chatter and ads get dropped.
6. **Math-quality filter for finemath/OpenMathReasoning.** String scans only: count `$`, `$$`, `\(`, `\begin{…}`, `\frac`, `\sum`, `\int`, `=`, `≤`, `≥` occurrences; compute LaTeX-density = math-delimiter chars / total chars; drop docs below a floor (calibrate on a sample: text you know is math vs forum Q&A). If finemath rows keep `int_score`/`language_score`/`token_count` fields, prefer those (already computed by HF) — zero-cost.
7. **Decontamination.** Precompute: tokenize GSM8K/MATH/MMLU/ARC train+test into **10-gram** sequences (13-gram per FineMath/Qwen2.5-Math), insert 10-gram hashes into a Bloom filter + exact set (few MB). Stream docs; drop any doc containing a hit. (DeepSeekMath used 10-gram exact [3]; FineMath 13-gram [12].)
8. **Quality classifier in pure C11 (the DeepSeekMath trick, no ML runtime).** Offline in Python: label ~1k docs (500 high: finemath-4+ samples / OpenMathReasoning; 500 low: boilerplate, short junk, random web text), train a **linear model on hashing-trick features** (word unigrams + char 4-grams, 2^20 buckets, e.g. a simple log-linear/online SGD or even a fastText-style classifier), export weights as a C header. In C11: `score = Σ w[hash(feature)]` over the doc's features; keep docs with score in the top N or above a threshold. This is literally DeepSeekMath's fastText classifier minus the library [3], and FineWeb-Edu shows classifier filtering is the highest-leverage quality step [2].
9. **Token-sequence dedup (optional, after tokenization).** Hash windows of 128 token ids (rolling hash) to catch documents that differ only in whitespace/case.

---

## 6. Epochs / token-repetition guidance (concrete)

- Target ≈ **3–5 epochs** over the deduplicated corpus for a 35M seed. 135M tokens × 4 ≈ 540M ≈ Chinchilla-ish for 35M [5]. 4 epochs is inside Muennighoff's "free repetition" zone [4].
- **Never go past ~10 epochs on any single source** (Infi-WebMath-4+ plateaued at ~10 [1]); beyond ~16 epochs each repeated token loses ~37% of its value [4].
- **Reshuffle between epochs** (new seeded order each epoch) — this is the standard cheap mitigation for repeated-data overfitting [4].
- If you want more effective tokens without more epochs: upsample math by 1.5–2× instead of adding epochs to the whole corpus (SmolLM2's "upsample in later stages" principle [1]).
- Watch val-loss on a held-out slice (never train on your eval slice); a rising train/val gap with flat val = repetition overfitting → cut epochs.

---

## 7. Sources (URLs)

1. SmolLM2: When Smol Goes Big — https://arxiv.org/abs/2502.02737 (pipeline, mixes, FineMath, tokenizer §4.1, epochs)
2. The FineWeb Datasets — https://arxiv.org/abs/2406.17557 (pipeline §3, dedup §3.4, heuristics §3.6, FineWeb-Edu §4) · blog: https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
3. DeepSeekMath — https://arxiv.org/abs/2402.03300 (data pipeline §2.1, decontamination, tokenizer 100K, mixes) · repo: https://github.com/deepseek-ai/deepseek-math
4. Scaling Data-Constrained Language Models (Muennighoff et al.) — https://arxiv.org/abs/2305.16264 (4-epoch rule, 16-epoch half-life)
5. Training Compute-Optimal LLMs (Chinchilla) — https://arxiv.org/abs/2203.15556
6. Data Mixing Laws (Ye et al.) — https://arxiv.org/abs/2403.16952 · RegMix — https://arxiv.org/abs/2407.01492
7. Scaling Laws for Optimal Data Mixtures — https://proceedings.neurips.cc/paper_files/paper/2025/file/bc1d640f841f752c689aae20b31198c1-Paper-Conference.pdf
8. Beyond Random Sampling: Efficient LM Pretraining via Curriculum Learning — https://arxiv.org/abs/2506.11300
9. Curriculum Learning for Small Code Language Models (ACL-SRW 2024) — https://aclanthology.org/2024.acl-srw.44.pdf
10. How Learning Rate Decay Wastes Your Best Data in Curriculum-Based Pretraining — https://openreview.net/forum?id=T5wkZJqzkz
11. RefinedWeb — https://arxiv.org/abs/2306.01116
12. FineMath dataset card (curation, scores, decontamination, dedup nuance) — https://huggingface.co/datasets/HuggingFaceTB/finemath
13. Cosmopedia blog (v1/v2, synthetic pretraining data) — https://huggingface.co/blog/cosmopedia · Cosmopedia v2 dataset — https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
14. smollm-corpus (cosmopedia-v2 subset, 39.1M rows; fineweb-edu-dedup; python-edu) — https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus · SmolLM blog — https://huggingface.co/blog/smollm
15. OpenMathReasoning (540K unique problems, AoPS/MSE/MO, AIMO-2) — https://huggingface.co/datasets/nvidia/OpenMathReasoning · paper https://arxiv.org/abs/2504.16891
16. datatrove (HuggingFace pipeline library — filter checklist to port) — https://github.com/huggingface/datatrove
17. Deduplicating Training Data Makes Language Models Better (Lee et al.) — https://arxiv.org/abs/2107.06499

---

*Prepared for the wubuwizard 35M seed. Next step after adopting this: implement stages 1–5 as C11 filters (section 5), run the 10/20/30% math ablation ladder, and lock the tokenizer on the final mix.*

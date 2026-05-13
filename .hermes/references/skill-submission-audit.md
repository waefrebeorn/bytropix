# WuBu Text AI — Skill Audit & Submission Assessment

## Assessment Criteria

A skill is suitable for Hermes online skills if:
1. **Self-contained** — no references to `/home/wubu/`, local paths, or user-specific projects
2. **Proper frontmatter** — name, description, version, tags, author (Hermes Agent), license (MIT), platforms
3. **Generalizable** — the technique/approach applies beyond one specific project
4. **Complete** — covers enough to be useful without requiring other skills

---

## Skills Assessment

### 1. wubu-mind-palace (`wubu-mind-palace`)
**Description:** Prestige prompt structure system with multi-stage directions, devils advocate loop, and slate markdown for project navigation.
**Size:** 254 lines, 24KB
**Frontmatter:** ❌ Missing version, tags, platforms, author, license
**Self-contained:** ❌ References to `/home/wubu/`, `~/HASHMIND/`, `~/bytropix/` (22 references)
**Submission verdict:** ❌ **NOT suitable** — deeply coupled to WuBu-specific projects, local paths everywhere

**How to fix for submission:**
- Strip all local paths — replace with `$PROJECT_ROOT` or template variables
- Remove WuBuNesting llama.cpp encoder content (that's a separate skill)
- Keep only: the prestige structure, devil's advocate loop, 4-phase review pattern, slate markdown format
- Add proper frontmatter with tags: `[workflow, project-management, quality-assurance, devils-advocate]`
- This would be a **general "Devil's Advocate Review System"** skill, not a WuBu skill

**Recommendation:** ⭐ **REFINE for submission** — the devil's advocate 4-phase loop, prestige prompt structure, and slate markdown format are genuinely useful patterns. Strip the WuBu specifics, keep the framework.

---

### 2. wubu-text-ai (`wubu-text-ai`)
**Description:** Pure C + CUDA language model project detail skill.
**Size:** 116 lines, 8KB
**Frontmatter:** ❌ Missing version, tags, platforms, author, license
**Self-contained:** ❌ 12 local path references, deeply WuBu-specific
**Submission verdict:** ❌ **NOT suitable** — this is a project notebook, not a generalizable skill

This is a **project detail skill** — meant to be loaded with `skill_view()` during CLI sessions on the bytropix project. It has no general value outside the WuBuText AI project. Keep local-only.

**Recommendation:** ❌ Keep local only. Not submission material.

---

### 3. wubu-cuda-kernels (`wubu-cuda-kernels`)
**Description:** CUDA GPU kernel infrastructure for WuBuText AI.
**Size:** 304 lines, 16KB
**Frontmatter:** ❌ Missing version, tags, platforms, author, license
**Self-contained:** ❌ 33 local path references, project-specific
**Submission verdict:** ❌ **NOT suitable** — too project-specific

However, the **parallel associative scan kernel technique** (ssm_parallel_scan_kernel) IS generalizable. The section on "SSM Recurrence → Linear Operator Form → Associative Scan" and the parallelization strategy for linear recurrences is valuable.

**Recommendation (conditional):** If the SSM parallel scan + linear recurrence section (lines ~200-250) is extracted into a standalone "Parallel Associative Scan for SSM Recurrences" skill, it could be submitted. The rest is WuBu-specific weight loading and project scaffolding.

---

### 4. llama-cpp-integration (`llama-cpp-integration`)
**Description:** Integrate custom GGML operations into llama.cpp — enum registration, CUDA dispatch, KV cache wiring, post-graph encoder patterns.
**Size:** 1,531 lines, 100KB — the largest skill
**Frontmatter:** ❌ Missing version, tags, platforms, author, license
**Self-contained:** ❌ 6 local path references
**Submission verdict:** ❌ **NOT suitable as-is** — too large, too project-specific

**However:** The post-graph encoder pattern, enum registration pattern, and CUDA dispatch pattern ARE generalizable techniques for anyone modifying llama.cpp with custom ops.

**Recommendation (conditional):** Could extract 3 separate smaller skills:
1. `llama-cpp-custom-op-registration` — How to register new GGML ops (enum, type_traits, CUDA dispatch, CPU skip)
2. `llama-cpp-post-graph-pattern` — How to run custom CUDA kernels after GGML graph compute (non-GGML-op pattern, dedicated stream, buffer lifetime)
3. Each would need full rewrite to be generic

---

### 5. optimizer-research-2026 (`optimizer-research-2026`)
**Description:** Findings from May 2026 optimizer papers (Aurora, PolarGrad) relevant to training loops.
**Size:** ~70 lines, 3KB
**Frontmatter:** ❌ Missing version, tags, platforms, author, license
**Self-contained:** ✅ No local path references. General paper summaries.
**Submission verdict:** ⭐ **SUITABLE with refinement**

This is the most submission-ready of the WuBu-related skills. It's:
- General: Aurora and PolarGrad are relevant to any LLM training pipeline
- Self-contained: no local paths
- Well-structured: problem → solution → relevance → implementation notes

**Fixes needed:**
1. Add proper frontmatter (version, tags: `[optimization, training, research]`, MIT license)
2. Remove "Relevance to WuBuText AI" section — replace with general relevance assessment
3. Expand implementation notes to be framework-agnostic
4. Add arXiv links for both papers

**Recommendation:** ⭐ **REFINE for submission** — this is small, clean, and generally useful.

---

### 6. session-goal-paste (`session-goal-paste`)
**Description:** Generate structured GOAL PASTE from last CLI session context for starting a new session with full state handoff.
**Size:** 175 lines, 10KB
**Frontmatter:** ✅ Has version (1.2.0), tags, platforms, author, license
**Self-contained:** ❌ Minor — references `/home/wubu/` in one example path, references wubu-specific example content
**Submission verdict:** ⭐ **SUITABLE with minor fixes**

Best-prepared skill. Frontmatter is correct. The pattern is general: any Hermes user doing multi-session project work needs session handoffs.

**Fixes needed:**
1. Remove `/home/wubu/` path reference in examples (use generic `/path/to/project`)
2. Remove WuBu-specific example content (use generic project names)
3. Add note about the vagua/overnight-map.md pattern
4. Tone down the "user frequently asks" framing — make it more universal

---

## Summary Table

| Skill | Local-Only? | Submission Quality | Effort to Fix |
|-------|-------------|-------------------|---------------|
| session-goal-paste | Minor path refs | ⭐ High (ready) | 5 min |
| optimizer-research-2026 | None | ⭐ High (ready) | 5 min |
| wubu-mind-palace (DA framework) | Heavy | 🟡 Medium (need extraction) | 30 min |
| wubu-cuda-kernels (scan kernel) | Heavy | 🟡 Medium (need extraction) | 20 min |
| llama-cpp-integration (3 subsets) | Medium | 🔴 Low (too coupled) | 60+ min |
| wubu-text-ai | Heavy | ❌ Not suitable | N/A |

## Recommended Submission Candidates

**Now:** `session-goal-paste` (fix paths) + `optimizer-research-2026` (add frontmatter, generalize)
**Next:** Extracted devil's advocate framework from `wubu-mind-palace`
**Maybe:** Extracted parallel associative scan pattern from `wubu-cuda-kernels`
**Never:** `wubu-text-ai` (project notebook), full `llama-cpp-integration` (too coupled)

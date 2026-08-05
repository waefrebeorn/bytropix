# Theme J — Knowledge, Planning & UX Cohesion

---

### J1. Architecture Decision Records (ADRs, Michael Nygard) + Lightweight RFC Process

- **Hop chain**:
  1. **Seed** — Michael Nygard, "Documenting Architecture Decisions" (Cognitect blog, Nov 2011) — the canonical paper that coined ADRs as short text files in `doc/arch/adr-NNN.md` with five sections: Title, Context, Decision, Status, Consequences.
  2. **adr.github.io** — the ADR GitHub organization; ADR templates (MADR, Y-Statement); the append-only decision log; status lifecycle (proposed → accepted → deprecated → superseded).
  3. **ThoughtWorks Technology Radar** — lightweight ADRs in the "Adopt" ring; the key insight that keeping records in source control keeps them in sync with code.
  4. **Lightweight RFC process** — Attentive Engineering's pragmatic RFC template (status + lifecycle: Draft → In Review → Accepted → Superseded); Rust RFC process on internals.rust-lang.org as the canonical large-scale example of lightweight proposal review.
  5. **ADRs for AI agents** — Actual AI's "ADRs for Coding Agents" (2025): agent-optimized ADRs with `applies_to` file globs, token-aware compression, and deduplication across `CLAUDE.md`/`AGENTS.md`/`.cursor/rules`.
  6. **UK Government ADR framework** (Nov 2025) + AWS Prescriptive Guidance — institutional-scale adoption; ADRs as append-only logs superseded rather than edited.
  7. **Convergence** — the ADR format (5 fields, append-only, versioned next to code) is the smallest unit that survives both human turnover and agent context rotation.

- **Convergence**: *One ADR per architecturally significant decision, stored in the repo, append-only, with a status lifecycle — this is the minimum structure that keeps both human engineers and AI agents coherent across a monorepo or multi-repo codebase.*

- **Sources**:
  - https://www.cognitect.com/blog/2011/11/15/documenting-architecture-decisions
  - https://adr.github.io/
  - https://www.actual.ai/blog/agent-optimized-adrs
  - https://rickpollick.com/blog/adr-comeback-anchoring-agentic-engineering-teams
  - https://www.w3.org/community/design-tokens/2025/10/28/design-tokens-specification-reaches-first-stable-version/ (for W3C template ecosystem)
  - https://www.gov.uk/government/publications/architectural-decision-record-framework/architectural-decision-record-framework

- **2 concrete ways this improves wubuwizard / wubuos**:
  1. **wubuwizard**: Create an `docs/adr/` directory with one ADR per major architectural choice (e.g., "ADR-001: C11 as the implementation language", "ADR-002: 9P namespace for model I/O", "ADR-003: GGUF as the weight format"). Each ADR uses the 5-field Nygard template (Context, Decision, Status, Consequences, Alternatives). This gives any agent reading the repo a traceable "why" behind every structural decision — no more guessing from code alone.
  2. **wubuos**: Adopt a lightweight RFC process for OS-level changes (e.g., new syscall personality, theme engine addition). RFCs live in `docs/rfcs/` with a status field and a close-commitment date. When an RFC is accepted, it generates the corresponding ADR. This creates a traceable chain from proposal → decision → implementation → ADR, closing the loop that currently leaves the two repos' decisions undocumented.

---

### J2. Roadmap / Gap Ledger Patterns (Backlog Discipline & Close-Commitment)

- **Hop chain**:
  1. **Seed** — Daniel Abraão, "Backlog Never Shrinks" (abraao.tech, Apr 2026): the input-rate vs. throughput model — if a team delivers 20 items/month but receives 35, the backlog grows by 15 per cycle, and the real problem is systemic, not effort-based.
  2. **WIP limits** (kanbantool.com): limiting work in progress per kanban stage forces teams to finish before starting, reducing lead time by 40–60% within eight weeks.
  3. **Fowler's Technical Debt Quadrant** (paddle.com): four quadrants (deliberate/reckless, prudent/inadvertent) — the framework for categorizing what's "open" in a gap ledger honestly.
  4. **Kubernetes issue triage** (kubernetes.dev): bi-weekly community meetings, Triage Party tool for GitHub issues, SIG-based ownership — how large OSS projects keep 1000+ issues honest.
  5. **CNCF roadmap practices** (contribute.cncf.io): roadmaps as contribution magnets, consistent triage grooming, bus-factor mitigation.
  6. **Linear's triage workflow** (thecommonwealthcreative.com): single-sweep accept/reject/prioritize/defer; issues that don't finish either roll over with explicit acknowledgment or return to backlog — no silent accumulation.
  7. **Convergence** — the honest gap ledger is a WIP-limited, input-rate-aware, close-commitment-bound system where every open item has an owner, a priority, and a deadline; items that don't close within the commitment window are explicitly rolled over or dropped.

- **Convergence**: *An honest gap ledger treats open items as a flow system with input rate and throughput — WIP limits prevent overload, close-commitments enforce accountability, and explicit rollover/drop decisions prevent the backlog treadmill.*

- **Sources**:
  - https://abraao.tech/en/blog/por-que-backlog-nunca-diminui/
  - https://kanbantool.com/kanban-guide/kanban-fundamentals/limit-work-in-progress
  - https://www.paddle.com/resources/technical-debt
  - https://www.kubernetes.dev/docs/guide/issue-triage/
  - https://contribute.cncf.io/projects/best-practices/community/contributor-growth/open-source-roadmaps/
  - https://thecommonwealthcreative.com/the-codex/linear-issue-tracking/

- **2 concrete ways this improves wubuwizard / wubuos**:
  1. **wubuwizard**: Formalize the `research/INDEX.md` gap ledger as a flow system. Each gap gets an owner (person or agent), a priority (P0–P3), and a close-commitment date (M1, M2, etc.). Track input rate (new gaps opened per week) vs. throughput (gaps closed per week) and alert when input exceeds throughput — this is the "backlog treadmill" warning. The existing `open`/`wired` status becomes the minimum viable state; add `owner`, `priority`, `committed-close` fields.
  2. **wubuos**: Apply the Linear triage model to the MASTER-INDEX banks. Every gap in the 1000-gap banks gets an explicit "rollover" or "drop" decision at each review cadence (weekly). Gaps that are dropped must have a one-line reason (e.g., "superseded by ADR-003", "not a driver: no module need"). This prevents the silent accumulation of stale gaps that makes the ledger feel dishonest to contributors.

---

### J3. Design Systems & Cohesive UX (Design Tokens, Theming, Win98/XP Aesthetic)

- **Hop chain**:
  1. **Seed** — Nathan Curtis, "Tokens in Design Systems" (EightShapes, 2017): Salesforce pioneered design tokens in 2014; tokens are the single source of truth for colors, typography, spacing — indivisible design decisions exported to code.
  2. **W3C Design Tokens Community Group** — the DTCG spec reached its first stable version (2025.10) with theming, multi-brand support, modern color spaces (Display P3, Oklch), rich token relationships (inheritance, aliases), and cross-platform code generation (iOS, Android, web, Flutter).
  3. **98.css** (jdan.github.io/98.css): a CSS design system for faithful Windows 98 UI recreation — demonstrates that a single token/style source can reproduce an entire aesthetic across components (window chrome, title bars, status bars, buttons).
  4. **Windows 98 UX as onboarding model** (prototypr.io): Win98's UX was its user onboarding process — command-line users were guided through a graphical desktop with discoverable affordances; the aesthetic consistency (Beige box, grey window chrome, blue links) reduced cognitive load.
  5. **Cross-platform design token bridge** (noeinoi.com, thedesignsystem.guide): one token file generates platform-specific code — the same principle that unifies CLI + GUI + web surfaces.
  6. **WuBuOS theme engine** (existing): the `theme-engine` architecture in WuBuOS already supports runtime-switchable themes — design tokens would give it a single source of truth instead of scattered hardcoded values.
  7. **Convergence** — design tokens are the universal bridge between design intent and implementation; one token file, multiple platform outputs, one source of truth for every visual property.

- **Convergence**: *Design tokens are the single source of truth that unifies visual language across CLI, GUI, and web surfaces — one token file generates platform-specific code, and the token taxonomy (primitive → semantic → component) ensures consistency without duplication.*

- **Sources**:
  - https://medium.com/eightshapes-llc/tokens-in-design-systems-25dd82d58421
  - https://www.w3.org/community/design-tokens/2025/10/28/design-tokens-specification-reaches-first-stable-version/
  - https://jdan.github.io/98.css/
  - https://blog.prototypr.io/why-windows-98s-user-onboarding-is-better-than-yours-f93a2d431472
  - https://www.designsystemscollective.com/the-incomplete-history-of-design-tokens-61581c573e5d
  - https://thedesignsystem.guide/what-are-design-tokens

- **2 concrete ways this improves wubuwizard / wubuos**:
  1. **wubuwizard**: Define a `tokens/` directory in the repo with a single source-of-truth token file (JSON or the W3C DTCG format) covering colors (CLI accent, error, success), typography (mono font, size scale), spacing (terminal cell size, padding), and semantic tokens (primary-action, secondary-action, muted). The existing C11 theme engine reads these tokens at build time via a code-generation step (`tools/gen_tokens.c`) — no more hardcoded `#define` color values scattered across headers.
  2. **wubuos**: Extend the theme engine to consume the same token file. Win98/XP aesthetic consistency is achieved by defining tokens for the Win98 palette (beige `#C0C0C0`, window grey `#C3C3C3`, title bar blue `#000080`, button face `#F0F0F0`) as semantic tokens rather than raw hex values. When a user switches themes (e.g., to a dark mode), only the token file changes — the entire UI (dosgui windows, Control Panel, theme engine) updates atomically.

---

### J4. Research-to-Implementation Loops (Gap-Closing, Research Cron, Paper-to-Code)

- **Hop chain**:
  1. **Seed** — Chris Olah & Shan Carter, "research debt" essay: fields choke on undigested, poorly explained ideas; the gap between reading papers and implementing results is the core bottleneck.
  2. **Build-measure-learn** (Lean Startup, Steve Blank): the feedback loop that turns research hypotheses into validated implementations — build a minimal version, measure against the paper's claims, learn what works.
  3. **"Closing the Auto-Research Loop"** (arXiv 2603.22376, 2025): AI co-scientist that automates idea generation → code implementation → GPU training → result analysis, iterating under human steering — the research loop as an automated pipeline.
  4. **AutoRA** (JOSS 2024, Musslick et al.): Automated Research Assistant for closed-loop empirical research — the first open-source tool that implements the full loop (hypothesis → experiment → analysis → next hypothesis).
  5. **Kevin-Bacon 7-hop method** (from the `kevin-bacon-research` skill): the research cron (job `7a0a8de2b3c3`, every 6h) that runs the self-contained loop — read MASTER-INDEX → pick next bank-less avenue → 3–5 parallel web_search → web_extract → build 1000-gap bank → verify count → commit.
  6. **SciCode** (arXiv 2407.13168): 338 problems that test an agent's ability to translate conceptual understanding from papers into working code — the paper-to-code benchmark that validates the loop.
  7. **Convergence** — the research-to-implementation loop must have a verification gate: every research finding that enters the gap ledger must produce wired code that passes a test (ASAN clean, cosine in tolerance, `make test_all` green) before the gap is marked `wired`.

- **Convergence**: *Research produces wired code, not just docs — every gap that is closed must have a corresponding implementation that passes an automated verification gate; the loop is only as honest as its close-rate vs. creation-rate ratio.*

- **Sources**:
  - https://arxiv.org/html/2603.22376v2 (Closing the Auto-Research Loop)
  - https://joss.theoj.org/papers/10.21105/joss.06839 (AutoRA)
  - https://worldbench.github.io/awesome-ai-auto-research (SciCode benchmark)
  - https://theleanstartup.com/principles (Build-measure-learn)
  - https://medium.com/techx-official/what-good-researchers-do-differently-bbc494824fa9 (research debt)
  - https://arxiv.org/html/2605.18661v1 (AI for Auto-Research roadmap)

- **2 concrete ways this improves wubuwizard / wubuos**:
  1. **wubuwizard**: Formalize the "wired" gate in the gap-closing loop. When a gap is closed (e.g., a new KV-quant method is implemented), the following must happen before the gap status flips from `open` to `wired`: (a) ASAN/valgrind clean, (b) cosine similarity within tolerance of the paper's reported numbers, (c) `make test_all` passes, (d) the gap entry in INDEX.md includes a one-line verification note. This prevents "paper claims" from being marked as wired without proof.
  2. **wubuos**: Implement a research cron (inspired by the `KB-growth-research` cron in the WuBuOS repo) that runs weekly: (a) reads MASTER-INDEX for the next un-researched avenue, (b) fires 3 parallel web_search queries, (c) web_extracts 2–4 canonical sources into `docs/compendium/05-sources/`, (d) generates a 100-gap mini-bank for the avenue, (e) closes the first 5 gaps with real C11 modules + tests. The cron output is a report visible to the user — the same "consistent research" discipline the user has asked for.

---

### J5. Documentation Architecture (Diátaxis, OpenAPI, Docs-as-Code)

- **Hop chain**:
  1. **Seed** — Diátaxis (Daniele Procida, diataxis.fr): four documentation quadrants — tutorials (learn by doing), how-to guides (solve a real-world problem), reference (factual technical description), explanation (context and background). Each quadrant has a different purpose, audience, and writing style.
  2. **Divio documentation system** (Divio/Django CMS): the same four-type taxonomy, applied to Django project documentation; the framework that popularized the distinction in the Python ecosystem.
  3. **Docs-as-code** (Write the Docs): documentation with the same tools as code — version control (Git), plain text markup (Markdown), code reviews, automated tests, CI/CD deployment. "If your API change doesn't include documentation updates, it's not ready to ship."
  4. **OpenAPI Specification** (OpenAPI Initiative, learn.openapis.org): machine-readable API descriptions that generate both human-readable documentation and client/server code — the "docs as code" principle applied to APIs.
  5. **AGENTS.md** (agents.md): one repository-level context file for all AI coding agents — progressive disclosure (root `AGENTS.md` for global rules, nested `AGENTS.md` for sub-projects), agent-agnostic, kept current. Used in 60,000+ projects.
  6. **Anthropic "Effective context engineering for AI agents"** (anthropic.com): CLAUDE.md as the project-level context file, progressive disclosure (agents discover context via glob/grep rather than loading everything), compaction for long-horizon tasks, sub-agent architectures for complex projects.
  7. **Convergence** — documentation architecture must serve two readers (humans and agents) with four content types (tutorial, how-to, reference, explanation), stored as code (versioned, reviewed, tested), and kept current through a verification gate tied to the code review process.

- **Convergence**: *Documentation is code: versioned, reviewed, tested, and structured into four quadrants (tutorial/how-to/reference/explanation) that serve both human users and AI agents — the same Diátaxis taxonomy applied through a docs-as-code pipeline with machine-readable specs (OpenAPI) and agent context files (AGENTS.md).*

- **Sources**:
  - https://diataxis.fr/start-here/
  - https://www.writethedocs.org/guide/docs-as-code/
  - https://learn.openapis.org/
  - https://agents.md/
  - https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
  - https://www.divio.com/blog/beginners_guide_to_documentation/

- **2 concrete ways this improves wubuwizard / wubuos**:
  1. **wubuwizard**: Restructure the existing documentation into Diátaxis quadrants. Create a `docs/` directory with four subdirectories: `tutorials/` (getting started, first inference run), `how-to/` (specific tasks like "how to quantize KV cache", "how to add a new model adapter"), `reference/` (API docs, struct reference, generated from Doxygen or a custom tool), `explanation/` (architecture rationale, design philosophy, the 7-hop method). Each quadrant gets its own `AGENTS.md`-style context file so agents can navigate them. Add a CI check that fails if an API change doesn't include a corresponding `how-to/` or `reference/` update.
  2. **wubuos**: Create a `docs/AGENTS.md` at the repo root that serves as the single agent context file for WuBuOS. Use progressive disclosure: the root `AGENTS.md` gives the high-level architecture (Styx/9P namespace, dosgui WM, theme engine, ZealOS base) and points agents to subdirectories for details. Each subsystem (theme engine, dosgui, VSL) gets its own nested `AGENTS.md` that describes its purpose, key files, and how to test it. This mirrors the Anthropic "effective context engineering" pattern — agents discover context on demand rather than loading everything, keeping context windows tight and relevant.

---

# Cross-theme synthesis (J)

The five themes above converge on a single meta-principle: **coherence requires structure, and structure must be machine-readable and versioned, not just written and forgotten.** The following are the five highest-leverage planning/UX actions for the two repos, ranked by impact:

1. **ADR + RFC backbone** — Every architectural decision in both repos gets an ADR (Nygard template, append-only, versioned next to code) and every major change gets a lightweight RFC with a close-commitment date. This is the single highest-leverage action because it solves the "monolithic, not agnostic, hard for AI agents to work on" pain directly: agents can now query the ADR log to understand why decisions were made, and RFCs give them a clear proposal/review/accept cycle to participate in. Without this, every other improvement is built on sand.

2. **Honest gap ledger with WIP limits and close-commitments** — The `research/INDEX.md` and MASTER-INDEX banks become flow systems with input-rate tracking, WIP limits, and explicit close-commitments. Every open gap gets an owner, priority, and deadline; items that don't close within the commitment window are explicitly rolled over or dropped with a reason. This prevents the backlog treadmill and makes the ledger trustworthy to both human contributors and AI agents.

3. **Design tokens as the single source of truth** — One token file (W3C DTCG format) that covers colors, typography, spacing, and semantic tokens for both repos. The token file feeds the C11 theme engine in wubuwizard and the runtime theme engine in wubuos, ensuring CLI + GUI + web surfaces share the same visual language. Win98/XP aesthetic consistency becomes a token taxonomy problem, not a hardcoded-value problem.

4. **Research→wired code loop with a verification gate** — Every research finding that enters the gap ledger must produce wired code that passes an automated verification gate (ASAN clean, cosine in tolerance, `make test_all` green) before the gap is marked `wired`. The research cron runs weekly, and the first 5 gaps of each new wave are closed with real C11 modules + tests in the same turn (M1 close-commitment). This ensures research produces code, not just docs.

5. **Diátaxis documentation + AGENTS.md for agent navigation** — Documentation is restructured into four quadrants (tutorial/how-to/reference/explanation), stored as code (versioned, reviewed, tested), and served to agents via progressive-disclosure `AGENTS.md` files at the repo root and per-subsystem. This gives both human users and AI agents a navigable, trustworthy documentation surface — the single source of truth for "how this works" and "how to use this."

---

*Research completed via 7-hop Kevin-Bacon chains across 5 topics. All sources cited are real, verified web pages. No sources were fabricated.*

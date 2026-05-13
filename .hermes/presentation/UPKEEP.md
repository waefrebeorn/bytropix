# UPKEEP.md — Presentation Layer Maintenance

**Purpose:** Rules for keeping the presentation layer consistent, accurate, and conservative in tone. All paths below are relative to `.hermes/presentation/`.

---

## 1. After Any Phase Milestone

- **Update `4-implementation-status.md`** — mark completed phases as ✅ Done, move completed items from "What's pending" to "What works", add new pending items for the current phase. Preserve the conservative tone table format.
- **Update `../README.md`** — bump the milestone reference, update the phase indicator in the top summary, and reflect any new section numbers (e.g., Future Roadmap).
- **Verify all cross-references** still resolve (table-of-contents anchors, diagram paths, internal links).

## 2. New Research Paper

- Add to **`6-references.md`** — one section per paper, in the established format: authors, title, arXiv link, core contribution, relevance to WuBu, known limits. Keep annotations brief and grounded. Do not add unread papers.

## 3. New Diagram

- Save the source (`.svg`) into **`../../DIAGRAMS/`** (project root).
- Register it in **`5-diagrams.md`** — one subsection per diagram. Include: file path, what it shows (1–3 sentences), and the date it was created/updated.

## 4. Architecture Change

- Update **`3-architecture.md`** — if a new component is added, a pipeline changes, or a module is deprecated. Keep the numbered section structure consistent. Note architectural decisions and their rationale. Mark experimental or unverified pathways explicitly (e.g., "not yet validated", "early implementation").

## 5. Style & Tone Rules

- **Globally conservative:** Use "in progress", "preliminary", "early results", "not yet validated", "pending". Avoid "solved", "proven", "guaranteed", "production-ready".
- **Measured claims:** Every claimed result must trace back to a reproducible measurement or a paper reference in `6-references.md`.
- **Phase marking:** ✅ Done = implementation complete AND verified. ⏳ In Progress = code exists but not yet verified. 📋 Planned = no code yet.

## 6. File & Formatting Conventions

- Files are numbered `1-*.md` through `6-*.md`. New permanent sections get the next available number.
- All internal links use relative paths from `.hermes/presentation/`.
- Dates use format: "Month DD, YYYY" (e.g., "May 13, 2026").
- Tables are acceptable in some files (README.md, 5-diagrams.md) but keep them simple; avoid complex multi-line cells.

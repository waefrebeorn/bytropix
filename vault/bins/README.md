# Vault Versioning System

## Purpose
Preserve historical mind palace snapshots so we can track what we knew and when.
Old versions are NOT deleted — they're archived here with timestamps.

## Structure
```
vault/bins/
  state-v{N}-{DATE}-{TIME}.md        — State dashboard snapshots
  plan-v{N}-{DATE}-{TIME}.md         — Plan versions
  goal-mantra-v{N}-{DATE}-{TIME}.md  — Goal paste versions
  testing-v{N}-{DATE}-{TIME}.md      — Testing protocol versions
  entry-v{N}-{DATE}-{TIME}.md        — Entry point versions
  project-{DATE}.md                  — Project overview snapshots
```

## Rule
Before overwriting any `.hermes/mind-palace/*.md`, copy the old version here.
This gives us:
- Audit trail: what did we believe on May 15 vs May 16?
- Rollback: was the old testing.md more useful?
- Meta: how has our understanding of the project changed?

## Current Archive (May 16)
| File | Date | Notes |
|------|------|-------|
| state-v10-May15-PM.md | May 15 PM | "Everything is ✅" era |
| plan-v11-May16-AM.md | May 16 AM | First honest bugs list |
| goal-mantra-v10-May15-PM.md | May 15 PM | Pre-honesty goal paste |
| testing-v7-May15-PM.md | May 15 PM | 20-test harness spec |
| project-May15.md | May 15 | Project overview |
| entry-v6-May15-PM.md | May 15 PM | Build commands |

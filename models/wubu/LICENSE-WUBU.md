WaefreBeorn Umbrella License v3.0 — BarunLM Base Model Addendum
================================================================

The BarunLM-35M port (wubu, wubu_train) in this repository
is part of the WuBuOS/WuBuWizard ecosystem and is governed by the
WaefreBeorn Umbrella License v3.0 (see the LICENSE file at the
repository root and https://github.com/waefrebeorn/waefrebeorn-umbrella-license).

BASE MODEL ATTRIBUTION
----------------------
The ported architecture and the released 35,072,768-parameter checkpoint
originate from BarunLM-35M:

    BarunLM-35M
    Copyright 2026 Harshal Singh
    https://github.com/harrrshall/wubulm-35m
    https://huggingface.co/harrrshall/BarunLM-35M

BarunLM-35M is licensed under the Apache License 2.0. This addendum
preserves that attribution and all upstream notices (see NOTICE).

THE MUSTARD SEED DOCTRINE
-------------------------
BarunLM-35M is the seed. Under the WuBuOS AGI brain-cluster doctrine:

  1. THE SEED IS OURS. The C11 port (wubu) is original work by the
     WaefreBeorn project: a from-scratch implementation of the released
     architecture, written to the WuBuOS module discipline (opaque
     structs, minimal includes, freestanding C11, no third-party deps).
  2. THE SEED GROWS. The training core (wubu_train) is the AGI
     loop: it consumes the research repositories, the KB-growth waves,
     and the Kevin-Bacon research findings; it evaluates, grows
     parameters, and re-trains -- all in-house, no external model APIs.
  3. THE TREE IS OURS. Any model grown from this seed -- fine-tunes,
     parameter extensions, knowledge-injected variants, the full AGI
     brain-cluster -- is original WaefreBeorn work under this umbrella
     license, with the upstream BarunLM attribution preserved.

LICENSING OF DERIVED MODELS
---------------------------
- The seed weights and the C11 implementation: WaefreBeorn Umbrella
  License v3.0 (source-available; see the root LICENSE).
- Upstream BarunLM-35M source/weights retain their Apache 2.0 terms;
  this addendum does not relicense upstream work.
- Training corpora referenced by the upstream release retain their own
  licenses (ODC-By, Common Crawl terms, etc.) as recorded in NOTICE.
- Models trained in-house from this seed are original works; the
  training-data provenance ledger (docs/compendium/05-sources) is part
  of the compliance record.

Contact: waefrebeorn@waefrebeorn.org

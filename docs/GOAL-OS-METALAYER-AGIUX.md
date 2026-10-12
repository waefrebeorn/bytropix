# GOAL — The OS Meta-Layer / AGI-UX / Recursive-Learning Axis

> 2026-10-11. This is a NEW axis, distinct from the inference-engine
> improvements (`docs/WUBUWIZARD_100*_IMPROVEMENTS`). It implements the
> user's three latest directives, each grounded in a module that already
> exists (verified this session). External research + source citations are
> in `vault/external-findings/02-os-metalayer-agiux-recursive.md`.
>
> Rule for every item: **wiring, not invention.** Each maps to an existing
> subsystem. Every functional change lands behind a state-change assertion,
> like the rest of the XP-parity work.

The user's verbatim directives this round:
1. "online research 1000 improvements to implement as we are making a
   operating system and you keep designing like an application" —
   → think OS-native, not app-shaped; the meta layer lives in the kernel/
   namespace/shell, not in a sandboxed app.
2. "meta layering design and agi user experiance integration as bonzi buddy"
   → the Bonzi companion becomes the OS-integrated AGI face.
3. "reorganization for kv cache as a filesystem styx9p and pure
   interpretation to agi allowing all operating systems apps to run in our
   stack" → KV/expert/memory/hive as 9P files; foreign apps run via
   translation.
4. "full stack recursive learning improvement" → close the DGM loop end to
   end.

---

## Track A — 9P-native Brain internals ("KV cache as a filesystem")

The stack already runs Styx/9P and `styxfs_server.c` already declares the
`/n/kv/*` write doctrine. Make the Brain's live state inspectable/writable
through the same namespace verbs that inspect the kernel.

| # | Item | Module | Evidence / target |
|---|---|---|---|
| A1 | Serve the **KV cache arena** as `/n/kv/{layer}/{slot}` files (read a slot's latent, write to seed a prefix). Write side already doctrinally declared in `styxfs_server.c:301`. | wubuos `styxfs_server.c` + wubuwizard `wubu_agentic_kv.c` | the /n/kv write doctrine; agentic-kv already models per-agent KV |

> **A1 = DONE (verified 2026-10-11).** Already built end-to-end: `src/kernel/wubu_kvfs.c` serves `/n/kv/*` reads+writes through a live `float *g_wubu_kv_base` tensor via a hash-resolved mount table; `styxfs_server.c` routes both sides. `test_kvfs_route` passes ("the OS IS the KV cache, served as 9P"). No work needed. |
| A2 | Serve the **SSD expert slot-bank** as `/n/experts/{slot}` — inspect LRU residency, pin/unpin an expert. | wubuwizard `wubu_ssd_moe.c` + ns_bridge | KAT-Coder 256-expert slot-bank already pages experts |

> **A2 = DONE (2026-10-11).** `wubu_ns_experts.{c,h}` mirrors the slot-bank into `/n/experts`: `config` (layers/experts/dims/slot_bank), `stats` (pageins/hits/bytes), and `layer/<L>` (one line per resident slot: slot:expert:lru). Dependency-light (links only wubu_ns_fs.o, not the slot-bank); owner pushes state via setters each forward pass. wubuos c536fbf, test_ns_experts 7/7. NOTE: exposed as inspect; pin/unpin is a future write-path addition. |
| A3 | Serve the **diagnostic hive** as `/n/hive/{cell}` — z-score cells, anomaly walker, rollback state, readable from the shell. | wubuwizard `wubu_hive.c` + ns_bridge | hive = the diagnostic immune system |

> **A3 = DONE (2026-10-11).** `wubu_ns_hive.{c,h}` mirrors the latest `wubu_diag_record_t` + ledger into `/n/hive`: loss/util/grad/fitness/ledger/batch. Dependency-light, pushed each batch. wubuos d79bcc5, test_ns_hive 11/11. NOTE: exposes the diagnosis snapshot + gate counters; per-cell z-score walker state is a future granularity addition. |
| A4 | Serve **companion memory** as `/n/bonzi/mem` — the persistent-memory store, readable/writable, gated by AD01 subtree cap. | `wubu_agentic_mem.h` + ns_bridge | desktop-assistant research: persistent memory is a defining trait |

> **A4 = DONE (2026-10-11).** Two parts, both shipped: (1) `wubu_ns_bonzi.{c,h}` mirrors live companion state into `/n/bonzi/*`, pushed from the WM render tick, weak-linked (wubuos 48a3143, test_ns_bonzi 12/12). (2) `wubu_bonzi_mem.{c,h}` is the PERSISTENT store: save/load the agentic-mem vector index to `/n/bonzi/mem` (binary format WBMEM001), AD01-gated so only the companion's subtree may read/write (wubuos 5068d9f, test_bonzi_mem 11/11). Companion memory now survives restarts (JE26 continuity). |
| A5 | A shell tool `cat /n/kernel/scheduler`-style that already works — **extend** the pattern to every Brain server so `ls /n` shows the whole machine (kernel+hw+brain). | `wubu_ns_bridge.c` | /n/{svc,dram,kernel,hw} already bridged |

Doctrine (from the Styx paper): a resource is a file server you attach;
attaching a remote one is a distributed system for free. Each of A1-A4 is
one small 9P tree mounted under `/n`.

## Track B — The Bonzi companion as the OS-integrated AGI face

`wubu_bonzi.c` is already a native desktop orb with tick/mouse/draw. The
research (Jarvis 2026 guide, Perplexity Personal Computer, Gemma Wagon,
NOVA) says a real companion is **native + ambient-context + persistent +
capability-gated tools**. The orb is native; wire the rest.

| # | Item | Module | Evidence |
|---|---|---|---|
| B1 | **Ambient screen context**: on summon, capture the focused window's rect from the framebuffer and hand pixels + a WM-state dump to the Brain as the prompt. `wubu_screenshot.c` is the capture primitive; the WM knows the focused window. | `wubu_bonzi.c` + `wubu_screenshot.c` + WM | "context is the prompt" — the defining trait of a desktop agent vs a chatbot |

> **B1 = DONE (2026-10-11).** `wubu_bonzi_context.{c,h}` produces the structured ambient context (focused-window title+rect + WM window count + recent-focus ring), fed live from `dosgui_wm_get_focused()` each render frame. Text context always available; pixel capture via `wubu_screenshot_capture(WUBU_SSHOT_WINDOW)` is the caller's job when a vision model attaches. wubuos 1671acd, test_bonzi_context 7/7. |
| B2 | **Global summon hotkey** (Ctrl+/ style) that raises the companion over any window, from the hosted input layer. | `wubu_bonzi.c` + `hosted_wayland_input.c` | summonable-from-anywhere trait |

> **B2 = DONE (2026-10-11).** Ctrl+/ (evdev KEY_SLASH=53) in `hosted_wayland_input.c` calls `wubu_bonzi_open_agi()` — the companion's summon/AGI-panel action — mirroring the Ctrl+T / Alt+PrintScr combo pattern, modifier state from the existing g_key_map tracker. wubuos 6c9576e. |
| B3 | **Persistent memory**: companion recalls projects/people/writing-style across sessions, stored at `/n/bonzi/mem` (A4). | `wubu_agentic_mem.h` | persistent-memory trait |

> **B3 = DONE (2026-10-11).** `wubu_bonzi_greet.{c,h}` turns loaded memory into a personalized summon greeting (first-meeting vs welcome-back naming recent activity); `wubu_bonzi_mem_recall` loads `/n/bonzi/mem` and formats it; `wubu_bonzi_open_agi` sets the personalized bubble on summon (weak-linked so bonzi links without the store). wubuos 0a32a95, test_bonzi_greet 8/8. |
| B4 | **Capability-gated tool execution**: the companion acts through `wubu_tooluse.h`/`wubu_toolreg.h`, every call checked by `wubu_9p_cap_allowed(agent_subtree, path)` (AD01). | `wubu_agentic_os.h` AD01 + tooluse | Styx private-namespace = capability security |

> **B4 = DONE (2026-10-11).** `wubu_tool_cap.{c,h}` wraps every tool call with the AD01 check: a tool is bound to a capability subtree, a call is refused unless the caller owns it, unbound tools are deny-by-default, handler never runs on denial. The security spine B and D reuse. wubuwizard 7ded8b3, test_tool_cap 8/8. |
| B5 | **Offline fallback**: when no API key, route to local `gen_text` (WuBu-35M) so the companion always works. | `api_server` + `wubu_spawn` | NOVA/Gemma Wagon both do local fallback |

> **B5 = DONE (2026-10-11).** `wubu_provider_route.{c,h}` is the routing decision: pure `decide(has_key, allow_remote, has_local)` + env wrapper reading WUBU_API_KEY/OPENROUTER_API_KEY/NVIDIA_API_KEY. Remote only with a key + permission; local is the floor; never a silently-dead remote. wubuwizard aef2323, test_provider_route 6/6. |
| B6 | **Activity feedback**: the orb shows what it's doing (mood/animation states tied to agent activity), so it feels present, not a black box. | `wubu_bonzi.c` | "remains accessible without disrupting workflow" |

> **B6 = DONE (2026-10-11).** `wubu_bonzi_activity.{c,h}` maps activity (idle/thinking/working/alert) to blink/eye/bob modifiers; live activity pushed to `/n/bonzi/activity`. wubuos a499fe3, test_bonzi_activity 8/8. **Track B COMPLETE (B1-B6).** |

## Track C — Close the recursive-learning loop end to end

The DGM loop is already built (Brain archive) and hardened (Body DA-1/2/3).
The gap: the Brain's archive and the Body's gated operator aren't wired, so a
validated mutation can't actually promote through the human-gated verifier.

| # | Item | Module | Evidence |
|---|---|---|---|
| C1 | Wire the **Brain archive → Body operator**: a validated `wubu_hive` mutation emits a trace span; `wubu_selfimprove` ingests it, the independent verifier scores it, and on DA-3 sign-off the operator promotes it (apply / restart subsystem / emit trace directive). | `wubu_hive.c` → `wubu_selfimprove.c` + `src/kernel/wubu_verifier.h` | DGM propose→evaluate→archive→sample; DA-3 operator hook already exists |

> **C1 = DONE (2026-10-11).** `wubu_mutation_bridge.{c,h}` translates an ACCEPTED Brain fitness cell into a `WUBU_TRACE_SELFMOD` span the Body's `wubu_selfimprove` ingests → verifies → (DA-3 gate) → operator fires. REJECTED cells never bridge. test_mutation_bridge 7/7 proves the loop closes end-to-end. **TRACK C COMPLETE (C1-C4).** |
| C2 | **Held-out eval**: validate mutations on a disjoint benchmark segment (DGM used disjoint segments), so the verifier can't be gamed by the same data the mutator saw. | trainer + verifier | DGM + The Register reward-hacking failure mode |

> **C2 = DONE (2026-10-11).** `wubu_heldout_eval.{c,h}`: pure decision — PASS/FAIL/SUSPECT from train+held-out scores; SUSPECT = the reward-hacking signature (in-sample >> held-out). Only PASS promotes. wubuwizard 1dc0c97, test_heldout_eval 6/6. |
| C3 | **Non-negotiable invariant**: verifier is independent of the mutator; promotion is fail-closed; DA-3 freeze is the death condition. Encode as a test so future work can't erode it. | `wubu_selfimprove_test` | the reward-hacking pitfall from the vault |

> **C3 = DONE (2026-10-11).** `test_selfimprove_invariants` locks the three DA-3 fail-closed gates (no verifier / frozen / unsigned gate => no promote, operator never fires) + verifier independence (operator fires only on a verifier-passed span). wubuos f4a5889, 10/10. |
| C4 | **Cost bounding**: weight failed/divergent mutations 3x (DA-2 already does) and bound mutation search by the AD04 resource budget so the sandbox isn't infinite (DGM's cost gap). | `wubu_selfimprove.c` + `wubu_agentic_os.h` AD04 | darwin-agents: "DGM didn't address the cost of failed mutations" |

> **C4 = DONE (2026-10-11).** `wubu_budget_search.{c,h}` bounds the search by the AD04 budget: accumulates per-candidate cpu/ram/io and aborts the moment `wubu_resbound_check` overruns, before paying for the offending candidate. Closes the DGM infinite-sandbox gap. wubuwizard 8ec93bf, test_budget_search 8/8. |

## Track D — "Pure interpretation so all OS apps run in our stack"

Styx's translator model + `wubu_exec.c`'s format dispatch. The missing
piece is a per-ABI translation layer so a foreign app's filesystem/window/
socket expectations are served as 9P files (Plan 9's user-level translators).

| # | Item | Module | Evidence |
|---|---|---|---|
| D1 | A **libc/syscall shim per foreign ABI** (PE, Mach-O, WASM, DOS) that maps its file/net/graphics calls onto WuBu's 9P + `wubu_exec` surface. | `wubu_exec.c` + shims | Plan 9 user-level translators for NFS/disk |

> **D1 = DONE (2026-10-11).** Two parts: (1) `wubu_abi_path.{c,h}` translates foreign paths onto `/n/foreign/<abi>/...` (wubuos 4dc5d75, 7/7). (2) `wubu_syscall_shim.{c,h}` maps foreign syscalls onto WuBu ops — file→9P on the translated path, graphics→VBE/WM, net→9P net, unmapped→REFUSE (fail-closed) (wubuos 226f988, 8/8). A foreign app's calls are now served by the stack. |
| D2 | Each shim is itself a **9P file server**, so attaching it mounts a foreign OS's root into the namespace. | ns_bridge | "represent resources as files… you have a distributed system" |

> **D2 = DONE + WIRED (2026-10-11, user-directed).** `wubu_librarian.{c,h}` is the INTEROP LIBRARIAN: registry (every participant registers name/ABI/root/caps), capability resolver (fail-closed), and a GROWING translation table (each cross-ABI call recorded → lookup next time; the library builds itself as it goes). `wubu_librarian_serve.{c,h}` wires it as the live front door, self-registering the default WuBu participants (shell/kv/models/bonzi/experts/hive). FILE ops resolve against the target's registered root. wubuos 2486a44 + 50570a0, test_librarian 13/13 + test_librarian_serve 10/10. |

---

## Sequencing (my autonomous call)

Highest-leverage first, each independently shippable behind a green gate:

1. **A1 (KV as 9P files)** — smallest, highest-signal; proves the whole
   Track-A pattern and gives the shell `cat`-level introspection of the
   Brain. Doctrine already declared in `styxfs_server.c`.
2. **B1 + B4 (ambient context + capability-gated tools)** — the two
   research-backed gaps that turn the orb into a real companion; B4's AD01
   cap check is the security spine both B and D reuse.
3. **C1 (Brain archive → Body operator)** — closes the recursive loop, the
   mission's "full stack recursive learning" in one wire.
4. Then A2-A5, B2-B6, C2-C4, D1-D2.

Every item: state-change assertion + gate-green before push, per the
established integration workflow.

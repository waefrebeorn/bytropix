# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Hermes substrate. No model calls: argv, staging and clipping only.

Run: python3 tests/test_hermes_substrate.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness2.substrate.hermes import HERMES_SURFACE, HermesSubstrate  # noqa: E402

FAILS = 0


def check(cond: bool, label: str) -> None:
    global FAILS
    if cond:
        print(f"PASS  {label}")
    else:
        FAILS += 1
        print(f"FAIL  {label}")


def main() -> int:
    sub = HermesSubstrate()

    s = sub.surface()
    check(s.name == "hermes", "surface name is hermes")
    check("AGENTS.md" in s.always_on, "AGENTS.md is an always-on file")
    check(s.skill_dir == "skills", "skill directory is skills")

    argv = sub.run_argv("do the thing", Path("/tmp/ws"), "model-x")
    check(argv[1] == "-z" and argv[2] == "do the thing", "prompt passed with -z")
    check("--in" in argv and "/tmp/ws" in argv, "--in sets the workspace")
    check("-m" in argv and "model-x" in argv, "-m sets the model")
    check(argv[-1] == "--pass-session-id", "session id is passed back")

    tmp = Path(tempfile.mkdtemp())
    tree = tmp / "tree"
    tree.mkdir()
    (tree / "AGENTS.md").write_text("# h\n", encoding="utf-8")
    (tree / "skills").mkdir()
    install = sub.materialise(tree, type("S", (), {"install_mode": "home_dir"})(),
                              {"file:notes/x.md": "hello"})
    check((install.staging_dir / "AGENTS.md").is_file(), "materialise copies the tree")
    staged_extra = install.staging_dir / "notes" / "x.md"
    check(staged_extra.is_file() and staged_extra.read_text() == "hello",
          "materialise writes ctx file: entries")
    check(len(install.fingerprint) > 0, "install carries a fingerprint")

    bad = tmp / "empty"
    bad.mkdir()
    try:
        sub.materialise(bad, type("S", (), {"install_mode": "home_dir"})(), {})
        check(False, "materialise rejects a tree without AGENTS.md")
    except ValueError:
        check(True, "materialise rejects a tree without AGENTS.md")

    t = tmp / "long.txt"
    t.write_text("a" * 5000, encoding="utf-8")
    out = sub.read_trajectory(t, obs_trunc=100, max_chars=1000)
    check(len(out) < 1200 and "truncated" in out, "read_trajectory clips long output")

    out2 = sub.read_trajectory(tmp / "missing.txt", obs_trunc=10, max_chars=10)
    check("unreadable" in out2, "read_trajectory reports a missing file")

    try:
        sub.preflight()
        check(True, "preflight finds the hermes binary")
    except RuntimeError as exc:
        check(False, f"preflight failed: {exc}")

    print(f"\n{'ALL PASSED' if FAILS == 0 else str(FAILS) + ' FAILURES'}")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
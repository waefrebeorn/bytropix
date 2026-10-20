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

"""Hermes Agent as a Harness² substrate.

Hermes is the executor used by the WuBu AGI loop, so a Hermes substrate lets
Harness² propose-probe-compose the very harness that drives that loop, instead of
a separate agent framework.

The editable surface is Hermes-native:

  AGENTS.md              always-on instructions (analogue of systemprompt.md)
  skills/<name>/SKILL.md on-demand skills, name must match the directory
  scripts/*.py           deterministic helpers
  system_prompt.md       optional persona / task framing

`hermes -z "<prompt>"` runs one non-interactive turn; `--in <workspace>` sets the
working directory and `-m <model>` pins the model. The session id can be captured
with `--pass-session-id`, which is what lets a probe be re-read after the fact.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

from ..core import runmark as RM
from ..core import tree as HT
from ..core.tree import SurfaceSpec
from .base import Install, Launcher, RunRequest

HERMES_SURFACE = SurfaceSpec(
    name="hermes",
    always_on=("AGENTS.md",),
    always_on_max=24_000,
    always_on_cap_reason="every token here competes with the agent's own output budget",
    skill_dir="skills",
    skill_name_rule="must_match",
    trigger_hint=(
        "Fix it by adding a trigger line naming `{name}` to AGENTS.md, e.g. "
        '\"When <the situation this skill handles>, load the `{name}` skill.\"'
    ),
    autotrigger_file="AGENTS.md",
    trigger_line="- When {desc}, load the `{name}` skill.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=None,
    stubs=(
        (
            "skills/README.md",
            "<!-- c6: reusable workflows loaded on demand via the skill tool. Add one as "
            "skills/<name>/SKILL.md with frontmatter `name:` matching the directory. -->\n",
        ),
        (
            "scripts/README.md",
            "<!-- c7: deterministic helpers, run as `python3 scripts/<name>.py`. -->\n",
        ),
    ),
)


def hermes_bin() -> Path:
    """The hermes executable: $HERMES_BIN, then PATH, then ~/.local/bin/hermes."""
    env = os.environ.get("HERMES_BIN")
    if env:
        return Path(env)
    found = shutil.which("hermes")
    if found:
        return Path(found)
    return Path.home() / ".local" / "bin" / "hermes"


class HermesSubstrate:
    """Hermes behind the one Substrate interface.

    `launcher` is optional: benchmarks that own a launcher body pass one; the
    self-contained path below runs `hermes -z` directly, which is what the WuBu
    AGI loop uses.
    """

    name = "hm"

    def __init__(self, launcher: Launcher | None = None):
        self._launcher = launcher

    # -- surface ---------------------------------------------------------
    def surface(self) -> SurfaceSpec:
        return HERMES_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        tree_dir = Path(tree_dir)
        if not HT.is_tree(HERMES_SURFACE, tree_dir):
            raise ValueError(
                f"{tree_dir} is not a hermes harness tree (AGENTS.md is absent)"
            )
        staged = Path(tempfile.mkdtemp(prefix="adapt-hm-"))
        try:
            shutil.copytree(tree_dir, staged, dirs_exist_ok=True)
            for key, body in ctx.items():
                if key.startswith("file:"):
                    rel = HT.normalize_rel(key[len("file:") :])
                    p = staged / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(body, encoding="utf-8")
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        return Install(
            staging_dir=staged,
            fingerprint=RM.fingerprint(tree_dir),
            extras={"install_mode": "home_dir"},
        )

    # -- execution -------------------------------------------------------
    def run_argv(
        self,
        prompt: str,
        workspace: Path,
        model: str,
        *,
        session: str | None = None,
        toolsets: str | None = None,
    ) -> list[str]:
        argv = [str(hermes_bin()), "-z", prompt, "--in", str(workspace)]
        if model:
            argv += ["-m", model]
        if session:
            argv += ["--resume", session]
        if toolsets:
            argv += ["-t", toolsets]
        argv.append("--pass-session-id")
        return argv

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        if self._launcher is not None:
            return self._launcher(
                RunRequest(task_ref=task_ref, run_ids=tuple(run_ids), timeout=int(timeout)),
                install,
            )
        raise RuntimeError(
            "HermesSubstrate.execute needs a launcher body (prompt + workspace); "
            "the WuBu driver tools/harness2_agi.py supplies one"
        )

    def run_once(
        self,
        prompt: str,
        workspace: Path,
        model: str,
        *,
        timeout: int = 600,
        env: Mapping[str, str] | None = None,
        session: str | None = None,
    ) -> tuple[int, str, str]:
        """Run one non-interactive Hermes turn. Returns (rc, stdout, stderr)."""
        workspace.mkdir(parents=True, exist_ok=True)
        argv = self.run_argv(prompt, workspace, model, session=session)
        e = dict(os.environ if env is None else env)
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, env=e, cwd=str(workspace)
            )
        except subprocess.TimeoutExpired as exc:
            def _text(v) -> str:
                if v is None:
                    return ""
                return v.decode("utf-8", "replace") if isinstance(v, bytes) else v

            return 124, _text(exc.stdout), _text(exc.stderr) + f"\n[timeout after {timeout}s]"
        return proc.returncode, proc.stdout, proc.stderr

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 400
    ) -> str:
        """Read back a transcript. Hermes writes plain text, so this clips it."""
        try:
            text = Path(stream_path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"[unreadable transcript: {exc}]"
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n…(+{len(text) - max_chars} chars truncated)"
        return text

    def preflight(self) -> None:
        b = hermes_bin()
        if not b.is_file():
            raise RuntimeError(
                f"hermes not found at {b}: install Hermes Agent or set HERMES_BIN"
            )
        # A run record is a JSON line; prove the file format is writable.
        json.dumps({"hermes": str(b)})
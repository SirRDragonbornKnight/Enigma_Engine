"""
Enigma Engine - FORGE External Teacher (HTTP) Subprocess Mixin
==============================================================

Spawns ``collect_distill_data.py`` as a subprocess that talks to an
external chat-completion endpoint (Ollama / vLLM / llama.cpp / OpenAI-
compatible) and streams its stdout into the FORGE log panel.

Why a subprocess (not in-process):
- The user's "no in-process AI-on-AI" pivot moved teacher generation
  off-engine: the teacher runs in another process, optionally on
  another machine, behind an HTTP boundary.
- ``collect_distill_data.py`` already implements prompts-mode +
  Magpie-mode + resume + dual-emit (.jsonl + .txt). Reusing it keeps
  one source of truth for the wire format.

Output lands at ``data/finetune/distill_<tag>.{jsonl,txt}``. On
success the FORGE training-data picker auto-fills with the .txt path.

Public surface on the host (ForgeMixin):
- ``_start_external_teacher_corpus()`` — button click handler.
- ``_stop_external_teacher_corpus()`` — STOP TEACHER button handler.
- ``_kill_teacher_subprocess()`` — idempotent shutdown hook
  called from ``_on_close``.

Required widget attributes (created in ``gui_pages_forge.py``):
- ``teacher_endpoint_var`` (StringVar) — endpoint URL.
- ``teacher_model_var`` (StringVar) — required, model name.
- ``teacher_mode_var`` (StringVar) — "magpie" | "prompts".
- ``teacher_magpie_var`` (StringVar) — N (int); required when mode=="magpie".
- ``teacher_prompts_path_var`` (StringVar) — file path; required when mode=="prompts".
- ``teacher_tag_var`` (StringVar) — output filename tag.
- ``teacher_max_tokens_var`` (StringVar) — max tokens per response.
- ``teacher_start_btn``, ``teacher_stop_btn`` — buttons.
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def _check_endpoint_reachable(
    endpoint: str,
    *,
    timeout: float = 2.0,
    _opener=None,
) -> tuple[bool, str]:
    """Best-effort GET ``{endpoint}/models`` to detect server-down before
    spending 30 s of HTTP timeouts inside ``collect_distill_data.py``.

    Volume table (loud-on-real-issue rule):
    - HTTP 2xx → (True, "ok") — silent success.
    - Non-2xx (404 etc.) → (True, "endpoint up, /models returned NNN")
      — best-effort: server is reachable, just doesn't expose /models.
      Spawn anyway, log INFO so user can see.
    - URLError / timeout / connection-refused → (False, reason)
      — the real failure mode; caller WARNs and bails.

    Pure function (stdlib only, no GUI). ``_opener`` is a test seam so
    unit tests can swap in a fake without monkeypatching
    ``urllib.request`` globally.
    """
    url = endpoint.rstrip("/") + "/models"
    opener = _opener if _opener is not None else urllib.request.urlopen
    try:
        resp = opener(url, timeout=timeout)
    except urllib.error.HTTPError as exc:
        # Server responded with non-2xx — endpoint IS reachable.
        return (True, f"endpoint up, /models returned {exc.code}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return (False, str(exc))
    try:
        code = getattr(resp, "status", None) or resp.getcode()
    except (AttributeError, OSError):
        code = 0
    finally:
        try:
            resp.close()
        except (AttributeError, OSError):
            pass
    if 200 <= int(code) < 300:
        return (True, "ok")
    return (True, f"endpoint up, /models returned {code}")


def _build_teacher_argv(
    *,
    endpoint: str,
    model: str,
    magpie_n: int,
    tag: str,
    max_tokens: int = 512,
    python_exe: str | None = None,
    script_path: str | Path | None = None,
    prompts_path: str | Path | None = None,
) -> list[str]:
    """Build argv for ``collect_distill_data.py`` invocation.

    Two mutually exclusive modes (mirrors the script's CLI):
    - **Magpie mode** (default): ``prompts_path is None`` → emit
      ``--magpie N``. Model invents N instruction/answer pairs.
    - **Prompts mode**: ``prompts_path`` set → emit
      ``--prompts <path>``. Model answers each prompt in the file.

    Pure function (no GUI / no I/O) so tests can pin the contract
    without spawning real subprocesses.
    """
    py = python_exe or sys.executable
    script = str(script_path) if script_path is not None else "collect_distill_data.py"
    argv = [
        py,
        script,
        "--endpoint", endpoint,
        "--model", model,
    ]
    if prompts_path is not None:
        argv.extend(["--prompts", str(prompts_path)])
    else:
        argv.extend(["--magpie", str(int(magpie_n))])
    argv.extend([
        "--tag", tag,
        "--max-tokens", str(int(max_tokens)),
        "--resume",
    ])
    return argv


# Progress lines from collect_distill_data.py look like:
#   INFO:__main__:[10/500] ok=10 failed=0 duplicate=0 (1.23 rows/s)
#   INFO:__main__:[10/500] ok=10 failed=0 skipped=0 (1.23 rows/s)
# Tight gate: require both `[N/M]` and ` ok=` so user-supplied prompt
# content that happens to contain `[1/5]` can't fake a progress bump.
_PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\]\s+ok=")


def _parse_teacher_progress(line: str) -> tuple[int, int] | None:
    """Parse a ``[N/M] ok=...`` progress line from the teacher subprocess.

    Returns ``(done, total)`` on match, ``None`` otherwise. Pure helper
    (no GUI / no I/O) so the regex contract is unit-testable.

    Volume table:
    - ``[10/500] ok=10 failed=0 duplicate=0 (1.23 rows/s)`` → (10, 500)
    - ``loaded 500 prompt(s) from ...``                    → None
    - ``[teacher] some user-supplied [1/5] text``          → None
      (no `` ok=`` follows the bracket pair)
    - ``[10/0] ok=...``                                    → None
      (zero total — caller can't compute pct)
    """
    match = _PROGRESS_RE.search(line)
    if match is None:
        return None
    done = int(match.group(1))
    total = int(match.group(2))
    if total <= 0:
        return None
    return (done, total)


def _count_distill_jsonl_rows(
    tag: str,
    *,
    base_dir: str | Path = "data/finetune",
) -> int:
    """Count valid rows in ``<base_dir>/distill_<tag>.jsonl``.

    Returns 0 when the file is missing, unreadable, or has zero rows.
    Pure helper (stdlib only, no GUI). ``base_dir`` is a test seam.
    Volume table:
    - File missing                 → 0 (silent; first-time tag)
    - File present, N>0 valid rows → N
    - File present, malformed/zero → 0 (silent; suggest fresh start)
    """
    p = Path(base_dir) / f"distill_{tag}.jsonl"
    if not p.exists():
        return 0
    try:
        with p.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except (OSError, UnicodeDecodeError):
        return 0


class ForgeTeacherMixin:
    """External-teacher (HTTP) subprocess management.

    Tracks at most one teacher subprocess at a time via
    ``self._teacher_proc``. Idempotent stop / kill — safe to call
    when no process is running.
    """

    # Single-flight: only one teacher subprocess allowed.
    _teacher_proc: subprocess.Popen | None = None
    # In-flight while health-check thread is running but proc not yet spawned.
    _teacher_health_in_flight: bool = False
    # Set by STOP during health-check window so the spawn callback bails.
    _teacher_cancel_requested: bool = False

    def _start_external_teacher_corpus(self) -> None:
        """Validate inputs, spawn ``collect_distill_data.py`` in Magpie
        mode (synth N pairs) or prompts mode (answer rows from a file)
        per ``teacher_mode_var``, stream stdout into the FORGE log,
        auto-fill data picker on exit 0.
        """
        # Single-flight guard: covers both "subprocess running" and
        # "health-check pending" states.
        if getattr(self, "_teacher_proc", None) is not None:
            self._log("[teacher] already running — STOP first")
            return
        if getattr(self, "_teacher_health_in_flight", False):
            self._log("[teacher] endpoint check already in progress")
            return

        endpoint = self.teacher_endpoint_var.get().strip()
        model = self.teacher_model_var.get().strip()
        tag = self.teacher_tag_var.get().strip() or "external"
        # Mode selection happens before magpie_n parse so an empty/bad
        # magpie field is fine when prompts-mode is selected.
        mode_var = getattr(self, "teacher_mode_var", None)
        mode = mode_var.get() if mode_var is not None else "magpie"

        magpie_n = 0
        if mode != "prompts":
            try:
                magpie_n = int(self.teacher_magpie_var.get().strip())
            except (TypeError, ValueError):
                self._log("[teacher] [!] Magpie N must be an integer")
                return
        try:
            max_tokens = int(self.teacher_max_tokens_var.get().strip())
        except (TypeError, ValueError):
            max_tokens = 512

        # Mode selection: prompts file (if given AND exists) OR Magpie.
        # Mutually exclusive — collect_distill_data.py rejects both at once.
        prompts_path: Path | None = None
        if mode == "prompts":
            raw = ""
            pv = getattr(self, "teacher_prompts_path_var", None)
            if pv is not None:
                raw = pv.get().strip()
            if not raw:
                self._log(
                    "[teacher] [!] prompts mode selected but no prompts file given")
                return
            prompts_path = Path(raw)
            if not prompts_path.exists():
                self._log(
                    f"[teacher] [!] prompts file not found: {prompts_path}")
                return

        # Loud-on-real-issue: missing required values fail at the boundary.
        if not endpoint:
            self._log("[teacher] [!] endpoint URL required")
            return
        if not model:
            self._log("[teacher] [!] model name required")
            return
        if prompts_path is None and magpie_n <= 0:
            self._log("[teacher] [!] Magpie N must be > 0")
            return

        argv = _build_teacher_argv(
            endpoint=endpoint,
            model=model,
            magpie_n=magpie_n,
            tag=tag,
            max_tokens=max_tokens,
            prompts_path=prompts_path,
        )
        self._log("[teacher] checking endpoint: " + endpoint)
        self._teacher_health_in_flight = True
        self._teacher_cancel_requested = False

        # Off main thread: don't freeze GUI on a slow/refused connection.
        threading.Thread(
            target=self._teacher_health_check_then_spawn,
            args=(endpoint, argv, tag),
            daemon=True,
            name="forge-teacher-health",
        ).start()

    def _teacher_health_check_then_spawn(
        self, endpoint: str, argv: list[str], tag: str,
    ) -> None:
        """Daemon thread: probe ``{endpoint}/models`` then either spawn the
        subprocess (on main thread via ``after(0, ...)``) or log + bail.
        Always clears ``_teacher_health_in_flight`` on exit so the GUI
        can accept another START request.
        """
        ok, msg = _check_endpoint_reachable(endpoint)
        if not ok:
            def _on_unreach(m=msg):
                self._log(f"[teacher] [!] endpoint not reachable: {m}")
                self._teacher_health_in_flight = False
            self.after(0, _on_unreach)
            return
        if msg != "ok":
            self.after(0, lambda m=msg: self._log(f"[teacher] {m}"))
        self.after(0, lambda: self._spawn_teacher_subprocess(argv, tag))

    def _spawn_teacher_subprocess(self, argv: list[str], tag: str) -> None:
        """Run on the main thread: actually spawn the subprocess after
        the health-check has cleared. Honors STOP-during-health-check
        via ``_teacher_cancel_requested``; clears the in-flight flag
        on every exit path.
        """
        try:
            if getattr(self, "_teacher_cancel_requested", False):
                self._log("[teacher] cancelled before spawn")
                return
            if getattr(self, "_teacher_proc", None) is not None:
                self._log("[teacher] already running — ignoring duplicate spawn")
                return
            self._log("[teacher] spawning: " + " ".join(argv))
        finally:
            # Health-check phase is over either way; the subprocess
            # itself becomes the new single-flight signal via _teacher_proc.
            self._teacher_health_in_flight = False

        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, ValueError) as exc:
            self._log(f"[teacher] [!] spawn failed: {exc}")
            return

        self._teacher_proc = proc
        self._teacher_tag = tag

        # Visual: flip start→stop button states.
        start_btn = getattr(self, "teacher_start_btn", None)
        stop_btn = getattr(self, "teacher_stop_btn", None)
        if start_btn is not None:
            start_btn.configure(state="disabled", text="GENERATING...")
        if stop_btn is not None:
            stop_btn.configure(state="normal")

        threading.Thread(
            target=self._teacher_reader_loop,
            args=(proc, tag),
            daemon=True,
            name="forge-teacher-reader",
        ).start()

    def _teacher_reader_loop(
        self, proc: subprocess.Popen, tag: str,
    ) -> None:
        """Daemon thread: read subprocess stdout line-by-line, push
        each line to the FORGE log via ``after(0, …)`` and update the
        FORGE progress bar when a ``[N/M] ok=`` line is parsed. On
        exit: log status, restore buttons, auto-fill data picker on
        success.
        """
        try:
            stdout = proc.stdout
            if stdout is not None:
                for raw_line in stdout:
                    line = raw_line.rstrip("\r\n")
                    if not line:
                        continue
                    # Marshal to main thread — _log itself is buffered+
                    # main-thread-flushed but we still defer the call.
                    self.after(0, lambda m=line: self._log("[teacher] " + m))
                    # Live progress: parse `[N/M] ok=` lines and push to
                    # the shared FORGE progress bar. _update_forge_progress
                    # already does its own after(0, ...) marshalling.
                    progress = _parse_teacher_progress(line)
                    if progress is not None:
                        done, total = progress
                        pct = int(done * 100 / total)
                        update = getattr(self, "_update_forge_progress", None)
                        if update is not None:
                            update(pct, f"teacher {done}/{total}")
        except (OSError, ValueError) as exc:
            self.after(0, lambda e=exc: self._log(
                f"[teacher] [!] reader crashed: {e}"))

        try:
            rc = proc.wait()
        except (OSError, ValueError):
            rc = -1

        # Clear single-flight before restoring widgets so a re-click
        # while finalize is still pending isn't blocked.
        if getattr(self, "_teacher_proc", None) is proc:
            self._teacher_proc = None

        self.after(0, lambda r=rc, t=tag: self._teacher_finalize(r, t))

    def _teacher_finalize(self, rc: int, tag: str) -> None:
        """Main-thread finalize: restore buttons + auto-fill picker on rc==0."""
        # Reset the progress bar so the next run starts at 0%.
        reset = getattr(self, "_reset_forge_progress", None)
        if reset is not None:
            reset()
        start_btn = getattr(self, "teacher_start_btn", None)
        stop_btn = getattr(self, "teacher_stop_btn", None)
        if start_btn is not None:
            start_btn.configure(state="normal",
                                text="GENERATE EXTERNAL TEACHER CORPUS")
        if stop_btn is not None:
            stop_btn.configure(state="disabled")

        if rc == 0:
            out_path = Path("data/finetune") / f"distill_{tag}.txt"
            if out_path.exists():
                # Auto-fill the FORGE training-data picker so the next
                # train click picks up the new corpus without manual nav.
                var = getattr(self, "train_data_var", None)
                if var is not None:
                    var.set(str(out_path))
                self._log(f"[teacher] [OK] wrote {out_path}")
            else:
                self._log(
                    f"[teacher] [!] exit 0 but output missing: {out_path}")
        else:
            self._log(f"[teacher] [!] exited with code {rc}")

    def _stop_external_teacher_corpus(self) -> None:
        """STOP TEACHER button handler — graceful terminate then kill.
        Also cancels a pending health-check so the spawn callback bails
        when the check completes.
        """
        # Cancel a pending health-check window even if no subprocess
        # has been spawned yet.
        if getattr(self, "_teacher_health_in_flight", False):
            self._teacher_cancel_requested = True
            self._log("[teacher] cancelling pending endpoint check")
        proc = getattr(self, "_teacher_proc", None)
        if proc is None:
            return
        self._log("[teacher] stopping...")
        try:
            proc.terminate()
        except (OSError, ValueError):
            pass
        # Don't block the GUI thread waiting for the kill — the reader
        # thread will pick up the EOF on the pipe and call _teacher_finalize.

    def _kill_teacher_subprocess(self) -> None:
        """Idempotent shutdown hook — called from ``_on_close``.

        Hard-kill if terminate didn't take, so the subprocess doesn't
        outlive the GUI.
        """
        proc = getattr(self, "_teacher_proc", None)
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        except (OSError, ValueError):
            pass
        self._teacher_proc = None

    def _browse_teacher_prompts_file(self) -> None:
        """Open a file picker and write the chosen path into
        ``teacher_prompts_path_var``. Triggered by the BROWSE button
        next to the prompts-file entry."""
        from tkinter import filedialog
        chosen = filedialog.askopenfilename(
            title="Choose teacher prompts file",
            filetypes=[
                ("Text / JSONL", "*.txt *.jsonl"),
                ("Text files", "*.txt"),
                ("JSONL files", "*.jsonl"),
                ("All files", "*.*"),
            ],
        )
        if not chosen:
            return
        var = getattr(self, "teacher_prompts_path_var", None)
        if var is not None:
            var.set(chosen)

    def _suggest_magpie_n_from_tag(self) -> None:
        """Read the current ``teacher_tag_var``, count existing rows in
        ``data/finetune/distill_<tag>.jsonl``, and bump
        ``teacher_magpie_var`` to ``existing + 500``.

        Click handler for the ↻ button next to Magpie-N. Idempotent:
        clicking with a fresh tag (file absent) sets the field to 500.
        """
        tag_var = getattr(self, "teacher_tag_var", None)
        magpie_var = getattr(self, "teacher_magpie_var", None)
        if tag_var is None or magpie_var is None:
            return
        tag = tag_var.get().strip() or "external"
        existing = _count_distill_jsonl_rows(tag)
        new_n = existing + 500
        magpie_var.set(str(new_n))
        if existing > 0:
            self._log(
                f"[teacher] tag '{tag}' has {existing} existing pairs; "
                f"Magpie-N suggested as {new_n}"
            )
        else:
            self._log(
                f"[teacher] tag '{tag}' is fresh; Magpie-N suggested as 500"
            )

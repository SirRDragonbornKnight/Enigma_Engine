"""GUI baseline measurement instrumentation (GUI-ARCH-0b, Phase 0b).

Active only when ``EnigmaGUI`` is constructed with ``baseline=True``
(via ``python run.py --gui --baseline``).  Prints
``[BASELINE] ...`` lines to stdout for the operator to capture into
``information/gui/BASELINE.md``.

Metrics
-------
* **M1** — Cold start: wall-clock from python process start until
  the GUI mainloop is idle and ready to accept input.  One emission
  per process; subsequent ``emit_m1`` calls are no-ops.
* **M2** — Page-switch latency: ms from sidebar button press to
  page-render complete.  One line per switch; operator picks the
  CORE → CONFIG and CORE → FORGE rows from the stdout transcript.
* **M5** — Frame stall: rolling max stall (ms) in the tk after-loop
  while a training step is running.  ``frame_tick`` is called from
  an ``after(16, ...)`` loop scheduled at the end of ``__init__``.

M3 (idle RSS) is captured externally via
``information/gui/measure_baseline.py --m3 --pid <PID>``.
M4 (packaged size) is a static disk-size estimate already filled
into BASELINE.md §3 (19.0 MB, May 13, 2026 — agent).

Design notes
------------
* No tk dependency in this module — pure helper class so it can be
  unit-tested without a live mainloop (see
  ``tests/test_baseline_instrument.py``).
* ``print(..., flush=True)`` so the operator sees the lines even if
  the stdout buffer would otherwise queue them past the next GC.
* No file I/O.  The operator copies stdout into BASELINE.md by hand,
  which keeps this slice strictly opt-in (no risk of stray .md
  writes on a normal GUI launch).
"""

from __future__ import annotations

import time


class BaselineMonitor:
    """Per-process baseline measurement state.

    Created at ``EnigmaGUI.__init__`` entry when ``baseline=True``.
    Single instance per GUI process.
    """

    def __init__(self, process_start: float) -> None:
        self._process_start = process_start
        self._max_stall_ms = 0.0
        self._last_tick = time.perf_counter()
        self._m1_emitted = False

    def emit_m1(self) -> None:
        """Print the M1 cold-start line.  Idempotent — only the first
        call prints, subsequent calls are no-ops.
        """
        if self._m1_emitted:
            return
        elapsed = time.perf_counter() - self._process_start
        print(f"[BASELINE] M1_cold_start_s={elapsed:.3f}", flush=True)
        self._m1_emitted = True

    def time_page_switch(
        self, from_page: str, to_page: str, start: float
    ) -> None:
        """Print the M2 page-switch line for one transition.

        Parameters
        ----------
        from_page, to_page:
            Sidebar page identifiers (e.g. ``"CORE"``, ``"CONFIG"``).
        start:
            ``time.perf_counter()`` captured at the top of
            ``_switch_page`` BEFORE any grid reflow.
        """
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(
            f"[BASELINE] M2_switch from={from_page} to={to_page} "
            f"ms={elapsed_ms:.2f}",
            flush=True,
        )

    def frame_tick(self) -> float:
        """Update the rolling max-stall and return it (ms).

        Called from an ``after(16, ...)`` loop.  A 16 ms interval is
        60 FPS; anything > 50 ms is visible jank.
        """
        now = time.perf_counter()
        dt_ms = (now - self._last_tick) * 1000.0
        if dt_ms > self._max_stall_ms:
            self._max_stall_ms = dt_ms
        self._last_tick = now
        return self._max_stall_ms

    @property
    def max_stall_ms(self) -> float:
        return self._max_stall_ms

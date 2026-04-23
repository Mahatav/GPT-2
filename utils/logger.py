"""
Terminal logger with colour output and an inline training progress bar.

Uses only ANSI escape codes — no external dependencies.
Falls back gracefully if the terminal doesn't support colour.
"""

import sys
import time


# ── ANSI colour codes ────────────────────────────────────────────────────────

class C:
    """Namespace for ANSI colour / style codes."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

    @staticmethod
    def strip(text: str) -> str:
        """Remove all ANSI codes from a string (for plain-text logging)."""
        import re
        return re.sub(r"\033\[[0-9;]*m", "", text)


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# ── Logger class ─────────────────────────────────────────────────────────────

class Logger:
    """
    Structured logger for training runs.

    Methods
    -------
    header(title)           — big coloured banner
    section(title)          — section separator
    info / warn / error     — timestamped log lines
    success(msg)            — green ✓ confirmation
    training_step(...)      — inline-overwriting progress line
    eval_checkpoint(...)    — persistent eval result line
    """

    def __init__(self, name: str = "GPT-2", color: bool = None):
        self.name  = name
        self._color = _supports_color() if color is None else color
        self._t0   = time.time()
        self._last_was_inline = False   # tracks whether last print used \r

    # ── Internal helpers ──────────────────────────────────────────────────

    def _c(self, code: str, text: str) -> str:
        return f"{code}{text}{C.RESET}" if self._color else text

    def _ts(self) -> str:
        return time.strftime("%H:%M:%S")

    def _elapsed(self) -> str:
        s = int(time.time() - self._t0)
        return f"{s//3600:02d}h{(s%3600)//60:02d}m{s%60:02d}s"

    def _newline_if_inline(self) -> None:
        if self._last_was_inline:
            print()
            self._last_was_inline = False

    def _print(self, line: str) -> None:
        self._newline_if_inline()
        print(line, flush=True)

    # ── Structural output ─────────────────────────────────────────────────

    def header(self, title: str) -> None:
        self._newline_if_inline()
        W = 62
        pad = (W - len(title)) // 2
        border = "═" * W
        print()
        print(self._c(C.BOLD + C.CYAN, f"╔{border}╗"))
        print(self._c(C.BOLD + C.CYAN,
              f"║{' ' * pad}{title}{' ' * (W - pad - len(title))}║"))
        print(self._c(C.BOLD + C.CYAN, f"╚{border}╝"))
        print()

    def section(self, title: str) -> None:
        self._newline_if_inline()
        bar = "─" * 64
        print(f"\n{self._c(C.BOLD, bar)}")
        print(f"  {self._c(C.BOLD, title)}")
        print(f"{self._c(C.BOLD, bar)}\n")

    def rule(self) -> None:
        self._newline_if_inline()
        print(self._c(C.DIM, "─" * 64))

    # ── Log levels ────────────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        self._print(
            f"{self._c(C.DIM, '[' + self._ts() + ']')} "
            f"{self._c(C.BLUE, 'INFO ')}  {msg}"
        )

    def warn(self, msg: str) -> None:
        self._print(
            f"{self._c(C.DIM, '[' + self._ts() + ']')} "
            f"{self._c(C.YELLOW, 'WARN ')}  {msg}"
        )

    def error(self, msg: str) -> None:
        self._print(
            f"{self._c(C.DIM, '[' + self._ts() + ']')} "
            f"{self._c(C.RED, 'ERROR')}  {msg}"
        )

    def success(self, msg: str) -> None:
        self._print(
            f"{self._c(C.DIM, '[' + self._ts() + ']')} "
            f"{self._c(C.GREEN, '  ✓  ')}  {msg}"
        )

    def kv(self, label: str, value, width: int = 20) -> None:
        """Print a key: value pair, right-aligning the label."""
        self._print(f"  {self._c(C.DIM, label.rjust(width))} : {self._c(C.BOLD, str(value))}")

    # ── Training-specific output ──────────────────────────────────────────

    @staticmethod
    def _bar(current: int, total: int, width: int = 22) -> str:
        filled = int(width * current / max(total, 1))
        return "█" * filled + "░" * (width - filled)

    def training_step(
        self,
        it:      int,
        total:   int,
        metrics: dict,
        dt_ms:   float = 0.0,   # ms per iteration
    ) -> None:
        """
        Overwrite the current terminal line with a compact progress bar.

        Use eval_checkpoint() at eval intervals to persist a result line.
        """
        pct = 100 * it / max(total, 1)
        bar = self._bar(it, total)

        parts = [
            self._c(C.BOLD, f"iter {it:5d}/{total}"),
            self._c(C.CYAN, f"[{bar}]"),
            f"{pct:5.1f}%",
        ]
        for k, v in metrics.items():
            v_str = f"{v:.2e}" if isinstance(v, float) and v < 0.01 else \
                    f"{v:.4f}" if isinstance(v, float) else str(v)
            parts.append(f"{self._c(C.DIM, k + ':')} {v_str}")

        if dt_ms > 0:
            parts.append(self._c(C.DIM, f"{dt_ms:.0f}ms/it"))

        line = "  " + "  │  ".join(parts)
        print(f"\r{line}", end="", flush=True)
        self._last_was_inline = True

    def eval_checkpoint(
        self,
        it:         int,
        train_loss: float,
        val_loss:   float,
        saved:      bool = False,
    ) -> None:
        """Print a persistent (non-overwritten) eval result line."""
        self._newline_if_inline()
        saved_tag = self._c(C.GREEN, "  [ckpt saved]") if saved else ""
        print(
            f"  {self._c(C.DIM, '─── eval @')} "
            f"{self._c(C.BOLD, f'iter {it:5d}')}  "
            f"train {self._c(C.BOLD, f'{train_loss:.4f}')}  "
            f"val {self._c(C.BOLD + C.CYAN, f'{val_loss:.4f}')}"
            f"{saved_tag}",
            flush=True,
        )

    def generated_text(self, text: str, prompt: str = "") -> None:
        self._newline_if_inline()
        self.rule()
        if prompt:
            print(f"  {self._c(C.DIM, 'Prompt:')} {self._c(C.BOLD, prompt)}")
        print(f"  {self._c(C.DIM, 'Output:')}")
        print()
        # Indent the generated text
        for line in text.splitlines():
            print(f"    {line}")
        print()
        self.rule()

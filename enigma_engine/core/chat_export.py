"""Chat export utilities – HTML and PDF export for conversation history.

Provides functions to convert chat history (list of ``{"role": …, "content": …}``
dicts) into styled HTML documents and optional PDF files.

PDF generation uses **fpdf2** when available, otherwise falls back to
writing an HTML file that the user can print to PDF from a browser.
"""
from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Any, Dict, List

__all__ = [
    "export_html",
    "export_pdf",
    "history_to_html",
]


# ── CSS styles ─────────────────────────────────────────────────────
_CSS = """\
:root {
  --bg: #1e1e2e;
  --fg: #cdd6f4;
  --user-bg: #313244;
  --ai-bg: #1e1e2e;
  --sys-bg: #181825;
  --accent: #89b4fa;
  --user-accent: #a6e3a1;
  --border: #45475a;
  --pre-bg: #11111b;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg); color: var(--fg);
  max-width: 860px; margin: 0 auto; padding: 24px 16px;
  line-height: 1.6;
}
h1 { color: var(--accent); margin-bottom: 4px; font-size: 1.6em; }
.meta { color: #6c7086; font-size: 0.85em; margin-bottom: 24px; }
.message { padding: 14px 18px; border-radius: 10px; margin-bottom: 12px; }
.message.user { background: var(--user-bg); border-left: 3px solid var(--user-accent); }
.message.assistant { background: var(--ai-bg); border-left: 3px solid var(--accent); }
.message.system { background: var(--sys-bg); border-left: 3px solid var(--border); font-style: italic; }
.role { font-weight: 600; margin-bottom: 6px; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.05em; }
.role.user { color: var(--user-accent); }
.role.assistant { color: var(--accent); }
.role.system { color: #6c7086; }
.content { white-space: pre-wrap; word-wrap: break-word; }
.content pre, .content code {
  background: var(--pre-bg); padding: 2px 6px; border-radius: 4px;
  font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 0.9em;
}
.content pre { display: block; padding: 12px; margin: 8px 0; overflow-x: auto; }
.footer { text-align: center; color: #6c7086; font-size: 0.8em; margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--border); }
@media (prefers-color-scheme: light) {
  :root {
    --bg: #ffffff; --fg: #1e1e2e; --user-bg: #f0f4f8;
    --ai-bg: #f8f9ff; --sys-bg: #f4f4f5; --accent: #1e66f5;
    --user-accent: #40a02b; --border: #ccd0da; --pre-bg: #e6e9ef;
  }
}
"""


# ── HTML generation ────────────────────────────────────────────────
def history_to_html(
    history: List[Dict[str, Any]],
    *,
    title: str = "Chat Export",
    ai_name: str = "AI",
    user_name: str = "You",
) -> str:
    """Convert chat history to a full HTML document string.

    Parameters
    ----------
    history:
        List of message dicts with ``role`` and ``content`` keys.
    title:
        Document title.
    ai_name:
        Display name for the assistant.
    user_name:
        Display name for the user.

    Returns
    -------
    str
        Complete HTML document.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    messages_html: list[str] = []

    for msg in history:
        role = msg.get("role", "user")
        content = html.escape(msg.get("content", ""))
        # Convert markdown-style code blocks to <pre>
        content = _format_code_blocks(content)

        if role == "user":
            display_name = user_name
        elif role == "assistant":
            display_name = ai_name
        elif role == "system":
            display_name = "System"
        else:
            display_name = role.title()

        messages_html.append(
            f'<div class="message {html.escape(role)}">'
            f'<div class="role {html.escape(role)}">{html.escape(display_name)}</div>'
            f'<div class="content">{content}</div>'
            f'</div>'
        )

    body = "\n".join(messages_html)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>\n{_CSS}</style>\n"
        "</head>\n<body>\n"
        f"<h1>{html.escape(title)}</h1>\n"
        f'<div class="meta">{html.escape(timestamp)} &bull; '
        f"{len(history)} messages</div>\n"
        f"{body}\n"
        f'<div class="footer">Exported from Enigma Engine</div>\n'
        "</body>\n</html>\n"
    )


def _format_code_blocks(escaped_text: str) -> str:
    """Convert triple-backtick code blocks to ``<pre>`` tags in escaped HTML."""
    import re
    # Pattern for ```...``` blocks (already HTML-escaped)
    parts = re.split(r"```(?:\w*)\n?", escaped_text)
    if len(parts) < 3:
        return escaped_text
    result: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            result.append(f"<pre>{part.strip()}</pre>")
        else:
            result.append(part)
    return "".join(result)


# ── File export ────────────────────────────────────────────────────
def export_html(
    history: List[Dict[str, Any]],
    path: str | Path,
    *,
    title: str = "Chat Export",
    ai_name: str = "AI",
    user_name: str = "You",
) -> Path:
    """Export chat history to an HTML file.

    Returns the resolved output path.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    html_content = history_to_html(
        history, title=title, ai_name=ai_name, user_name=user_name
    )
    out.write_text(html_content, encoding="utf-8")
    return out


def export_pdf(
    history: List[Dict[str, Any]],
    path: str | Path,
    *,
    title: str = "Chat Export",
    ai_name: str = "AI",
    user_name: str = "You",
) -> Path:
    """Export chat history to a PDF file.

    Uses **fpdf2** if available for native PDF generation.
    Falls back to writing an HTML file (same path but ``.html`` suffix)
    and raises a note that the user should print it to PDF.

    Returns the resolved path of the file written.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        return _export_pdf_fpdf2(history, out, title, ai_name, user_name)
    except ImportError:
        pass

    # Fallback – write HTML and inform the caller
    html_path = out.with_suffix(".html")
    export_html(history, html_path, title=title, ai_name=ai_name, user_name=user_name)
    raise ImportError(
        f"fpdf2 is not installed.  An HTML version was saved to "
        f"{html_path.name} instead.  Install fpdf2 for direct PDF export: "
        f"pip install fpdf2"
    )


def _export_pdf_fpdf2(
    history: List[Dict[str, Any]],
    path: Path,
    title: str,
    ai_name: str,
    user_name: str,
) -> Path:
    """Generate a PDF using fpdf2."""
    from fpdf import FPDF  # type: ignore[import-untyped]

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, title, ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(
        0, 6,
        f"Exported {time.strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"{len(history)} messages",
        ln=True,
    )
    pdf.ln(6)

    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            name = user_name
            pdf.set_text_color(64, 160, 43)
        elif role == "assistant":
            name = ai_name
            pdf.set_text_color(30, 102, 245)
        elif role == "system":
            name = "System"
            pdf.set_text_color(120, 120, 120)
        else:
            name = role.title()
            pdf.set_text_color(80, 80, 80)

        # Role header
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, name, ln=True)

        # Content
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "", 10)
        # fpdf2 multi_cell handles long text with wrapping
        pdf.multi_cell(0, 5, content)
        pdf.ln(4)

    pdf.output(str(path))
    return path

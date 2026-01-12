"""
Text Formatting - Allow AI to express emphasis.

Supported formats:
  **bold** - Strong emphasis
  *italic* - Light emphasis
  __underline__ - Important
  ~~strikethrough~~ - Correction
  `code` - Technical terms
  # Header - Big announcement
  > quote - Quoting something
  
  UPPERCASE - Shouting/excitement
  lowercase whisper - Quiet/shy
"""

import re


class TextFormatter:
    """Parse and apply text formatting."""
    
    @staticmethod
    def to_html(text: str) -> str:
        """Convert markdown-style formatting to HTML for GUI display."""
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        # Italic
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        # Underline
        text = re.sub(r'__(.+?)__', r'<u>\1</u>', text)
        # Strikethrough
        text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
        # Code
        text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
        # Headers (big text)
        text = re.sub(r'^# (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        # Quote
        text = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
        return text
    
    @staticmethod
    def strip_formatting(text: str) -> str:
        """Remove all formatting for plain text output."""
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'~~(.+?)~~', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'^#+ ', '', text, flags=re.MULTILINE)
        text = re.sub(r'^> ', '', text, flags=re.MULTILINE)
        return text

"""
Text Formatting - Allow AI to express emphasis.

Supported formats:
  **bold** - Strong emphasis
  *italic* - Light emphasis
  __underline__ - Important
  ~~strikethrough~~ - Correction
  `code` - Technical terms
  ```language - Code blocks with syntax highlighting
  # Header - Big announcement
  > quote - Quoting something
  - bullet - List items
  1. numbered - Numbered lists
  [text](url) - Links
  | table | - Tables
  
  UPPERCASE - Shouting/excitement
  lowercase whisper - Quiet/shy
"""

import hashlib
import html
import re

# Code block counter for unique IDs
_code_block_counter = 0


class TextFormatter:
    """Parse and apply text formatting."""
    
    @staticmethod
    def to_html(text: str) -> str:
        """Convert markdown-style formatting to HTML for GUI display."""
        global _code_block_counter
        
        # First handle multi-line code blocks (before other formatting)
        def replace_code_block(match):
            global _code_block_counter
            _code_block_counter += 1
            language = match.group(1) or "text"
            code = match.group(2)
            # Escape HTML in code
            escaped_code = html.escape(code.strip())
            # Create unique ID for this code block
            block_id = f"code_block_{_code_block_counter}"
            # Store code for copy functionality (base64 would be safer but this works)
            code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
            
            # Color mapping for languages
            lang_colors = {
                'python': '#3776ab',
                'javascript': '#f7df1e',
                'typescript': '#3178c6',
                'java': '#b07219',
                'c': '#555555',
                'cpp': '#f34b7d',
                'c++': '#f34b7d',
                'csharp': '#178600',
                'c#': '#178600',
                'rust': '#dea584',
                'go': '#00add8',
                'ruby': '#701516',
                'php': '#4f5d95',
                'html': '#e34c26',
                'css': '#563d7c',
                'sql': '#e38c00',
                'bash': '#4eaa25',
                'shell': '#4eaa25',
                'json': '#292929',
                'yaml': '#cb171e',
                'xml': '#0060ac',
                'markdown': '#083fa1',
                'dockerfile': '#384d54',
                'makefile': '#427819',
                'lua': '#000080',
                'perl': '#0298c3',
                'r': '#198ce7',
                'scala': '#c22d40',
                'swift': '#f05138',
                'kotlin': '#a97bff',
            }
            lang_color = lang_colors.get(language.lower(), '#6c7086')
            
            return (
                f'<div style="position: relative; margin: 8px 0; border: 1px solid #45475a; '
                f'border-radius: 4px; overflow: hidden;">'
                f'<div style="background: #313244; padding: 4px 8px; color: {lang_color}; '
                f'font-size: 11px; border-bottom: 1px solid #45475a;">'
                f'{language}'
                f'<a href="copy:{code_hash}" style="float: right; color: #89b4fa; '
                f'text-decoration: none; font-size: 10px;" title="Copy code">Copy</a>'
                f'</div>'
                f'<pre style="background: #1e1e2e; margin: 0; padding: 10px; overflow-x: auto; '
                f'font-family: monospace; font-size: 12px; color: #cdd6f4; white-space: pre-wrap;">'
                f'<code id="{block_id}" data-code="{code_hash}">{escaped_code}</code></pre>'
                f'</div>'
            )
        
        # Replace ```language\ncode\n``` blocks
        text = re.sub(
            r'```(\w*)\n(.*?)```',
            replace_code_block,
            text,
            flags=re.DOTALL
        )
        
        # Handle tables (simple markdown tables)
        def replace_table(match):
            lines = match.group(0).strip().split('\n')
            if len(lines) < 2:
                return match.group(0)
            
            # Parse header
            header_cells = [c.strip() for c in lines[0].split('|')[1:-1]]
            
            # Skip separator line (line with dashes)
            data_lines = [l for l in lines[2:] if l.strip() and not re.match(r'^[\|\-\s:]+$', l)]
            
            html_table = (
                '<table style="border-collapse: collapse; margin: 8px 0; width: 100%;">'
                '<thead><tr>'
            )
            for cell in header_cells:
                html_table += f'<th style="border: 1px solid #45475a; padding: 6px 10px; background: #313244; color: #cba6f7;">{cell}</th>'
            html_table += '</tr></thead><tbody>'
            
            for line in data_lines:
                cells = [c.strip() for c in line.split('|')[1:-1]]
                html_table += '<tr>'
                for cell in cells:
                    html_table += f'<td style="border: 1px solid #45475a; padding: 6px 10px;">{cell}</td>'
                html_table += '</tr>'
            
            html_table += '</tbody></table>'
            return html_table
        
        # Match markdown tables
        text = re.sub(
            r'^\|.+\|\n\|[\-\s\|:]+\|\n(?:\|.+\|\n?)+',
            replace_table,
            text,
            flags=re.MULTILINE
        )
        
        # Links [text](url)
        text = re.sub(
            r'\[([^\]]+)\]\(([^)]+)\)',
            r'<a href="\2" style="color: #89b4fa; text-decoration: underline;">\1</a>',
            text
        )
        
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        # Italic
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        # Underline
        text = re.sub(r'__(.+?)__', r'<u>\1</u>', text)
        # Strikethrough
        text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
        # Inline code (single backticks, but not inside code blocks)
        text = re.sub(r'`([^`]+)`', r'<code style="background: #313244; padding: 2px 4px; border-radius: 3px;">\1</code>', text)
        # Headers (big text)
        text = re.sub(r'^# (.+)$', r'<h2 style="color: #cba6f7; margin: 8px 0;">\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'<h3 style="color: #89b4fa; margin: 6px 0;">\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^### (.+)$', r'<h4 style="color: #a6e3a1; margin: 4px 0;">\1</h4>', text, flags=re.MULTILINE)
        # Quote
        text = re.sub(
            r'^> (.+)$',
            r'<blockquote style="border-left: 3px solid #89b4fa; padding-left: 10px; color: #a6adc8; margin: 8px 0;">\1</blockquote>',
            text,
            flags=re.MULTILINE
        )
        # Horizontal rule
        text = re.sub(r'^---+$', r'<hr style="border: none; border-top: 1px solid #45475a; margin: 16px 0;">', text, flags=re.MULTILINE)
        
        # Bullet lists (- or * at start of line)
        def replace_bullet_list(match):
            items = match.group(0).strip().split('\n')
            html_list = '<ul style="margin: 8px 0; padding-left: 20px;">'
            for item in items:
                content = re.sub(r'^[\-\*]\s+', '', item)
                html_list += f'<li style="margin: 2px 0;">{content}</li>'
            html_list += '</ul>'
            return html_list
        
        text = re.sub(
            r'(?:^[\-\*]\s+.+$\n?)+',
            replace_bullet_list,
            text,
            flags=re.MULTILINE
        )
        
        # Numbered lists (1. 2. 3. at start of line)
        def replace_numbered_list(match):
            items = match.group(0).strip().split('\n')
            html_list = '<ol style="margin: 8px 0; padding-left: 20px;">'
            for item in items:
                content = re.sub(r'^\d+\.\s+', '', item)
                html_list += f'<li style="margin: 2px 0;">{content}</li>'
            html_list += '</ol>'
            return html_list
        
        text = re.sub(
            r'(?:^\d+\.\s+.+$\n?)+',
            replace_numbered_list,
            text,
            flags=re.MULTILINE
        )
        
        # Task lists (- [ ] or - [x])
        text = re.sub(
            r'^- \[x\] (.+)$',
            r'<div style="margin: 2px 0;"><span style="color: #a6e3a1;">&#9745;</span> <s style="color: #6c7086;">\1</s></div>',
            text,
            flags=re.MULTILINE
        )
        text = re.sub(
            r'^- \[ \] (.+)$',
            r'<div style="margin: 2px 0;"><span style="color: #6c7086;">&#9744;</span> \1</div>',
            text,
            flags=re.MULTILINE
        )
        
        # Convert newlines to <br> for proper display
        text = text.replace('\n', '<br>')
        
        return text
    
    @staticmethod
    def to_html_with_code_storage(text: str) -> tuple:
        """
        Convert markdown to HTML and return code blocks separately.
        
        Returns:
            tuple: (html_string, dict of code_hash -> code_content)
        """
        code_blocks = {}
        
        def store_code_block(match):
            language = match.group(1) or "text"
            code = match.group(2).strip()
            code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
            code_blocks[code_hash] = code
            # Return placeholder that will be formatted
            return f"```{language}\n{code}\n```"
        
        # First pass: extract and store code
        text = re.sub(
            r'```(\w*)\n(.*?)```',
            store_code_block,
            text,
            flags=re.DOTALL
        )
        
        # Second pass: format
        formatted = TextFormatter.to_html(text)
        
        return formatted, code_blocks
    
    @staticmethod
    def strip_formatting(text: str) -> str:
        """Remove all formatting for plain text output."""
        # Remove code blocks but keep content
        text = re.sub(r'```\w*\n(.*?)```', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'~~(.+?)~~', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'^#+ ', '', text, flags=re.MULTILINE)
        text = re.sub(r'^> ', '', text, flags=re.MULTILINE)
        text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # Remove list markers
        text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        # Remove task list markers
        text = re.sub(r'^- \[[x ]\] ', '', text, flags=re.MULTILINE)
        # Remove table formatting (keep content)
        text = re.sub(r'\|', ' ', text)
        text = re.sub(r'^[\-\s:]+$', '', text, flags=re.MULTILINE)
        return text
    
    @staticmethod
    def extract_code_blocks(text: str) -> list:
        """
        Extract all code blocks from text.
        
        Returns:
            List of dicts with 'language' and 'code' keys
        """
        blocks = []
        for match in re.finditer(r'```(\w*)\n(.*?)```', text, flags=re.DOTALL):
            blocks.append({
                'language': match.group(1) or 'text',
                'code': match.group(2).strip()
            })
        return blocks

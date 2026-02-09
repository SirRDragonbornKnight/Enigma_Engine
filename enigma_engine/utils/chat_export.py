"""
Chat Export System for Enigma AI Engine

Export conversations in various formats.

Usage:
    from enigma_engine.utils.chat_export import export_chat, ChatExporter
    
    # Export current conversation
    export_chat(
        messages=[...],
        output_path="chat.md",
        format="markdown"
    )
    
    # Or use the exporter class
    exporter = ChatExporter()
    exporter.export_markdown(messages, "chat.md")
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[str] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatExporter:
    """
    Export chat conversations to various formats.
    
    Supported formats:
    - Markdown (.md)
    - JSON (.json)
    - Plain text (.txt)
    - HTML (.html)
    """
    
    def __init__(self, model_name: str = "Enigma AI"):
        self.model_name = model_name
    
    def export(
        self,
        messages: List[ChatMessage],
        output_path: Path,
        format: str = "markdown",
        title: Optional[str] = None
    ) -> bool:
        """
        Export messages to specified format.
        
        Args:
            messages: List of chat messages
            output_path: Path to save the export
            format: Export format (markdown, json, txt, html)
            title: Optional title for the export
            
        Returns:
            True if successful
        """
        output_path = Path(output_path)
        
        try:
            if format == "markdown" or format == "md":
                return self.export_markdown(messages, output_path, title)
            elif format == "json":
                return self.export_json(messages, output_path, title)
            elif format == "txt" or format == "text":
                return self.export_text(messages, output_path, title)
            elif format == "html":
                return self.export_html(messages, output_path, title)
            else:
                logger.error(f"Unknown export format: {format}")
                return False
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def export_markdown(
        self,
        messages: List[ChatMessage],
        output_path: Path,
        title: Optional[str] = None
    ) -> bool:
        """Export to Markdown format."""
        lines = []
        
        # Header
        title = title or f"Chat with {self.model_name}"
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Messages
        for msg in messages:
            # Role header
            if msg.role == "user":
                lines.append("## User")
            elif msg.role == "assistant":
                lines.append(f"## {self.model_name}")
            elif msg.role == "system":
                lines.append("## System")
            else:
                lines.append(f"## {msg.role.title()}")
            
            # Timestamp if available
            if msg.timestamp:
                lines.append(f"*{msg.timestamp}*")
            
            lines.append("")
            
            # Content
            lines.append(msg.content)
            lines.append("")
            lines.append("---")
            lines.append("")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Exported {len(messages)} messages to {output_path}")
        return True
    
    def export_json(
        self,
        messages: List[ChatMessage],
        output_path: Path,
        title: Optional[str] = None
    ) -> bool:
        """Export to JSON format."""
        data = {
            "title": title or f"Chat with {self.model_name}",
            "model": self.model_name,
            "exported_at": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "model": msg.model,
                    "metadata": msg.metadata
                }
                for msg in messages
            ]
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(messages)} messages to {output_path}")
        return True
    
    def export_text(
        self,
        messages: List[ChatMessage],
        output_path: Path,
        title: Optional[str] = None
    ) -> bool:
        """Export to plain text format."""
        lines = []
        
        title = title or f"Chat with {self.model_name}"
        lines.append(title)
        lines.append("=" * len(title))
        lines.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        for msg in messages:
            role = msg.role.upper()
            lines.append(f"[{role}]")
            lines.append(msg.content)
            lines.append("")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Exported {len(messages)} messages to {output_path}")
        return True
    
    def export_html(
        self,
        messages: List[ChatMessage],
        output_path: Path,
        title: Optional[str] = None
    ) -> bool:
        """Export to HTML format with styling."""
        title = title or f"Chat with {self.model_name}"
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg: #1e1e2e;
            --surface: #2a2a3a;
            --text: #cdd6f4;
            --user-bg: #313244;
            --assistant-bg: #45475a;
            --system-bg: #585b70;
        }}
        
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        
        h1 {{
            border-bottom: 2px solid var(--surface);
            padding-bottom: 10px;
        }}
        
        .meta {{
            color: #6c7086;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        
        .message {{
            margin: 15px 0;
            padding: 15px;
            border-radius: 10px;
        }}
        
        .user {{
            background-color: var(--user-bg);
            margin-left: 40px;
        }}
        
        .assistant {{
            background-color: var(--assistant-bg);
            margin-right: 40px;
        }}
        
        .system {{
            background-color: var(--system-bg);
            font-style: italic;
            text-align: center;
        }}
        
        .role {{
            font-weight: bold;
            font-size: 0.85em;
            text-transform: uppercase;
            color: #89b4fa;
            margin-bottom: 5px;
        }}
        
        .user .role {{
            color: #a6e3a1;
        }}
        
        .content {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        .timestamp {{
            font-size: 0.8em;
            color: #6c7086;
            margin-top: 5px;
        }}
        
        code {{
            background-color: var(--bg);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Consolas', monospace;
        }}
        
        pre {{
            background-color: var(--bg);
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }}
        
        @media (prefers-color-scheme: light) {{
            :root {{
                --bg: #eff1f5;
                --surface: #dce0e8;
                --text: #4c4f69;
                --user-bg: #ccd0da;
                --assistant-bg: #bcc0cc;
                --system-bg: #acb0be;
            }}
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="meta">Exported on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
"""
        
        for msg in messages:
            role_class = msg.role
            role_name = msg.role.upper()
            if msg.role == "assistant":
                role_name = self.model_name.upper()
            
            content = self._escape_html(msg.content)
            
            html += f"""
    <div class="message {role_class}">
        <div class="role">{role_name}</div>
        <div class="content">{content}</div>
"""
            if msg.timestamp:
                html += f'        <div class="timestamp">{msg.timestamp}</div>\n'
            
            html += "    </div>\n"
        
        html += """
</body>
</html>
"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Exported {len(messages)} messages to {output_path}")
        return True
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace('\n', '<br>\n'))


def export_chat(
    messages: List[Dict[str, Any]],
    output_path: str,
    format: str = "markdown",
    title: Optional[str] = None,
    model_name: str = "Enigma AI"
) -> bool:
    """
    Convenience function to export chat messages.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        output_path: Where to save the export
        format: Export format (markdown, json, txt, html)
        title: Optional title for the export
        model_name: Name of the AI model
        
    Returns:
        True if successful
    """
    # Convert dicts to ChatMessage objects
    chat_messages = [
        ChatMessage(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            timestamp=msg.get("timestamp"),
            model=msg.get("model"),
            metadata=msg.get("metadata")
        )
        for msg in messages
    ]
    
    exporter = ChatExporter(model_name)
    return exporter.export(
        chat_messages,
        Path(output_path),
        format=format,
        title=title
    )


def get_export_formats() -> List[Dict[str, str]]:
    """Get list of supported export formats."""
    return [
        {"id": "markdown", "name": "Markdown", "extension": ".md"},
        {"id": "json", "name": "JSON", "extension": ".json"},
        {"id": "txt", "name": "Plain Text", "extension": ".txt"},
        {"id": "html", "name": "HTML", "extension": ".html"},
    ]

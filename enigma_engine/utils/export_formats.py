"""
================================================================================
Export Formats - Export data in multiple formats.
================================================================================

Export features:
- Multiple formats: JSON, CSV, TSV, Markdown, HTML, YAML
- Schema detection and validation
- Streaming export for large datasets
- Custom formatters
- Import/export roundtrip

USAGE:
    from enigma_engine.utils.export_formats import Exporter, get_exporter
    
    exporter = get_exporter()
    
    # Export data
    data = [{"name": "test", "value": 123}]
    exporter.export(data, "output.json", format="json")
    exporter.export(data, "output.csv", format="csv")
    exporter.export(data, "output.md", format="markdown")
    
    # Import data
    imported = exporter.import_file("data.csv")
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    TSV = "tsv"
    MARKDOWN = "markdown"
    HTML = "html"
    YAML = "yaml"
    JSONL = "jsonl"
    XML = "xml"
    TEXT = "text"


class FormatHandler(ABC):
    """Abstract base for format handlers."""
    
    @abstractmethod
    def export(
        self,
        data: Any,
        stream: io.IOBase,
        **options
    ) -> None:
        """Export data to stream."""
    
    @abstractmethod
    def import_data(
        self,
        stream: io.IOBase,
        **options
    ) -> Any:
        """Import data from stream."""


class JSONHandler(FormatHandler):
    """JSON format handler."""
    
    def export(
        self,
        data: Any,
        stream: io.IOBase,
        indent: int = 2,
        ensure_ascii: bool = False,
        **options
    ) -> None:
        if is_dataclass(data) and not isinstance(data, type):
            data = asdict(data)
        elif isinstance(data, list) and data and is_dataclass(data[0]):
            data = [asdict(d) for d in data]
        
        json.dump(data, stream, indent=indent, ensure_ascii=ensure_ascii, default=str)
    
    def import_data(self, stream: io.IOBase, **options) -> Any:
        return json.load(stream)


class JSONLHandler(FormatHandler):
    """JSON Lines format handler."""
    
    def export(
        self,
        data: Any,
        stream: io.IOBase,
        **options
    ) -> None:
        if not isinstance(data, list):
            data = [data]
        
        for item in data:
            if is_dataclass(item) and not isinstance(item, type):
                item = asdict(item)
            json.dump(item, stream, ensure_ascii=False, default=str)
            stream.write('\n')
    
    def import_data(self, stream: io.IOBase, **options) -> list[Any]:
        result = []
        for line in stream:
            line = line.strip()
            if line:
                result.append(json.loads(line))
        return result


class CSVHandler(FormatHandler):
    """CSV format handler."""
    
    def export(
        self,
        data: Any,
        stream: io.IOBase,
        delimiter: str = ',',
        headers: bool = True,
        **options
    ) -> None:
        if not data:
            return
        
        # Normalize to list of dicts
        if is_dataclass(data) and not isinstance(data, type):
            data = [asdict(data)]
        elif not isinstance(data, list):
            data = [data]
        elif data and is_dataclass(data[0]):
            data = [asdict(d) for d in data]
        
        # Get all keys
        all_keys = []
        for item in data:
            if isinstance(item, dict):
                for key in item.keys():
                    if key not in all_keys:
                        all_keys.append(key)
        
        writer = csv.DictWriter(
            stream,
            fieldnames=all_keys,
            delimiter=delimiter,
            extrasaction='ignore'
        )
        
        if headers:
            writer.writeheader()
        
        for row in data:
            if isinstance(row, dict):
                # Convert non-string values
                clean_row = {}
                for k, v in row.items():
                    if isinstance(v, (list, dict)):
                        clean_row[k] = json.dumps(v)
                    else:
                        clean_row[k] = v
                writer.writerow(clean_row)
    
    def import_data(
        self,
        stream: io.IOBase,
        delimiter: str = ',',
        **options
    ) -> list[dict[str, Any]]:
        reader = csv.DictReader(stream, delimiter=delimiter)
        result = []
        for row in reader:
            # Try to parse JSON values
            clean_row = {}
            for k, v in row.items():
                if v and v.startswith(('[', '{')):
                    try:
                        clean_row[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        clean_row[k] = v
                else:
                    clean_row[k] = v
            result.append(clean_row)
        return result


class TSVHandler(CSVHandler):
    """TSV format handler (extends CSV with tab delimiter)."""
    
    def export(self, data: Any, stream: io.IOBase, **options) -> None:
        super().export(data, stream, delimiter='\t', **options)
    
    def import_data(self, stream: io.IOBase, **options) -> list[dict[str, Any]]:
        return super().import_data(stream, delimiter='\t', **options)


class MarkdownHandler(FormatHandler):
    """Markdown format handler."""
    
    def export(
        self,
        data: Any,
        stream: io.IOBase,
        title: str = "Data Export",
        table_style: bool = True,
        **options
    ) -> None:
        if not data:
            stream.write("*No data*\n")
            return
        
        # Normalize
        if is_dataclass(data) and not isinstance(data, type):
            data = [asdict(data)]
        elif not isinstance(data, list):
            data = [data]
        elif data and is_dataclass(data[0]):
            data = [asdict(d) for d in data]
        
        stream.write(f"# {title}\n\n")
        stream.write(f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        if isinstance(data[0], dict) and table_style:
            # Export as table
            keys = list(data[0].keys())
            
            # Header
            stream.write("| " + " | ".join(str(k) for k in keys) + " |\n")
            stream.write("| " + " | ".join("---" for _ in keys) + " |\n")
            
            # Rows
            for row in data:
                values = []
                for k in keys:
                    v = row.get(k, "")
                    if isinstance(v, (list, dict)):
                        v = json.dumps(v)
                    v = str(v).replace("|", "\\|").replace("\n", "<br>")
                    values.append(v[:50] + "..." if len(str(v)) > 50 else v)
                stream.write("| " + " | ".join(values) + " |\n")
        else:
            # Export as list
            for i, item in enumerate(data, 1):
                stream.write(f"## Item {i}\n\n")
                if isinstance(item, dict):
                    for k, v in item.items():
                        stream.write(f"- **{k}:** {v}\n")
                else:
                    stream.write(f"{item}\n")
                stream.write("\n")
    
    def import_data(self, stream: io.IOBase, **options) -> list[dict[str, Any]]:
        content = stream.read()
        
        # Try to parse markdown table
        table_match = re.search(r'\|(.+)\|\n\|[-\s|]+\|\n((?:\|.+\|\n?)+)', content)
        if table_match:
            headers = [h.strip() for h in table_match.group(1).split("|") if h.strip()]
            rows_text = table_match.group(2).strip().split("\n")
            
            result = []
            for row_text in rows_text:
                values = [v.strip() for v in row_text.split("|") if v.strip() or row_text.strip().startswith("|")]
                # Remove empty first/last elements from |col1|col2| format
                if values and not values[0]:
                    values = values[1:]
                if values and not values[-1]:
                    values = values[:-1]
                
                if len(values) == len(headers):
                    result.append(dict(zip(headers, values)))
            
            return result
        
        return [{"content": content}]


class HTMLHandler(FormatHandler):
    """HTML format handler."""
    
    def export(
        self,
        data: Any,
        stream: io.IOBase,
        title: str = "Data Export",
        **options
    ) -> None:
        if not data:
            stream.write("<html><body><p>No data</p></body></html>")
            return
        
        # Normalize
        if is_dataclass(data) and not isinstance(data, type):
            data = [asdict(data)]
        elif not isinstance(data, list):
            data = [data]
        elif data and is_dataclass(data[0]):
            data = [asdict(d) for d in data]
        
        stream.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #ddd; }}
    </style>
</head>
<body>
<h1>{title}</h1>
<p><em>Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
""")
        
        if isinstance(data[0], dict):
            keys = list(data[0].keys())
            
            stream.write("<table>\n<thead><tr>")
            for k in keys:
                stream.write(f"<th>{self._escape_html(str(k))}</th>")
            stream.write("</tr></thead>\n<tbody>\n")
            
            for row in data:
                stream.write("<tr>")
                for k in keys:
                    v = row.get(k, "")
                    if isinstance(v, (list, dict)):
                        v = json.dumps(v)
                    stream.write(f"<td>{self._escape_html(str(v))}</td>")
                stream.write("</tr>\n")
            
            stream.write("</tbody></table>\n")
        else:
            stream.write("<ul>\n")
            for item in data:
                stream.write(f"<li>{self._escape_html(str(item))}</li>\n")
            stream.write("</ul>\n")
        
        stream.write("</body></html>")
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML entities."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
    
    def import_data(self, stream: io.IOBase, **options) -> list[dict[str, Any]]:
        content = stream.read()
        
        # Try to parse HTML table
        table_match = re.search(r'<table[^>]*>(.*?)</table>', content, re.DOTALL | re.IGNORECASE)
        if table_match:
            table_content = table_match.group(1)
            
            # Extract headers
            header_match = re.search(r'<thead[^>]*>(.*?)</thead>', table_content, re.DOTALL | re.IGNORECASE)
            if header_match:
                headers = re.findall(r'<th[^>]*>(.*?)</th>', header_match.group(1), re.DOTALL | re.IGNORECASE)
            else:
                # Try first row as header
                first_row = re.search(r'<tr[^>]*>(.*?)</tr>', table_content, re.DOTALL | re.IGNORECASE)
                if first_row:
                    headers = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', first_row.group(1), re.DOTALL | re.IGNORECASE)
                else:
                    headers = []
            
            # Clean headers
            headers = [re.sub(r'<[^>]+>', '', h).strip() for h in headers]
            
            # Extract body rows
            body_match = re.search(r'<tbody[^>]*>(.*?)</tbody>', table_content, re.DOTALL | re.IGNORECASE)
            if body_match:
                rows_content = body_match.group(1)
            else:
                rows_content = table_content
            
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', rows_content, re.DOTALL | re.IGNORECASE)
            
            result = []
            for row in rows:
                cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
                cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
                
                if len(cells) == len(headers):
                    result.append(dict(zip(headers, cells)))
            
            return result if result else [{"content": content}]
        
        return [{"content": content}]


class YAMLHandler(FormatHandler):
    """YAML format handler."""
    
    def export(
        self,
        data: Any,
        stream: io.IOBase,
        **options
    ) -> None:
        try:
            import yaml
            
            if is_dataclass(data) and not isinstance(data, type):
                data = asdict(data)
            elif isinstance(data, list) and data and is_dataclass(data[0]):
                data = [asdict(d) for d in data]
            
            yaml.dump(data, stream, default_flow_style=False, allow_unicode=True)
        except ImportError:
            # Fallback to simple YAML-like format
            self._simple_yaml_export(data, stream)
    
    def _simple_yaml_export(self, data: Any, stream: io.IOBase, indent: int = 0) -> None:
        """Simple YAML export without pyyaml."""
        prefix = "  " * indent
        
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    stream.write(f"{prefix}{k}:\n")
                    self._simple_yaml_export(v, stream, indent + 1)
                else:
                    stream.write(f"{prefix}{k}: {v}\n")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    stream.write(f"{prefix}-\n")
                    self._simple_yaml_export(item, stream, indent + 1)
                else:
                    stream.write(f"{prefix}- {item}\n")
        else:
            stream.write(f"{prefix}{data}\n")
    
    def import_data(self, stream: io.IOBase, **options) -> Any:
        try:
            import yaml
            return yaml.safe_load(stream)
        except ImportError:
            # Fallback to JSON if YAML not available
            content = stream.read()
            try:
                return json.loads(content)
            except (json.JSONDecodeError, ValueError):
                return {"content": content}


class XMLHandler(FormatHandler):
    """XML format handler."""
    
    def export(
        self,
        data: Any,
        stream: io.IOBase,
        root_name: str = "data",
        item_name: str = "item",
        **options
    ) -> None:
        stream.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        stream.write(f'<{root_name}>\n')
        
        if is_dataclass(data) and not isinstance(data, type):
            data = [asdict(data)]
        elif not isinstance(data, list):
            data = [data]
        elif data and is_dataclass(data[0]):
            data = [asdict(d) for d in data]
        
        for item in data:
            self._write_xml_element(stream, item_name, item, indent=1)
        
        stream.write(f'</{root_name}>\n')
    
    def _write_xml_element(
        self,
        stream: io.IOBase,
        name: str,
        value: Any,
        indent: int = 0
    ) -> None:
        prefix = "  " * indent
        name = re.sub(r'[^\w]', '_', str(name))  # Sanitize element name
        
        if isinstance(value, dict):
            stream.write(f'{prefix}<{name}>\n')
            for k, v in value.items():
                self._write_xml_element(stream, k, v, indent + 1)
            stream.write(f'{prefix}</{name}>\n')
        elif isinstance(value, list):
            stream.write(f'{prefix}<{name}>\n')
            for item in value:
                self._write_xml_element(stream, "item", item, indent + 1)
            stream.write(f'{prefix}</{name}>\n')
        else:
            escaped = self._escape_xml(str(value))
            stream.write(f'{prefix}<{name}>{escaped}</{name}>\n')
    
    def _escape_xml(self, text: str) -> str:
        """Escape XML entities."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )
    
    def import_data(self, stream: io.IOBase, **options) -> Any:
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(stream)
            root = tree.getroot()
            return self._xml_to_dict(root)
        except Exception as e:
            logger.error(f"XML import failed: {e}")
            return {}
    
    def _xml_to_dict(self, element) -> Any:
        """Convert XML element to dict."""
        if len(element) == 0:
            return element.text or ""
        
        result = {}
        for child in element:
            child_data = self._xml_to_dict(child)
            
            if child.tag in result:
                # Convert to list if duplicate
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result


class TextHandler(FormatHandler):
    """Plain text format handler."""
    
    def export(
        self,
        data: Any,
        stream: io.IOBase,
        separator: str = "\n",
        **options
    ) -> None:
        if is_dataclass(data) and not isinstance(data, type):
            data = asdict(data)
        
        if isinstance(data, dict):
            for k, v in data.items():
                stream.write(f"{k}: {v}{separator}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for k, v in item.items():
                        stream.write(f"{k}: {v}")
                        stream.write(separator)
                    stream.write(separator)
                else:
                    stream.write(str(item) + separator)
        else:
            stream.write(str(data))
    
    def import_data(self, stream: io.IOBase, **options) -> Any:
        content = stream.read()
        
        # Try to parse key: value format
        lines = content.strip().split("\n")
        if all(":" in line for line in lines if line.strip()):
            result = {}
            for line in lines:
                if ":" in line:
                    k, v = line.split(":", 1)
                    result[k.strip()] = v.strip()
            return result
        
        return {"content": content}


class Exporter:
    """
    Main exporter class supporting multiple formats.
    """
    
    def __init__(self):
        """Initialize exporter with format handlers."""
        self._handlers: dict[ExportFormat, FormatHandler] = {
            ExportFormat.JSON: JSONHandler(),
            ExportFormat.JSONL: JSONLHandler(),
            ExportFormat.CSV: CSVHandler(),
            ExportFormat.TSV: TSVHandler(),
            ExportFormat.MARKDOWN: MarkdownHandler(),
            ExportFormat.HTML: HTMLHandler(),
            ExportFormat.YAML: YAMLHandler(),
            ExportFormat.XML: XMLHandler(),
            ExportFormat.TEXT: TextHandler(),
        }
    
    def get_supported_formats(self) -> list[ExportFormat]:
        """Get list of supported formats."""
        return list(self._handlers.keys())
    
    def register_handler(
        self,
        format_type: ExportFormat,
        handler: FormatHandler
    ) -> None:
        """Register a custom format handler."""
        self._handlers[format_type] = handler
    
    def export(
        self,
        data: Any,
        output: str | Path | io.IOBase,
        format: str | ExportFormat | None = None,
        **options
    ) -> None:
        """
        Export data to file or stream.
        
        Args:
            data: Data to export
            output: Output path or stream
            format: Export format (auto-detected from extension if not provided)
            **options: Format-specific options
        """
        # Determine format
        if format is None and isinstance(output, (str, Path)):
            format = self._detect_format(Path(output))
        elif format is None:
            format = ExportFormat.JSON
        
        if isinstance(format, str):
            format = ExportFormat(format.lower())
        
        handler = self._handlers.get(format)
        if not handler:
            raise ValueError(f"Unsupported format: {format}")
        
        # Open file if needed
        if isinstance(output, (str, Path)):
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8', newline='') as f:
                handler.export(data, f, **options)
            logger.info(f"Exported to {path}")
        else:
            handler.export(data, output, **options)
    
    def export_string(
        self,
        data: Any,
        format: str | ExportFormat,
        **options
    ) -> str:
        """
        Export data to string.
        
        Args:
            data: Data to export
            format: Export format
            **options: Format-specific options
            
        Returns:
            Exported string
        """
        buffer = io.StringIO()
        self.export(data, buffer, format, **options)
        return buffer.getvalue()
    
    def import_file(
        self,
        input_path: str | Path,
        format: str | ExportFormat | None = None,
        **options
    ) -> Any:
        """
        Import data from file.
        
        Args:
            input_path: Input file path
            format: Format (auto-detected if not provided)
            **options: Format-specific options
            
        Returns:
            Imported data
        """
        path = Path(input_path)
        
        if format is None:
            format = self._detect_format(path)
        
        if isinstance(format, str):
            format = ExportFormat(format.lower())
        
        handler = self._handlers.get(format)
        if not handler:
            raise ValueError(f"Unsupported format: {format}")
        
        with open(path, encoding='utf-8') as f:
            return handler.import_data(f, **options)
    
    def import_string(
        self,
        content: str,
        format: str | ExportFormat,
        **options
    ) -> Any:
        """
        Import data from string.
        
        Args:
            content: String content to import
            format: Format of the content
            **options: Format-specific options
            
        Returns:
            Imported data
        """
        if isinstance(format, str):
            format = ExportFormat(format.lower())
        
        handler = self._handlers.get(format)
        if not handler:
            raise ValueError(f"Unsupported format: {format}")
        
        buffer = io.StringIO(content)
        return handler.import_data(buffer, **options)
    
    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        input_format: str | ExportFormat | None = None,
        output_format: str | ExportFormat | None = None,
        **options
    ) -> None:
        """
        Convert between formats.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            input_format: Input format (auto-detected if not provided)
            output_format: Output format (auto-detected if not provided)
            **options: Format-specific options
        """
        data = self.import_file(input_path, input_format)
        self.export(data, output_path, output_format, **options)
    
    def _detect_format(self, path: Path) -> ExportFormat:
        """Detect format from file extension."""
        ext = path.suffix.lower().lstrip('.')
        
        format_map = {
            'json': ExportFormat.JSON,
            'jsonl': ExportFormat.JSONL,
            'csv': ExportFormat.CSV,
            'tsv': ExportFormat.TSV,
            'md': ExportFormat.MARKDOWN,
            'markdown': ExportFormat.MARKDOWN,
            'html': ExportFormat.HTML,
            'htm': ExportFormat.HTML,
            'yaml': ExportFormat.YAML,
            'yml': ExportFormat.YAML,
            'xml': ExportFormat.XML,
            'txt': ExportFormat.TEXT,
            'text': ExportFormat.TEXT,
        }
        
        return format_map.get(ext, ExportFormat.JSON)


# Singleton instance
_exporter_instance: Exporter | None = None


def get_exporter() -> Exporter:
    """Get or create the singleton exporter."""
    global _exporter_instance
    if _exporter_instance is None:
        _exporter_instance = Exporter()
    return _exporter_instance


# Convenience functions
def export_data(
    data: Any,
    output: str | Path,
    format: str | None = None,
    **options
) -> None:
    """Export data to file."""
    get_exporter().export(data, output, format, **options)


def import_data(
    input_path: str | Path,
    format: str | None = None
) -> Any:
    """Import data from file."""
    return get_exporter().import_file(input_path, format)


def convert_format(
    input_path: str | Path,
    output_path: str | Path
) -> None:
    """Convert between formats."""
    get_exporter().convert(input_path, output_path)

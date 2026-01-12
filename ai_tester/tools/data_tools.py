"""
Data Tools - CSV, JSON, SQL, plotting (requires pandas/matplotlib).
"""

import os
import re
import json
import sqlite3
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from .tool_registry import Tool

OUTPUT_DIR = Path.home() / ".ai_tester" / "outputs" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class CSVAnalyzeTool(Tool):
    """Analyze CSV files."""
    name = "csv_analyze"
    description = "Load and analyze CSV file. Returns stats, columns, sample data."
    parameters = {"path": "CSV file path", "rows": "Sample rows (default: 5)"}
    
    def execute(self, path: str, rows: int = 5, **kwargs) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            if not path.exists(): return {"success": False, "error": f"File not found: {path}"}
            
            try:
                import pandas as pd
                df = pd.read_csv(str(path))
                result = {
                    "success": True, "rows": len(df), "columns": list(df.columns),
                    "types": {c: str(df[c].dtype) for c in df.columns},
                    "sample": df.head(rows).to_dict('records'),
                }
                numeric = df.select_dtypes(include=['number']).columns
                if len(numeric) > 0: result["stats"] = df[numeric].describe().to_dict()
                result["missing"] = df.isnull().sum().to_dict()
                return result
            except ImportError:
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    sample = [row for i, row in enumerate(reader) if i < rows]
                return {"success": True, "columns": reader.fieldnames, "sample": sample, "note": "Install pandas for stats"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class CSVQueryTool(Tool):
    """Query CSV data."""
    name = "csv_query"
    description = "Query CSV using simple filters (column > value)."
    parameters = {"path": "CSV file path", "query": "Filter like 'age > 30'", "limit": "Max rows (default: 100)"}
    
    def execute(self, path: str, query: str, limit: int = 100, **kwargs) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            try:
                import pandas as pd
                df = pd.read_csv(str(path))
                match = re.match(r'(\w+)\s*([><=!]+)\s*(.+)', query.strip())
                if match:
                    col, op, val = match.groups()
                    val = val.strip().strip('"\'')
                    try: val = float(val)
                    except: pass
                    ops = {'>': lambda x,v: x>v, '<': lambda x,v: x<v, '>=': lambda x,v: x>=v, 
                           '<=': lambda x,v: x<=v, '==': lambda x,v: x==v, '=': lambda x,v: x==v, '!=': lambda x,v: x!=v}
                    if op in ops: df = df[ops[op](df[col], val)]
                return {"success": True, "rows": len(df), "data": df.head(limit).to_dict('records')}
            except ImportError:
                return {"success": False, "error": "Install pandas: pip install pandas"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class PlotChartTool(Tool):
    """Generate charts."""
    name = "plot_chart"
    description = "Create chart from CSV/JSON data. Requires matplotlib."
    parameters = {"data": "CSV path or JSON data", "chart_type": "line, bar, scatter, pie, histogram", 
                  "x": "X column", "y": "Y column", "title": "Chart title"}
    
    def execute(self, data: str, chart_type: str = "line", x: str = None, y: str = None, title: str = None, **kwargs) -> Dict[str, Any]:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            output = OUTPUT_DIR / f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.figure(figsize=(10, 6))
            
            # Load data
            if data.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(data)
                x_data = df[x] if x else df.index
                y_data = df[y] if y else df.iloc[:, 0]
            else:
                records = json.loads(data) if isinstance(data, str) else data
                x_data = [r.get(x) for r in records] if x else range(len(records))
                y_data = [r.get(y) for r in records] if y else [list(r.values())[0] for r in records]
            
            if chart_type == 'line': plt.plot(x_data, y_data, marker='o')
            elif chart_type == 'bar': plt.bar(x_data, y_data)
            elif chart_type == 'scatter': plt.scatter(x_data, y_data)
            elif chart_type == 'pie': plt.pie(y_data, labels=x_data, autopct='%1.1f%%')
            elif chart_type == 'histogram': plt.hist(y_data, bins=20, edgecolor='black')
            
            if title: plt.title(title)
            if x and chart_type != 'pie': plt.xlabel(x)
            if y and chart_type not in ['pie', 'histogram']: plt.ylabel(y)
            plt.tight_layout()
            plt.savefig(str(output), dpi=150)
            plt.close()
            return {"success": True, "output": str(output), "type": chart_type}
        except ImportError:
            return {"success": False, "error": "Install matplotlib: pip install matplotlib"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class JSONQueryTool(Tool):
    """Query JSON data."""
    name = "json_query"
    description = "Query JSON using dot notation (e.g., 'users.0.name')."
    parameters = {"path": "JSON file path (or None)", "query": "Dot notation path", "data": "Direct JSON string"}
    
    def execute(self, query: str, path: str = None, data: str = None, **kwargs) -> Dict[str, Any]:
        try:
            if path:
                with open(Path(path).expanduser().resolve(), 'r') as f: json_data = json.load(f)
            elif data:
                json_data = json.loads(data) if isinstance(data, str) else data
            else:
                return {"success": False, "error": "Provide path or data"}
            
            # Navigate by dot notation
            parts = query.replace('$', '').strip('.').split('.')
            current = json_data
            for part in parts:
                if not part: continue
                try:
                    idx = int(part)
                    current = current[idx]
                except ValueError:
                    current = current.get(part) if isinstance(current, dict) else getattr(current, part, None)
            return {"success": True, "query": query, "result": current}
        except Exception as e:
            return {"success": False, "error": str(e)}


class SQLQueryTool(Tool):
    """Query SQLite database."""
    name = "sql_query"
    description = "Execute SELECT query on SQLite database."
    parameters = {"database": "Database path", "query": "SQL SELECT query", "limit": "Max rows (default: 100)"}
    
    def execute(self, database: str, query: str, limit: int = 100, **kwargs) -> Dict[str, Any]:
        try:
            db_path = Path(database).expanduser().resolve()
            if not db_path.exists(): return {"success": False, "error": f"Database not found: {db_path}"}
            if not query.strip().upper().startswith('SELECT'):
                return {"success": False, "error": "Only SELECT queries allowed. Use sql_execute for other statements."}
            
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            rows = [dict(row) for row in cursor.fetchmany(limit)]
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            conn.close()
            return {"success": True, "columns": columns, "rows": len(rows), "data": rows}
        except Exception as e:
            return {"success": False, "error": str(e)}


class SQLExecuteTool(Tool):
    """Execute SQL statements."""
    name = "sql_execute"
    description = "Execute INSERT/UPDATE/DELETE on SQLite database."
    parameters = {"database": "Database path", "statement": "SQL statement"}
    
    def execute(self, database: str, statement: str, **kwargs) -> Dict[str, Any]:
        try:
            db_path = Path(database).expanduser().resolve()
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(statement)
            affected = cursor.rowcount
            conn.commit()
            conn.close()
            return {"success": True, "affected_rows": affected}
        except Exception as e:
            return {"success": False, "error": str(e)}


class DataConvertTool(Tool):
    """Convert between data formats."""
    name = "data_convert"
    description = "Convert CSV↔JSON, JSON↔YAML."
    parameters = {"input_path": "Input file", "output_format": "csv, json, yaml", "output_path": "Output file (optional)"}
    
    def execute(self, input_path: str, output_format: str, output_path: str = None, **kwargs) -> Dict[str, Any]:
        try:
            input_path = Path(input_path).expanduser().resolve()
            suffix = input_path.suffix.lower()
            
            # Load
            if suffix == '.csv':
                import pandas as pd
                data = pd.read_csv(str(input_path)).to_dict('records')
            elif suffix == '.json':
                with open(input_path) as f: data = json.load(f)
            elif suffix in ['.yaml', '.yml']:
                import yaml
                with open(input_path) as f: data = yaml.safe_load(f)
            else:
                return {"success": False, "error": f"Unknown input format: {suffix}"}
            
            # Output
            if not output_path:
                output_path = input_path.with_suffix(f'.{output_format}')
            else:
                output_path = Path(output_path)
            
            if output_format == 'csv':
                import pandas as pd
                pd.DataFrame(data).to_csv(str(output_path), index=False)
            elif output_format == 'json':
                with open(output_path, 'w') as f: json.dump(data, f, indent=2)
            elif output_format == 'yaml':
                import yaml
                with open(output_path, 'w') as f: yaml.dump(data, f)
            
            return {"success": True, "input": str(input_path), "output": str(output_path)}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Export
DATA_TOOLS = [CSVAnalyzeTool(), CSVQueryTool(), PlotChartTool(), JSONQueryTool(), SQLQueryTool(), SQLExecuteTool(), DataConvertTool()]
def get_data_tools(): return DATA_TOOLS

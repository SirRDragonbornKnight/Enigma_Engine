"""
Data Tools - CSV, JSON, SQL, plotting (requires pandas/matplotlib).
"""

import csv
import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .tool_registry import Tool, RichParameter

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path.home() / ".enigma_engine" / "outputs" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class CSVAnalyzeTool(Tool):
    """Analyze CSV files."""
    name = "csv_analyze"
    description = "Load and analyze CSV file. Returns stats, columns, sample data."
    parameters = {"path": "CSV file path", "rows": "Sample rows (default: 5)"}
    category = "data"
    rich_parameters = [
        RichParameter(name="path", type="string", description="CSV file path", required=True),
        RichParameter(name="rows", type="integer", description="Sample rows to return", required=False, default=5, min_value=1, max_value=100),
    ]
    examples = ["csv_analyze(path='data.csv')", "csv_analyze(path='sales.csv', rows=10)"]
    
    def execute(self, path: str, rows: int = 5, **kwargs) -> dict[str, Any]:
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
                with open(path, encoding='utf-8') as f:
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
    category = "data"
    rich_parameters = [
        RichParameter(name="path", type="string", description="CSV file path", required=True),
        RichParameter(name="query", type="string", description="Filter expression (e.g., 'age > 30')", required=True),
        RichParameter(name="limit", type="integer", description="Max rows to return", required=False, default=100, min_value=1, max_value=10000),
    ]
    examples = ["csv_query(path='users.csv', query='age > 30')", "csv_query(path='sales.csv', query='amount >= 1000', limit=50)"]
    
    def execute(self, path: str, query: str, limit: int = 100, **kwargs) -> dict[str, Any]:
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
                    except Exception as e:
                        logger.debug(f"Could not convert value '{val}' to float: {e}")
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
    category = "data"
    rich_parameters = [
        RichParameter(name="data", type="string", description="CSV path or JSON data", required=True),
        RichParameter(name="chart_type", type="string", description="Chart type", required=False, default="line", enum=["line", "bar", "scatter", "pie", "histogram"]),
        RichParameter(name="x", type="string", description="X-axis column name", required=False),
        RichParameter(name="y", type="string", description="Y-axis column name", required=False),
        RichParameter(name="title", type="string", description="Chart title", required=False),
    ]
    examples = ["plot_chart(data='sales.csv', chart_type='bar', x='month', y='revenue')", "plot_chart(data='data.csv', chart_type='pie', x='category', y='count')"]
    
    def execute(self, data: str, chart_type: str = "line", x: str = None, y: str = None, title: str = None, **kwargs) -> dict[str, Any]:
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
    category = "data"
    rich_parameters = [
        RichParameter(name="query", type="string", description="Dot notation path (e.g., 'users.0.name')", required=True),
        RichParameter(name="path", type="string", description="JSON file path", required=False),
        RichParameter(name="data", type="string", description="Direct JSON string", required=False),
    ]
    examples = ["json_query(path='config.json', query='settings.theme')", "json_query(data='{\"a\": 1}', query='a')"]
    
    def execute(self, query: str, path: str = None, data: str = None, **kwargs) -> dict[str, Any]:
        try:
            if path:
                with open(Path(path).expanduser().resolve()) as f: json_data = json.load(f)
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
    category = "data"
    rich_parameters = [
        RichParameter(name="database", type="string", description="SQLite database path", required=True),
        RichParameter(name="query", type="string", description="SQL SELECT query", required=True),
        RichParameter(name="limit", type="integer", description="Max rows to return", required=False, default=100, min_value=1, max_value=10000),
    ]
    examples = ["sql_query(database='app.db', query='SELECT * FROM users WHERE active=1')"]
    
    def execute(self, database: str, query: str, limit: int = 100, **kwargs) -> dict[str, Any]:
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
    category = "data"
    rich_parameters = [
        RichParameter(name="database", type="string", description="SQLite database path", required=True),
        RichParameter(name="statement", type="string", description="SQL INSERT/UPDATE/DELETE statement", required=True),
    ]
    examples = ["sql_execute(database='app.db', statement='UPDATE users SET active=0 WHERE id=5')"]
    
    def execute(self, database: str, statement: str, **kwargs) -> dict[str, Any]:
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
    category = "data"
    rich_parameters = [
        RichParameter(name="input_path", type="string", description="Input file path", required=True),
        RichParameter(name="output_format", type="string", description="Output format", required=True, enum=["csv", "json", "yaml"]),
        RichParameter(name="output_path", type="string", description="Output file path (auto-generated if not specified)", required=False),
    ]
    examples = ["data_convert(input_path='data.csv', output_format='json')", "data_convert(input_path='config.yaml', output_format='json', output_path='config.json')"]
    
    def execute(self, input_path: str, output_format: str, output_path: str = None, **kwargs) -> dict[str, Any]:
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

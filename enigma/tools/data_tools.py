"""
Data Tools - CSV analysis, plotting, JSON queries, SQL databases.

Tools:
  - csv_analyze: Load and analyze CSV files
  - csv_query: Query CSV data with natural language
  - plot_chart: Generate charts and graphs
  - json_query: Query JSON data
  - sql_query: Query SQLite databases
  - sql_execute: Execute SQL statements
  - data_convert: Convert between data formats
"""

import os
import re
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from .tool_registry import Tool

# Output directories
PLOTS_OUTPUT_DIR = Path.home() / ".enigma" / "outputs" / "plots"
DATA_OUTPUT_DIR = Path.home() / ".enigma" / "outputs" / "data"

PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CSV TOOLS
# ============================================================================

class CSVAnalyzeTool(Tool):
    """Analyze CSV files."""
    
    name = "csv_analyze"
    description = "Load and analyze a CSV file. Returns statistics, column info, and sample data."
    parameters = {
        "path": "Path to the CSV file",
        "rows": "Number of sample rows to return (default: 5)",
        "stats": "Include statistics (default: True)",
    }
    
    def execute(self, path: str, rows: int = 5, stats: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            
            if not path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            
            # Try pandas
            try:
                import pandas as pd
                
                df = pd.read_csv(str(path))
                
                result = {
                    "success": True,
                    "path": str(path),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns),
                    "column_types": {col: str(df[col].dtype) for col in df.columns},
                    "sample_data": df.head(int(rows)).to_dict('records'),
                }
                
                if stats:
                    # Numeric statistics
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        stats_df = df[numeric_cols].describe()
                        result["statistics"] = stats_df.to_dict()
                    
                    # Missing values
                    result["missing_values"] = df.isnull().sum().to_dict()
                
                return result
                
            except ImportError:
                pass
            
            # Fallback: basic CSV parsing
            import csv
            
            with open(path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                
                sample_data = []
                row_count = 0
                for row in reader:
                    row_count += 1
                    if len(sample_data) < int(rows):
                        sample_data.append(row)
            
            return {
                "success": True,
                "path": str(path),
                "rows": row_count,
                "columns": len(columns),
                "column_names": columns,
                "sample_data": sample_data,
                "note": "For full statistics, install: pip install pandas",
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class CSVQueryTool(Tool):
    """Query CSV data."""
    
    name = "csv_query"
    description = "Query a CSV file using SQL-like syntax or simple filters."
    parameters = {
        "path": "Path to the CSV file",
        "query": "Query: SQL (e.g., 'SELECT * WHERE age > 30') or filter (e.g., 'age > 30')",
        "limit": "Maximum rows to return (default: 100)",
    }
    
    def execute(self, path: str, query: str, limit: int = 100, **kwargs) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            
            if not path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            
            # Try pandas with pandasql
            try:
                import pandas as pd
                
                df = pd.read_csv(str(path))
                
                # Try pandasql for SQL queries
                try:
                    import pandasql as ps
                    
                    # Ensure query has SELECT
                    if not query.strip().upper().startswith('SELECT'):
                        query = f"SELECT * FROM df WHERE {query}"
                    else:
                        query = query.replace('FROM csv', 'FROM df').replace('FROM data', 'FROM df')
                        if 'FROM df' not in query:
                            query = query.replace('SELECT', 'SELECT').replace('WHERE', 'FROM df WHERE')
                    
                    result_df = ps.sqldf(query, locals())
                    
                    return {
                        "success": True,
                        "rows": len(result_df),
                        "columns": list(result_df.columns),
                        "data": result_df.head(int(limit)).to_dict('records'),
                    }
                    
                except ImportError:
                    pass
                
                # Fallback: simple pandas filtering
                # Parse simple conditions like "column > value"
                match = re.match(r'(\w+)\s*([><=!]+)\s*(.+)', query.strip())
                if match:
                    col, op, val = match.groups()
                    val = val.strip().strip('"\'')
                    
                    # Try to convert to number
                    try:
                        val = float(val)
                    except:
                        pass
                    
                    if op == '>':
                        result_df = df[df[col] > val]
                    elif op == '<':
                        result_df = df[df[col] < val]
                    elif op == '>=':
                        result_df = df[df[col] >= val]
                    elif op == '<=':
                        result_df = df[df[col] <= val]
                    elif op in ['==', '=']:
                        result_df = df[df[col] == val]
                    elif op == '!=':
                        result_df = df[df[col] != val]
                    else:
                        result_df = df
                    
                    return {
                        "success": True,
                        "rows": len(result_df),
                        "columns": list(result_df.columns),
                        "data": result_df.head(int(limit)).to_dict('records'),
                    }
                
                return {
                    "success": False,
                    "error": "Could not parse query. Use format: 'column > value' or install pandasql for SQL",
                }
                
            except ImportError:
                pass
            
            return {
                "success": False,
                "error": "pandas not available. Install: pip install pandas pandasql",
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# PLOTTING TOOLS
# ============================================================================

class PlotChartTool(Tool):
    """Generate charts and graphs."""
    
    name = "plot_chart"
    description = "Create a chart or graph from data. Supports line, bar, scatter, pie, histogram."
    parameters = {
        "data_path": "Path to CSV/JSON data file, or None to use provided data",
        "x_column": "Column for X axis",
        "y_column": "Column for Y axis (or values for pie chart)",
        "chart_type": "Type: 'line', 'bar', 'scatter', 'pie', 'histogram' (default: line)",
        "title": "Chart title",
        "output_path": "Path for output image (default: auto-generated)",
        "data": "Direct data as JSON array (alternative to data_path)",
    }
    
    def execute(self, x_column: str = None, y_column: str = None, 
                chart_type: str = "line", title: str = None,
                data_path: str = None, output_path: str = None, 
                data: str = None, **kwargs) -> Dict[str, Any]:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            # Load data
            df = None
            
            if data_path:
                try:
                    import pandas as pd
                    path = Path(data_path).expanduser().resolve()
                    if path.suffix.lower() == '.csv':
                        df = pd.read_csv(str(path))
                    elif path.suffix.lower() == '.json':
                        df = pd.read_json(str(path))
                except ImportError:
                    # Manual loading
                    import csv
                    with open(path, 'r') as f:
                        reader = csv.DictReader(f)
                        records = list(reader)
                    data = json.dumps(records)
            
            if data and df is None:
                try:
                    import pandas as pd
                    records = json.loads(data) if isinstance(data, str) else data
                    df = pd.DataFrame(records)
                except ImportError:
                    records = json.loads(data) if isinstance(data, str) else data
            
            # Generate output path
            if not output_path:
                output_path = PLOTS_OUTPUT_DIR / f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            else:
                output_path = Path(output_path).expanduser().resolve()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            
            if df is not None:
                import pandas as pd
                
                if chart_type == 'line':
                    if x_column and y_column:
                        plt.plot(df[x_column], df[y_column], marker='o')
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                    else:
                        df.plot()
                        
                elif chart_type == 'bar':
                    if x_column and y_column:
                        plt.bar(df[x_column], df[y_column])
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                    else:
                        df.plot(kind='bar')
                        
                elif chart_type == 'scatter':
                    if x_column and y_column:
                        plt.scatter(df[x_column], df[y_column])
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                        
                elif chart_type == 'pie':
                    if y_column:
                        labels = df[x_column] if x_column else df.index
                        plt.pie(df[y_column], labels=labels, autopct='%1.1f%%')
                        
                elif chart_type == 'histogram':
                    col = y_column or x_column or df.columns[0]
                    plt.hist(df[col], bins=20, edgecolor='black')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
            
            else:
                # Simple data from records
                records = json.loads(data) if isinstance(data, str) else data
                if x_column and y_column:
                    x_vals = [r.get(x_column) for r in records]
                    y_vals = [r.get(y_column) for r in records]
                    
                    if chart_type == 'line':
                        plt.plot(x_vals, y_vals, marker='o')
                    elif chart_type == 'bar':
                        plt.bar(x_vals, y_vals)
                    elif chart_type == 'scatter':
                        plt.scatter(x_vals, y_vals)
            
            if title:
                plt.title(title)
            
            plt.tight_layout()
            plt.savefig(str(output_path), dpi=150)
            plt.close()
            
            return {
                "success": True,
                "output_path": str(output_path),
                "chart_type": chart_type,
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "matplotlib not available. Install: pip install matplotlib pandas",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# JSON TOOLS
# ============================================================================

class JSONQueryTool(Tool):
    """Query JSON data."""
    
    name = "json_query"
    description = "Query JSON data using JSONPath or simple dot notation."
    parameters = {
        "path": "Path to JSON file, or None to use provided data",
        "query": "Query: JSONPath (e.g., '$.users[*].name') or dot notation (e.g., 'users.0.name')",
        "data": "Direct JSON data (alternative to path)",
    }
    
    def execute(self, query: str, path: str = None, data: str = None, **kwargs) -> Dict[str, Any]:
        try:
            # Load data
            if path:
                path = Path(path).expanduser().resolve()
                with open(path, 'r') as f:
                    json_data = json.load(f)
            elif data:
                json_data = json.loads(data) if isinstance(data, str) else data
            else:
                return {"success": False, "error": "Must provide path or data"}
            
            # Try jsonpath-ng for JSONPath queries
            try:
                from jsonpath_ng import parse
                
                jsonpath_expr = parse(query)
                matches = jsonpath_expr.find(json_data)
                
                results = [match.value for match in matches]
                
                return {
                    "success": True,
                    "query": query,
                    "count": len(results),
                    "results": results,
                }
                
            except ImportError:
                pass
            
            # Fallback: simple dot notation
            def get_by_path(obj, path_str):
                parts = path_str.replace('$', '').strip('.').split('.')
                current = obj
                
                for part in parts:
                    if not part:
                        continue
                    
                    # Handle array index
                    if part.isdigit():
                        current = current[int(part)]
                    elif '[' in part:
                        # Handle array access like "users[0]"
                        key, idx = part.split('[')
                        idx = int(idx.rstrip(']'))
                        current = current[key][idx]
                    else:
                        current = current[part]
                
                return current
            
            try:
                result = get_by_path(json_data, query)
                return {
                    "success": True,
                    "query": query,
                    "result": result,
                    "note": "For advanced queries, install: pip install jsonpath-ng",
                }
            except (KeyError, IndexError, TypeError) as e:
                return {"success": False, "error": f"Query failed: {e}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class JSONTransformTool(Tool):
    """Transform JSON data."""
    
    name = "json_transform"
    description = "Transform JSON data: flatten, filter, select fields, etc."
    parameters = {
        "path": "Path to JSON file",
        "operation": "Operation: 'flatten', 'filter', 'select', 'sort'",
        "params": "Operation parameters as JSON",
        "output_path": "Path for output (default: adds '_transformed')",
    }
    
    def execute(self, path: str, operation: str, params: str = None,
                output_path: str = None, **kwargs) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            params = json.loads(params) if params else {}
            
            result = data
            
            if operation == 'flatten':
                # Flatten nested structures
                def flatten(obj, prefix=''):
                    items = {}
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            new_key = f"{prefix}.{k}" if prefix else k
                            items.update(flatten(v, new_key))
                    elif isinstance(obj, list):
                        for i, v in enumerate(obj):
                            items.update(flatten(v, f"{prefix}[{i}]"))
                    else:
                        items[prefix] = obj
                    return items
                
                if isinstance(data, list):
                    result = [flatten(item) for item in data]
                else:
                    result = flatten(data)
                    
            elif operation == 'filter':
                # Filter array by condition
                key = params.get('key')
                value = params.get('value')
                op = params.get('op', '==')
                
                if isinstance(data, list) and key:
                    if op == '==':
                        result = [item for item in data if item.get(key) == value]
                    elif op == '!=':
                        result = [item for item in data if item.get(key) != value]
                    elif op == '>':
                        result = [item for item in data if item.get(key, 0) > value]
                    elif op == '<':
                        result = [item for item in data if item.get(key, 0) < value]
                    elif op == 'contains':
                        result = [item for item in data if value in str(item.get(key, ''))]
                        
            elif operation == 'select':
                # Select specific fields
                fields = params.get('fields', [])
                if isinstance(data, list) and fields:
                    result = [{k: item.get(k) for k in fields} for item in data]
                elif isinstance(data, dict) and fields:
                    result = {k: data.get(k) for k in fields}
                    
            elif operation == 'sort':
                # Sort array by key
                key = params.get('key')
                reverse = params.get('reverse', False)
                if isinstance(data, list) and key:
                    result = sorted(data, key=lambda x: x.get(key, 0), reverse=reverse)
            
            # Save result
            if not output_path:
                output_path = path.parent / f"{path.stem}_transformed.json"
            else:
                output_path = Path(output_path).expanduser().resolve()
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            return {
                "success": True,
                "operation": operation,
                "output_path": str(output_path),
                "result_count": len(result) if isinstance(result, list) else 1,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# SQL TOOLS
# ============================================================================

class SQLQueryTool(Tool):
    """Query SQLite databases."""
    
    name = "sql_query"
    description = "Execute a SELECT query on a SQLite database."
    parameters = {
        "database": "Path to SQLite database file",
        "query": "SQL SELECT query",
        "limit": "Maximum rows to return (default: 100)",
    }
    
    def execute(self, database: str, query: str, limit: int = 100, **kwargs) -> Dict[str, Any]:
        try:
            db_path = Path(database).expanduser().resolve()
            
            if not db_path.exists():
                return {"success": False, "error": f"Database not found: {db_path}"}
            
            # Safety check - only allow SELECT
            query_upper = query.strip().upper()
            if not query_upper.startswith('SELECT'):
                return {"success": False, "error": "Only SELECT queries allowed. Use sql_execute for other operations."}
            
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            
            try:
                cursor = conn.execute(query)
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchmany(int(limit))
                
                data = [dict(row) for row in rows]
                
                return {
                    "success": True,
                    "columns": columns,
                    "row_count": len(data),
                    "data": data,
                }
                
            finally:
                conn.close()
            
        except sqlite3.Error as e:
            return {"success": False, "error": f"SQL error: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class SQLExecuteTool(Tool):
    """Execute SQL statements."""
    
    name = "sql_execute"
    description = "Execute SQL statements (INSERT, UPDATE, DELETE, CREATE) on a SQLite database."
    parameters = {
        "database": "Path to SQLite database file",
        "statement": "SQL statement to execute",
    }
    
    # Blocked destructive operations
    BLOCKED_PATTERNS = ['DROP DATABASE', 'DROP TABLE', 'TRUNCATE']
    
    def execute(self, database: str, statement: str, **kwargs) -> Dict[str, Any]:
        try:
            # Safety check
            statement_upper = statement.strip().upper()
            for pattern in self.BLOCKED_PATTERNS:
                if pattern in statement_upper:
                    return {"success": False, "error": f"Blocked operation: {pattern}"}
            
            db_path = Path(database).expanduser().resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(str(db_path))
            
            try:
                cursor = conn.execute(statement)
                conn.commit()
                
                return {
                    "success": True,
                    "rows_affected": cursor.rowcount,
                    "lastrowid": cursor.lastrowid,
                }
                
            finally:
                conn.close()
            
        except sqlite3.Error as e:
            return {"success": False, "error": f"SQL error: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class SQLTablesTool(Tool):
    """List tables in a SQLite database."""
    
    name = "sql_tables"
    description = "List all tables in a SQLite database with their schemas."
    parameters = {
        "database": "Path to SQLite database file",
    }
    
    def execute(self, database: str, **kwargs) -> Dict[str, Any]:
        try:
            db_path = Path(database).expanduser().resolve()
            
            if not db_path.exists():
                return {"success": False, "error": f"Database not found: {db_path}"}
            
            conn = sqlite3.connect(str(db_path))
            
            try:
                # Get table names
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                tables = [row[0] for row in cursor.fetchall()]
                
                # Get schema for each table
                schemas = {}
                for table in tables:
                    cursor = conn.execute(f"PRAGMA table_info({table})")
                    columns = []
                    for row in cursor.fetchall():
                        columns.append({
                            "name": row[1],
                            "type": row[2],
                            "notnull": bool(row[3]),
                            "primary_key": bool(row[5]),
                        })
                    schemas[table] = columns
                
                return {
                    "success": True,
                    "table_count": len(tables),
                    "tables": tables,
                    "schemas": schemas,
                }
                
            finally:
                conn.close()
            
        except sqlite3.Error as e:
            return {"success": False, "error": f"SQL error: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# DATA CONVERSION TOOLS
# ============================================================================

class DataConvertTool(Tool):
    """Convert between data formats."""
    
    name = "data_convert"
    description = "Convert data between formats: CSV, JSON, SQLite, Excel."
    parameters = {
        "input_path": "Path to input file",
        "output_format": "Output format: 'csv', 'json', 'sqlite', 'excel'",
        "output_path": "Path for output file (default: auto-generated)",
        "table_name": "Table name for SQLite output (default: 'data')",
    }
    
    def execute(self, input_path: str, output_format: str, 
                output_path: str = None, table_name: str = "data", **kwargs) -> Dict[str, Any]:
        try:
            input_path = Path(input_path).expanduser().resolve()
            
            if not input_path.exists():
                return {"success": False, "error": f"File not found: {input_path}"}
            
            input_ext = input_path.suffix.lower()
            
            # Try pandas for conversions
            try:
                import pandas as pd
                
                # Load input
                if input_ext == '.csv':
                    df = pd.read_csv(str(input_path))
                elif input_ext == '.json':
                    df = pd.read_json(str(input_path))
                elif input_ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(str(input_path))
                elif input_ext in ['.db', '.sqlite', '.sqlite3']:
                    conn = sqlite3.connect(str(input_path))
                    # Get first table
                    tables = pd.read_sql_query(
                        "SELECT name FROM sqlite_master WHERE type='table'", conn
                    )
                    if len(tables) > 0:
                        df = pd.read_sql_query(f"SELECT * FROM {tables['name'][0]}", conn)
                    conn.close()
                else:
                    return {"success": False, "error": f"Unsupported input format: {input_ext}"}
                
                # Generate output path
                if not output_path:
                    ext_map = {'csv': '.csv', 'json': '.json', 'sqlite': '.db', 'excel': '.xlsx'}
                    output_path = DATA_OUTPUT_DIR / f"{input_path.stem}{ext_map.get(output_format, '.out')}"
                else:
                    output_path = Path(output_path).expanduser().resolve()
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save output
                if output_format == 'csv':
                    df.to_csv(str(output_path), index=False)
                elif output_format == 'json':
                    df.to_json(str(output_path), orient='records', indent=2)
                elif output_format == 'sqlite':
                    conn = sqlite3.connect(str(output_path))
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                    conn.close()
                elif output_format == 'excel':
                    df.to_excel(str(output_path), index=False)
                else:
                    return {"success": False, "error": f"Unsupported output format: {output_format}"}
                
                return {
                    "success": True,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "rows": len(df),
                    "columns": len(df.columns),
                }
                
            except ImportError:
                return {
                    "success": False,
                    "error": "pandas not available. Install: pip install pandas openpyxl",
                }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

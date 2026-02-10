"""
Built-in Code Generator

Zero-dependency code generation using templates and patterns.
Not as smart as AI but works without any installs.
"""

import re
import time
from typing import Any

# Code templates for common patterns
TEMPLATES = {
    "python": {
        "function": '''def {name}({params}):
    """
    {description}
    
    Args:
        {param_docs}
    
    Returns:
        {return_doc}
    """
    # TODO: Implement {name}
    {body}
''',
        "class": '''class {name}:
    """
    {description}
    """
    
    def __init__(self{init_params}):
        """Initialize {name}."""
        {init_body}
    
    def __repr__(self):
        return f"{name}()"
''',
        "main": '''#!/usr/bin/env python3
"""
{description}
"""

def main():
    """Main entry point."""
    {body}

if __name__ == "__main__":
    main()
''',
        "test": '''import unittest

class Test{name}(unittest.TestCase):
    """Tests for {name}."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def test_{test_name}(self):
        """Test {description}."""
        # TODO: Implement test
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
''',
        "api": '''from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/{endpoint}", methods=["GET"])
def {name}():
    """
    {description}
    """
    return jsonify({{"status": "ok", "message": "{name}"}})

if __name__ == "__main__":
    app.run(debug=True)
''',
    },
    "javascript": {
        "function": '''/**
 * {description}
 * @param {{{param_types}}} {params}
 * @returns {{{return_type}}}
 */
function {name}({params}) {{
    // TODO: Implement {name}
    {body}
}}
''',
        "class": '''/**
 * {description}
 */
class {name} {{
    /**
     * Create a {name}.
     */
    constructor({init_params}) {{
        {init_body}
    }}
    
    /**
     * String representation.
     */
    toString() {{
        return `{name}`;
    }}
}}

export default {name};
''',
        "react": '''import React, {{ useState }} from 'react';

/**
 * {description}
 */
function {name}({{ {props} }}) {{
    const [state, setState] = useState(null);
    
    return (
        <div className="{name}">
            {{/* TODO: Implement {name} */}}
            <p>{name} Component</p>
        </div>
    );
}}

export default {name};
''',
        "api": '''const express = require('express');
const app = express();

app.use(express.json());

/**
 * {description}
 */
app.get('/{endpoint}', (req, res) => {{
    res.json({{ status: 'ok', message: '{name}' }});
}});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {{
    console.log(`Server running on port ${{PORT}}`);
}});
''',
    },
    "html": {
        "page": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: system-ui, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <main>
        <!-- TODO: Add content -->
        <p>{description}</p>
    </main>
    <script>
        // TODO: Add JavaScript
    </script>
</body>
</html>
''',
        "form": '''<form id="{name}" action="{action}" method="{method}">
    <div class="form-group">
        <label for="input1">Input 1:</label>
        <input type="text" id="input1" name="input1" required>
    </div>
    <div class="form-group">
        <label for="input2">Input 2:</label>
        <input type="text" id="input2" name="input2">
    </div>
    <button type="submit">Submit</button>
</form>
''',
    },
    "sql": {
        "table": '''-- Create {name} table
CREATE TABLE {name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {columns}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on {name}
CREATE INDEX idx_{name}_created ON {name}(created_at);
''',
        "query": '''-- {description}
SELECT 
    {columns}
FROM {table}
WHERE {condition}
ORDER BY {order_by}
LIMIT {limit};
''',
    },
    "bash": {
        "script": '''#!/bin/bash
# {description}
# Usage: ./{name}.sh [options]

set -e  # Exit on error

# Configuration
{config}

# Functions
{name}() {{
    echo "Running {name}..."
    # TODO: Implement
}}

# Main
main() {{
    echo "Starting script..."
    {name}
    echo "Done!"
}}

main "$@"
''',
    },
}


class BuiltinCodeGen:
    """
    Built-in code generator using templates.
    No external dependencies required.
    """
    
    def __init__(self):
        self.is_loaded = False
        
    def load(self) -> bool:
        """Load the generator (always succeeds)."""
        self.is_loaded = True
        return True
    
    def unload(self):
        """Unload."""
        self.is_loaded = False
    
    def _extract_intent(self, prompt: str) -> dict[str, Any]:
        """Extract intent and parameters from the prompt."""
        prompt_lower = prompt.lower()
        
        # Detect what type of code
        intent = {
            "type": "function",  # Default
            "name": "generated",
            "description": prompt,
            "params": "",
            "body": "pass",
        }
        
        # Detect type keywords
        if any(w in prompt_lower for w in ["class", "object", "model"]):
            intent["type"] = "class"
        elif any(w in prompt_lower for w in ["test", "unittest", "testing"]):
            intent["type"] = "test"
        elif any(w in prompt_lower for w in ["api", "endpoint", "rest", "server"]):
            intent["type"] = "api"
        elif any(w in prompt_lower for w in ["react", "component", "jsx"]):
            intent["type"] = "react"
        elif any(w in prompt_lower for w in ["page", "html", "website"]):
            intent["type"] = "page"
        elif any(w in prompt_lower for w in ["form", "input", "submit"]):
            intent["type"] = "form"
        elif any(w in prompt_lower for w in ["table", "database", "schema"]):
            intent["type"] = "table"
        elif any(w in prompt_lower for w in ["query", "select", "find"]):
            intent["type"] = "query"
        elif any(w in prompt_lower for w in ["script", "bash", "shell"]):
            intent["type"] = "script"
        elif any(w in prompt_lower for w in ["main", "program", "application"]):
            intent["type"] = "main"
        
        # Try to extract a name
        name_patterns = [
            r"(?:called|named|name)\s+['\"]?(\w+)['\"]?",
            r"(?:function|class|def)\s+(\w+)",
            r"(\w+)\s+(?:function|class|component)",
            r"create\s+(?:a\s+)?(\w+)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                intent["name"] = match.group(1)
                break
        
        # Clean up name
        intent["name"] = re.sub(r'[^a-zA-Z0-9_]', '', intent["name"])
        if not intent["name"] or intent["name"][0].isdigit():
            intent["name"] = "generated"
        
        return intent
    
    def generate(self, prompt: str, language: str = "python", **kwargs) -> dict[str, Any]:
        """Generate code based on the prompt."""
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        if not prompt.strip():
            return {"success": False, "error": "Empty prompt"}
        
        try:
            start = time.time()
            
            # Get intent from prompt
            intent = self._extract_intent(prompt)
            
            # Get templates for this language
            lang_templates = TEMPLATES.get(language, TEMPLATES.get("python", {}))
            
            # Find best matching template
            template_key = intent["type"]
            if template_key not in lang_templates:
                # Fallback to function
                template_key = "function" if "function" in lang_templates else list(lang_templates.keys())[0]
            
            template = lang_templates.get(template_key, "// TODO: {description}")
            
            # Fill in template
            code = template.format(
                name=intent["name"],
                description=intent["description"][:200],
                params="",
                param_docs="None",
                return_doc="None",
                body="pass" if language == "python" else "// TODO",
                init_params="",
                init_body="pass" if language == "python" else "// TODO",
                param_types="*",
                return_type="*",
                props="",
                endpoint=intent["name"].lower(),
                title=intent["name"].replace("_", " ").title(),
                action="/submit",
                method="POST",
                columns="name TEXT,\n    value TEXT,",
                table=intent["name"],
                condition="1=1",
                order_by="created_at DESC",
                limit="10",
                config=f'NAME="{intent["name"]}"',
                test_name=intent["name"].lower(),
            )
            
            # Add a helpful comment at the top
            comment_prefix = {
                "python": "#",
                "javascript": "//",
                "typescript": "//",
                "java": "//",
                "cpp": "//",
                "go": "//",
                "rust": "//",
                "html": "<!--",
                "css": "/*",
                "sql": "--",
                "bash": "#",
            }.get(language, "#")
            
            comment_suffix = {
                "html": " -->",
                "css": " */",
            }.get(language, "")
            
            header = f"{comment_prefix} Generated by Enigma AI Engine Built-in Code Generator{comment_suffix}\n"
            header += f"{comment_prefix} Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}{comment_suffix}\n\n"
            
            code = header + code
            
            return {
                "success": True,
                "code": code,
                "duration": time.time() - start,
                "template_used": template_key,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

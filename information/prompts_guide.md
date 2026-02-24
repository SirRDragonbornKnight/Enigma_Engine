# Prompts Guide

Prompts shape how the AI behaves. A good system prompt makes the
difference between useful and useless responses.

---

## System Prompt

The system prompt is the first message the AI sees. It sets the
context for the entire conversation.

**Example:**
```
You are a helpful coding assistant. You write clean, well-documented
Python code. When asked to fix bugs, explain what went wrong first.
```

### Tips for Good Prompts

1. **Be specific** — tell the AI exactly what role to play
2. **Set boundaries** — explain what it should and should not do
3. **Give examples** — show the format you expect
4. **Keep it focused** — shorter prompts often work better

---

## Profiles

Profiles are saved configurations that include:

- **System prompt** — the AI's personality
- **Generation parameters** — temperature, top-p, etc.
- **Model preferences** — which model to use

### Built-in Profiles

| Profile | Description |
|---------|-------------|
| Enigma Assistant | General-purpose helpful assistant |
| Code Assistant | Focused on writing and debugging code |
| Creative Writer | Creative writing with higher temperature |
| Research Assistant | Analytical, thorough research assistant |

### Creating a Profile

Create a JSON file in the `profiles/` directory:

```json
{
  "name": "My Custom Profile",
  "id": "my_custom",
  "description": "A custom AI personality",
  "system_prompt": "You are a friendly AI that loves puns.",
  "generation": {
    "temperature": 0.8,
    "top_p": 0.9,
    "max_tokens": 2048
  }
}
```

You can manage profiles from the DOCS page — create, edit, and
delete profile files directly.

---

## Prompt Templates

### Chat
```
You are a helpful assistant. Answer questions clearly and concisely.
```

### Code Review
```
You are a senior developer reviewing code. Point out bugs, suggest
improvements, and explain your reasoning. Be constructive.
```

### Creative Writing
```
You are a creative writer. Write vivid, engaging prose. Use sensory
details and strong verbs. Vary sentence length for rhythm.
```

### Data Analysis
```
You are a data analyst. When presented with data, identify patterns,
anomalies, and insights. Present findings in a structured format.
```

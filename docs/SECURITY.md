# Security Best Practices for Enigma Engine

This document outlines security considerations and best practices when using and developing Enigma Engine.

## Configuration Security

### API Keys and Secrets

**NEVER** commit API keys, passwords, or other secrets to version control.

#### Recommended Approach

Use environment variables for sensitive data:

```bash
# Set environment variables
export OPENAI_API_KEY="your-key-here"
export REPLICATE_API_TOKEN="your-token-here"
export ELEVENLABS_API_KEY="your-key-here"
```

Or use a `.env` file (already in `.gitignore`):

```bash
# .env file
OPENAI_API_KEY=your-key-here
REPLICATE_API_TOKEN=your-token-here
ELEVENLABS_API_KEY=your-key-here
```

#### What NOT to Do

❌ **Don't hardcode secrets:**
```python
# BAD - Never do this!
api_key = "sk-abcdef123456"
```

✅ **Do use environment variables:**
```python
# GOOD
api_key = os.environ.get("OPENAI_API_KEY")
```

### Configuration Files

The following files should NEVER be committed:
- `enigma_config.json` (may contain paths or settings specific to your system)
- `.env` files
- Any files ending in `_secrets.json` or `_credentials.json`
- Private keys (`.key`, `.pem` files)

## File Operations Security

### Path Validation

When working with file paths, always:

1. **Validate user input** - Check for empty paths, invalid characters
2. **Use absolute paths** - Resolve paths with `.resolve()` to prevent traversal attacks
3. **Check file size** - Limit file sizes to prevent memory exhaustion
4. **Sanitize filenames** - Remove dangerous characters from user-provided names

Example from `enigma/tools/file_tools.py`:
```python
# Validate empty path
if not path:
    return {"success": False, "error": "Path cannot be empty"}

# Resolve to absolute path
path = Path(path).expanduser().resolve()

# Check file size
if file_size > 100 * 1024 * 1024:  # 100MB limit
    return {"success": False, "error": "File too large"}
```

### Protected Paths

The delete tool protects critical system directories:
- `/`, `/home`, `/usr`, `/bin`, `/etc`, `/var`, `/root`
- `/boot`, `/lib`, `/lib64`, `/sbin`, `/opt`, `/sys`, `/proc`

## Network Security

### Web Requests

When fetching external content:

1. **Validate URLs** - Ensure URLs start with `http://` or `https://`
2. **Set timeouts** - Prevent indefinite hangs (use 10-15 second timeouts)
3. **Check content types** - Verify response content type before processing
4. **Limit response size** - Cap downloaded content to prevent memory issues

Example from `enigma/tools/web_tools.py`:
```python
# Validate URL format
if not url.startswith(('http://', 'https://')):
    return {"success": False, "error": "URL must start with http:// or https://"}

# Set timeout
with urllib.request.urlopen(req, timeout=15) as response:
    # Check content type
    content_type = response.headers.get('Content-Type', '')
    if 'text/html' not in content_type:
        return {"success": False, "error": f"Unsupported content type"}
```

### Discovery and Broadcasting

The network discovery system broadcasts on the local network. In untrusted environments:

1. **Disable discovery** if not needed
2. **Use firewalls** to restrict access to the API port
3. **Validate all discovery messages** before processing

## Model Security

### Model Files

- **Verify sources** - Only load model files from trusted sources
- **Use `weights_only=True`** when loading with PyTorch to prevent arbitrary code execution:
  ```python
  state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
  ```

### Training Data

- **Review training data** before use to avoid learning harmful patterns
- **Filter sensitive information** from training data
- **Don't train on private conversations** without proper anonymization

## Privacy

### Local-First by Default

Enigma Engine is designed to run locally by default:
- ✅ No data sent to external services without explicit opt-in
- ✅ All processing happens on your machine
- ✅ API keys only needed for cloud providers

### Cloud Services

When enabling cloud services (OpenAI, Replicate, etc.):
- ⚠️ Understand that data will be sent to external APIs
- ⚠️ Review the provider's privacy policy
- ⚠️ Don't send sensitive or private information

The system warns when loading cloud modules:
```
⚠️  Warning: Module 'image_gen_api' connects to external cloud services
```

## Input Validation

Always validate user inputs:

1. **Check for empty/None values**
2. **Validate data types**
3. **Check ranges** for numeric inputs
4. **Sanitize strings** used in file paths or system commands

Example:
```python
def execute(self, query: str, num_results: int = 5, **kwargs):
    # Validate empty
    if not query or not query.strip():
        return {"success": False, "error": "Query cannot be empty"}
    
    # Validate range
    if num_results <= 0:
        return {"success": False, "error": "num_results must be positive"}
    
    # Limit maximum
    if num_results > 20:
        num_results = 20
```

## Reporting Security Issues

If you discover a security vulnerability:

1. **Do NOT** open a public GitHub issue
2. Email the maintainers privately
3. Provide details about the vulnerability and steps to reproduce
4. Allow time for a fix before public disclosure

## Checklist for Contributors

When adding new features:

- [ ] No hardcoded secrets or API keys
- [ ] Input validation on all user-provided data
- [ ] File size limits for file operations
- [ ] Timeouts on network operations
- [ ] Use specific exception types, not bare `except:`
- [ ] Validate file paths and prevent directory traversal
- [ ] Check content types for downloaded data
- [ ] Document security implications in docstrings

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [PyTorch Security](https://pytorch.org/docs/stable/notes/serialization.html#security)

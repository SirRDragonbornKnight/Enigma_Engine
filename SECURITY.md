# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: sirknighth3@gmail.com

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## Security Best Practices

When using Enigma Engine:

1. **File Paths**: Always validate and sanitize user-provided file paths
2. **API Keys**: Never commit API keys or secrets to the repository
3. **User Input**: Validate all user input before processing
4. **Dependencies**: Keep dependencies updated to patch security vulnerabilities
5. **Local-First**: Use local modules by default; only enable cloud services when needed
6. **Network**: If running the API server, use authentication and HTTPS in production

## Security Features

Enigma Engine includes several security features:

- Input validation on all public APIs
- Path sanitization for file operations
- No hardcoded secrets (environment variables only)
- Optional dependencies for cloud services
- Configurable resource limits to prevent resource exhaustion

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find any similar problems
3. Prepare fixes for all supported releases
4. Release patches as soon as possible

## Comments on this Policy

If you have suggestions on how this process could be improved, please submit a pull request.

# Commands Reference

The AI can execute commands using `[CMD]command[/CMD]` blocks in its
responses. You can also run commands directly from the CMD page.

---

## Engine Commands

### Config
| Command | Description |
|---------|-------------|
| config.get KEY | Get a config value |
| config.set KEY VALUE | Set a config value |
| config.list | List all config values |

### Model
| Command | Description |
|---------|-------------|
| model.info | Show loaded model info |
| model.list | List available models |
| model.load PATH | Load a model |
| model.switch PATH | Switch to a different model |
| model.download ID | Download from HuggingFace |

### Training
| Command | Description |
|---------|-------------|
| train.start | Start training |
| train.stop | Stop training |
| train.status | Check training progress |
| train.data.add PATH | Add training data |
| train.data.list | List training data files |

### Files
| Command | Description |
|---------|-------------|
| file.read PATH | Read a file |
| file.write PATH CONTENT | Write to a file |
| file.append PATH CONTENT | Append to a file |
| file.list DIR | List files in directory |

### Search
| Command | Description |
|---------|-------------|
| search.files PATTERN | Search for files |
| search.content DIR QUERY | Search file contents in a directory |
| web.fetch URL | Fetch content from a URL |

### System
| Command | Description |
|---------|-------------|
| system.info | Show system information |
| system.clear | Clear terminal |
| clipboard.copy TEXT | Copy text to clipboard |
| shell COMMAND | Run a terminal command (restricted) |
| history | Show command history |
| help | Show all available commands |
| stop | Stop current generation |

### Bricks
| Command | Description |
|---------|-------------|
| brick.list | List all bricks |
| brick.status | Check brick status |
| brick.start | Start bricks |
| brick.stop | Stop bricks |
| brick.send NAME CMD | Send command to a brick |

### Memory
| Command | Description |
|---------|-------------|
| memory.save NAME | Save current conversation to memory |
| memory.load KEY | Load from memory |
| memory.list | List saved memory keys |

### Notes
| Command | Description |
|---------|-------------|
| note.add TEXT | Add a note |
| note.list | List all notes |

---

## CMD Page Modes

### SYSTEM Mode
Real PowerShell terminal. Run any system command.

### ENGINE Mode
Run engine commands. Use `ask <question>` to query the AI.

### AI ACCESS Toggle
When ON, the AI can execute real system commands through PowerShell.
When OFF, the AI is restricted to engine commands only.

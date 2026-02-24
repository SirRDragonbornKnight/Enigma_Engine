# Echo Brick

A simple test brick that echoes messages back. Use this as a
reference for building your own bricks.

## Commands

| Command | Description |
|---------|-------------|
| echo | Echo back a message |
| reverse | Reverse a string |
| count | Count words in text |

## Usage

The echo brick is mainly for testing the brick plugin system.
Send it a message and it sends it right back.

## Building Your Own Brick

1. Copy the `_template/` folder in `bricks/`
2. Edit `brick.json` with your brick's info
3. Edit `main.py` with your brick's logic
4. Add a `docs/` folder with documentation
5. Restart the engine — your brick appears automatically

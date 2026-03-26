"""Temporary script to batch-replace non-atomic writes."""
import re

def fix_file(path, old, new):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    count = content.count(old)
    if count == 0:
        print(f"  No matches in {path}")
        return
    content = content.replace(old, new)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  {path}: replaced {count} occurrences")

# #27: gui_pages_config.py — 10 identical patterns
old27 = (
    '            settings_path.write_text(\n'
    '                json.dumps(settings, indent=2),\n'
    '                encoding="utf-8")'
)
new27 = (
    '            from enigma_engine.core.safe_save import atomic_write_json\n'
    '            atomic_write_json(settings_path, settings)'
)
fix_file('enigma_engine/gui/gui_pages_config.py', old27, new27)

# #28: gui_logic.py — 2 sites (settings_path.write_text with data)
old28 = (
    '            settings_path.write_text(\n'
    '                json.dumps(data, indent=2), encoding="utf-8")'
)
new28 = (
    '            from enigma_engine.core.safe_save import atomic_write_json\n'
    '            atomic_write_json(settings_path, data)'
)
fix_file('enigma_engine/gui/gui_logic.py', old28, new28)

# #29: gui_logic_chat.py — path.write_text(json.dumps(data, ...))
# Need to read and check each pattern
print("\n#29: gui_logic_chat.py — checking patterns...")
with open('enigma_engine/gui/gui_logic_chat.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern 1: path.write_text(json.dumps(data, indent=2), encoding="utf-8")
old29a = 'path.write_text(json.dumps(data, indent=2), encoding="utf-8")'
new29a = ('from enigma_engine.core.safe_save import atomic_write_json\n'
          '                atomic_write_json(path, data)')
c = content.count(old29a)
print(f"  Pattern 'path.write_text(json.dumps(data, indent=2),...': {c} matches")
if c:
    content = content.replace(old29a, new29a)

# Pattern 2: prompts_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
old29b = 'prompts_path.write_text(json.dumps(data, indent=2), encoding="utf-8")'
new29b = ('from enigma_engine.core.safe_save import atomic_write_json\n'
          '            atomic_write_json(prompts_path, data)')
c = content.count(old29b)
print(f"  Pattern 'prompts_path.write_text(json.dumps(data, ...': {c} matches")
if c:
    content = content.replace(old29b, new29b)

with open('enigma_engine/gui/gui_logic_chat.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("  gui_logic_chat.py updated")

print("\nDone!")

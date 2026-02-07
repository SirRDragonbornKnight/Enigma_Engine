#!/bin/bash
# Force kill all Enigma AI Engine processes

echo "Killing Enigma AI Engine processes..."

# Kill by process name
pkill -9 -f "python.*run.py"
pkill -9 -f "Enigma AI Engine"

# Kill any Python processes in the Enigma AI Engine directory
ps aux | grep "[p]ython.*Enigma AI Engine" | awk '{print $2}' | xargs -r kill -9 2>/dev/null

# Clean up any lock files
rm -f /home/pi/Enigma AI Engine/*.lock 2>/dev/null
rm -f /home/pi/Enigma AI Engine/models/*.lock 2>/dev/null

echo "Done! All Enigma AI Engine processes terminated."
echo "You can now run: python run.py --gui"

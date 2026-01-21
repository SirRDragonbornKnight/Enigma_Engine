#!/bin/bash
# Force kill all ForgeAI processes

echo "Killing ForgeAI processes..."

# Kill by process name
pkill -9 -f "python.*run.py"
pkill -9 -f "ForgeAI"

# Kill any Python processes in the ForgeAI directory
ps aux | grep "[p]ython.*ForgeAI" | awk '{print $2}' | xargs -r kill -9 2>/dev/null

# Clean up any lock files
rm -f /home/pi/ForgeAI/*.lock 2>/dev/null
rm -f /home/pi/ForgeAI/models/*.lock 2>/dev/null

echo "Done! All ForgeAI processes terminated."
echo "You can now run: python run.py --gui"

PROTOCOL CONFIGURATION
======================

This folder contains configuration files for connecting Enigma AI to external systems.

HOW TO ADD A NEW CONNECTION
---------------------------

1. Create a JSON file in the appropriate subfolder:
   - game/     - For game connections (Unity, Godot, custom games)
   - robot/    - For robot connections (Arduino, ROS, GPIO)
   - api/      - For API connections (REST, WebSocket, MQTT)

2. Use this template:

   {
     "name": "My Connection",
     "description": "What this connects to",
     "protocol": "websocket",
     "host": "localhost",
     "port": 8080,
     "endpoint": "/",
     "enabled": true
   }

3. Available protocols:
   - websocket  : Real-time bidirectional (games, chat)
   - http       : REST API calls
   - tcp        : Raw TCP socket
   - udp        : Raw UDP socket (low latency)
   - serial     : USB/COM port (Arduino, robots)
   - ros        : ROS robotics framework
   - mqtt       : IoT message broker
   - osc        : Open Sound Control (music/art)
   - gpio       : Raspberry Pi GPIO pins

4. The AI will automatically load enabled configs on startup.

EXAMPLES
--------

Game (Unity WebSocket):
{
  "name": "Unity Game",
  "protocol": "websocket",
  "host": "localhost",
  "port": 8765,
  "endpoint": "/game"
}

Robot (Arduino Serial):
{
  "name": "Arduino Robot",
  "protocol": "serial",
  "port": "COM3",
  "baud": 115200
}

API (REST):
{
  "name": "Home Assistant",
  "protocol": "http",
  "host": "192.168.1.100",
  "port": 8123,
  "endpoint": "/api",
  "headers": {"Authorization": "Bearer YOUR_TOKEN"}
}

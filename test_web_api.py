#!/usr/bin/env python3
"""
Test Enigma AI Engine Web Server API Endpoints

This script tests all the REST API endpoints to ensure they work correctly.
"""

import sys
import time
import json
from pathlib import Path
import subprocess
import requests

# Add enigma_engine to path
sys.path.insert(0, str(Path(__file__).parent))

SERVER_URL = "http://localhost:8080"
TOKEN = None


def start_server():
    """Start the web server in background."""
    print("Starting web server...")
    process = subprocess.Popen(
        [sys.executable, "test_web_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start and capture token
    global TOKEN
    for _ in range(30):  # Wait up to 30 seconds
        line = process.stdout.readline()
        if "Authentication Token:" in line:
            TOKEN = line.split("Authentication Token:")[1].strip()
            print(f"✓ Server started (token: {TOKEN[:20]}...)")
            time.sleep(2)  # Give server a bit more time
            return process
        time.sleep(1)
    
    print("✗ Failed to start server")
    return None


def test_health():
    """Test health endpoint."""
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get(f"{SERVER_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        print("   ✓ Health check passed")
        return True
    except Exception as e:
        print(f"   ✗ Health check failed: {e}")
        return False


def test_info():
    """Test info endpoint."""
    print("\n2. Testing /api/info endpoint...")
    try:
        response = requests.get(f"{SERVER_URL}/api/info")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        print(f"   ✓ Info: {data['name']} v{data['version']}")
        return True
    except Exception as e:
        print(f"   ✗ Info check failed: {e}")
        return False


def test_auth_required():
    """Test that authentication is required."""
    print("\n3. Testing authentication requirement...")
    try:
        # Try without token - should fail
        response = requests.post(
            f"{SERVER_URL}/api/chat",
            json={"content": "Hello"}
        )
        assert response.status_code == 401
        print("   ✓ Authentication required (as expected)")
        
        # Try with token - should work
        response = requests.post(
            f"{SERVER_URL}/api/chat?token={TOKEN}",
            json={"content": "Hello"}
        )
        # May succeed or fail based on model availability, but shouldn't be 401
        assert response.status_code != 401
        print("   ✓ Token authentication works")
        return True
    except Exception as e:
        print(f"   ✗ Auth test failed: {e}")
        return False


def test_conversations():
    """Test conversations endpoints."""
    print("\n4. Testing /api/conversations...")
    try:
        response = requests.get(f"{SERVER_URL}/api/conversations?token={TOKEN}")
        assert response.status_code == 200
        data = response.json()
        assert "conversations" in data
        print(f"   ✓ Found {len(data['conversations'])} conversations")
        return True
    except Exception as e:
        print(f"   ✗ Conversations test failed: {e}")
        return False


def test_settings():
    """Test settings endpoints."""
    print("\n5. Testing /api/settings...")
    try:
        # Get settings
        response = requests.get(f"{SERVER_URL}/api/settings?token={TOKEN}")
        assert response.status_code == 200
        data = response.json()
        assert "settings" in data
        print(f"   ✓ Settings retrieved: {data['settings']}")
        
        # Update settings
        response = requests.put(
            f"{SERVER_URL}/api/settings?token={TOKEN}",
            json={"settings": {"temperature": 0.9}}
        )
        assert response.status_code == 200
        print("   ✓ Settings updated")
        return True
    except Exception as e:
        print(f"   ✗ Settings test failed: {e}")
        return False


def test_models():
    """Test models endpoints."""
    print("\n6. Testing /api/models...")
    try:
        response = requests.get(f"{SERVER_URL}/api/models?token={TOKEN}")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        print(f"   ✓ Found {len(data['models'])} models")
        return True
    except Exception as e:
        print(f"   ✗ Models test failed: {e}")
        return False


def test_stats():
    """Test stats endpoint."""
    print("\n7. Testing /api/stats...")
    try:
        response = requests.get(f"{SERVER_URL}/api/stats?token={TOKEN}")
        assert response.status_code == 200
        data = response.json()
        assert "cpu_percent" in data
        print(f"   ✓ Stats: CPU {data['cpu_percent']}%, Memory {data['memory_percent']}%")
        return True
    except Exception as e:
        print(f"   ✗ Stats test failed: {e}")
        return False


def test_modules():
    """Test modules endpoints."""
    print("\n8. Testing /api/modules...")
    try:
        response = requests.get(f"{SERVER_URL}/api/modules?token={TOKEN}")
        assert response.status_code == 200
        data = response.json()
        assert "modules" in data
        print(f"   ✓ Found {len(data['modules'])} modules")
        return True
    except Exception as e:
        print(f"   ✗ Modules test failed: {e}")
        return False


def test_static_files():
    """Test static files are served."""
    print("\n9. Testing static files...")
    try:
        response = requests.get(f"{SERVER_URL}/")
        assert response.status_code == 200
        assert "Enigma AI Engine" in response.text
        print("   ✓ Index page served")
        
        response = requests.get(f"{SERVER_URL}/static/styles.css")
        assert response.status_code == 200
        print("   ✓ CSS file served")
        
        response = requests.get(f"{SERVER_URL}/static/app.js")
        assert response.status_code == 200
        print("   ✓ JavaScript file served")
        
        return True
    except Exception as e:
        print(f"   ✗ Static files test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Enigma AI Engine Web Server API Tests")
    print("="*60)
    
    # Start server
    server_process = start_server()
    if not server_process:
        print("\n✗ Cannot start server, aborting tests")
        return
    
    try:
        # Run tests
        results = []
        results.append(("Health Check", test_health()))
        results.append(("Info Endpoint", test_info()))
        results.append(("Authentication", test_auth_required()))
        results.append(("Conversations", test_conversations()))
        results.append(("Settings", test_settings()))
        results.append(("Models", test_models()))
        results.append(("Stats", test_stats()))
        results.append(("Modules", test_modules()))
        results.append(("Static Files", test_static_files()))
        
        # Print summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status:8} - {name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n✅ All tests passed!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed")
        
    finally:
        # Stop server
        print("\nStopping server...")
        server_process.terminate()
        server_process.wait(timeout=5)
        print("Server stopped")


if __name__ == "__main__":
    main()

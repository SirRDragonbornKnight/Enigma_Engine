"""
Minimal remote client to call the local API server.

Uses urllib (built-in) to avoid requiring external dependencies.
"""
import json
import urllib.request
import urllib.error


class RemoteClient:
    """Simple client for connecting to an Enigma API server."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        """
        Args:
            base_url: Base URL of the AI Tester API server
        """
        self.base = base_url.rstrip("/")

    def generate(self, prompt: str, max_gen: int = 50, temperature: float = 1.0) -> str:
        """
        Generate text from the remote server.
        
        Args:
            prompt: Input prompt
            max_gen: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
            
        Raises:
            ConnectionError: If server is unreachable
            RuntimeError: If generation fails
        """
        data = json.dumps({
            "prompt": prompt,
            "max_gen": max_gen,
            "temperature": temperature
        }).encode()
        
        req = urllib.request.Request(
            f"{self.base}/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode())
                return result.get("text", "")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Could not connect to server: {e}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Server returned error: {e.code} - {e.reason}")
    
    def health(self) -> dict:
        """
        Check server health.
        
        Returns:
            Server status dict
        """
        req = urllib.request.Request(f"{self.base}/health")
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError):
            return {"status": "unreachable"}
    
    def info(self) -> dict:
        """
        Get server info.
        
        Returns:
            Server info dict
        """
        req = urllib.request.Request(f"{self.base}/info")
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError):
            return {}
    
    def is_available(self) -> bool:
        """Check if server is available."""
        return self.health().get("status") != "unreachable"

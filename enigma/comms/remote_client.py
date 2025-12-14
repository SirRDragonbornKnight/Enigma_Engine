"""
Minimal remote client to call the local API server.
"""
import requests

class RemoteClient:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base = base_url.rstrip("/")

    def generate(self, prompt, max_gen=50, temperature=1.0):
        r = requests.post(self.base + "/generate", json={"prompt": prompt, "max_gen": max_gen, "temperature": temperature})
        r.raise_for_status()
        return r.json().get("text", "")

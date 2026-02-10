"""
Plugin Signing and Verification

Cryptographic signing for plugin authenticity verification.
Supports RSA signatures with SHA-256 hashing.

FILE: enigma_engine/plugins/signing.py
TYPE: Plugin Security
MAIN CLASSES: PluginSigner, PluginVerifier, SignatureInfo
"""

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional cryptography imports
try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography not installed - plugin signing disabled")


@dataclass
class SignatureInfo:
    """Information about a plugin signature."""
    plugin_id: str
    plugin_version: str
    signer_id: str
    signer_name: str
    timestamp: str
    signature: str
    certificate_chain: list[str]
    
    # Verification result
    is_valid: bool = False
    verification_message: str = ""
    
    def to_dict(self) -> dict:
        return {
            "plugin_id": self.plugin_id,
            "plugin_version": self.plugin_version,
            "signer_id": self.signer_id,
            "signer_name": self.signer_name,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "certificate_chain": self.certificate_chain
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SignatureInfo':
        return cls(
            plugin_id=data["plugin_id"],
            plugin_version=data["plugin_version"],
            signer_id=data["signer_id"],
            signer_name=data["signer_name"],
            timestamp=data["timestamp"],
            signature=data["signature"],
            certificate_chain=data.get("certificate_chain", [])
        )


class KeyManager:
    """Manages cryptographic keys for signing."""
    
    def __init__(self, key_dir: Path):
        self.key_dir = Path(key_dir)
        self.key_dir.mkdir(parents=True, exist_ok=True)
        
        self._private_key = None
        self._public_key = None
        self._signer_id = None
        self._signer_name = None
    
    def generate_keypair(
        self,
        signer_id: str,
        signer_name: str,
        key_size: int = 2048
    ) -> tuple[str, str]:
        """
        Generate a new RSA keypair.
        
        Returns:
            Tuple of (public_key_pem, private_key_pem)
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography library required")
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        
        # Get public key
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Save keys
        private_path = self.key_dir / f"{signer_id}_private.pem"
        public_path = self.key_dir / f"{signer_id}_public.pem"
        
        with open(private_path, 'wb') as f:
            f.write(private_pem)
        os.chmod(private_path, 0o600)  # Restrict access
        
        with open(public_path, 'wb') as f:
            f.write(public_pem)
        
        # Save signer info
        info_path = self.key_dir / f"{signer_id}_info.json"
        with open(info_path, 'w') as f:
            json.dump({
                "signer_id": signer_id,
                "signer_name": signer_name,
                "created": datetime.now().isoformat(),
                "key_size": key_size
            }, f)
        
        logger.info(f"Generated keypair for {signer_name} ({signer_id})")
        return public_pem.decode(), private_pem.decode()
    
    def load_private_key(self, signer_id: str, password: bytes = None):
        """Load private key for signing."""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography library required")
        
        private_path = self.key_dir / f"{signer_id}_private.pem"
        if not private_path.exists():
            raise FileNotFoundError(f"Private key not found: {private_path}")
        
        with open(private_path, 'rb') as f:
            self._private_key = serialization.load_pem_private_key(
                f.read(),
                password=password,
                backend=default_backend()
            )
        
        # Load signer info
        info_path = self.key_dir / f"{signer_id}_info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
                self._signer_id = info["signer_id"]
                self._signer_name = info["signer_name"]
        else:
            self._signer_id = signer_id
            self._signer_name = signer_id
        
        logger.info(f"Loaded private key for {self._signer_name}")
    
    def load_public_key(self, signer_id: str):
        """Load public key for verification."""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography library required")
        
        public_path = self.key_dir / f"{signer_id}_public.pem"
        if not public_path.exists():
            raise FileNotFoundError(f"Public key not found: {public_path}")
        
        with open(public_path, 'rb') as f:
            self._public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )
        
        logger.info(f"Loaded public key for {signer_id}")
    
    def load_public_key_from_pem(self, pem_data: str):
        """Load public key from PEM string."""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography library required")
        
        self._public_key = serialization.load_pem_public_key(
            pem_data.encode(),
            backend=default_backend()
        )
    
    @property
    def private_key(self):
        return self._private_key
    
    @property
    def public_key(self):
        return self._public_key
    
    @property
    def signer_id(self) -> str:
        return self._signer_id or "unknown"
    
    @property
    def signer_name(self) -> str:
        return self._signer_name or "Unknown"


class PluginSigner:
    """Signs plugins for authenticity verification."""
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
    
    def compute_plugin_hash(self, plugin_path: Path) -> str:
        """
        Compute hash of plugin contents.
        
        Hashes all files in the plugin directory deterministically.
        """
        plugin_path = Path(plugin_path)
        
        if plugin_path.is_file():
            # Single file plugin
            with open(plugin_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        
        # Directory plugin - hash all files
        hasher = hashlib.sha256()
        
        # Get all files sorted for deterministic hashing
        all_files = sorted(plugin_path.rglob('*'))
        
        for file_path in all_files:
            if file_path.is_file():
                # Skip signature file itself
                if file_path.name == 'signature.json':
                    continue
                
                # Add relative path to hash
                rel_path = file_path.relative_to(plugin_path)
                hasher.update(str(rel_path).encode())
                
                # Add file contents
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def sign_plugin(
        self,
        plugin_path: Path,
        plugin_id: str,
        plugin_version: str
    ) -> SignatureInfo:
        """
        Sign a plugin.
        
        Args:
            plugin_path: Path to plugin file or directory
            plugin_id: Plugin identifier
            plugin_version: Plugin version string
        
        Returns:
            SignatureInfo with signature
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography library required")
        
        if not self.key_manager.private_key:
            raise ValueError("Private key not loaded")
        
        # Compute hash
        plugin_hash = self.compute_plugin_hash(plugin_path)
        
        # Create signing payload
        payload = {
            "plugin_id": plugin_id,
            "plugin_version": plugin_version,
            "hash": plugin_hash,
            "signer_id": self.key_manager.signer_id,
            "timestamp": datetime.now().isoformat()
        }
        
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        
        # Sign payload
        signature = self.key_manager.private_key.sign(
            payload_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        signature_b64 = base64.b64encode(signature).decode()
        
        sig_info = SignatureInfo(
            plugin_id=plugin_id,
            plugin_version=plugin_version,
            signer_id=self.key_manager.signer_id,
            signer_name=self.key_manager.signer_name,
            timestamp=payload["timestamp"],
            signature=signature_b64,
            certificate_chain=[],
            is_valid=True
        )
        
        # Save signature file
        plugin_path = Path(plugin_path)
        if plugin_path.is_dir():
            sig_path = plugin_path / "signature.json"
        else:
            sig_path = plugin_path.with_suffix('.sig.json')
        
        with open(sig_path, 'w') as f:
            json.dump(sig_info.to_dict(), f, indent=2)
        
        logger.info(f"Signed plugin {plugin_id} v{plugin_version}")
        return sig_info
    
    def sign_data(self, data: bytes) -> str:
        """Sign arbitrary data and return base64 signature."""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography library required")
        
        if not self.key_manager.private_key:
            raise ValueError("Private key not loaded")
        
        signature = self.key_manager.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode()


class PluginVerifier:
    """Verifies plugin signatures."""
    
    def __init__(self, trusted_keys_dir: Path):
        """
        Initialize verifier with trusted keys directory.
        
        Args:
            trusted_keys_dir: Directory containing trusted public keys
        """
        self.trusted_keys_dir = Path(trusted_keys_dir)
        self.trusted_keys_dir.mkdir(parents=True, exist_ok=True)
        
        self._trusted_signers: dict[str, bytes] = {}
        self._load_trusted_keys()
    
    def _load_trusted_keys(self):
        """Load all trusted public keys."""
        for key_file in self.trusted_keys_dir.glob("*_public.pem"):
            signer_id = key_file.stem.replace("_public", "")
            with open(key_file, 'rb') as f:
                self._trusted_signers[signer_id] = f.read()
            logger.debug(f"Loaded trusted key: {signer_id}")
        
        logger.info(f"Loaded {len(self._trusted_signers)} trusted signers")
    
    def add_trusted_key(self, signer_id: str, public_key_pem: str):
        """Add a trusted public key."""
        key_path = self.trusted_keys_dir / f"{signer_id}_public.pem"
        with open(key_path, 'w') as f:
            f.write(public_key_pem)
        
        self._trusted_signers[signer_id] = public_key_pem.encode()
        logger.info(f"Added trusted signer: {signer_id}")
    
    def remove_trusted_key(self, signer_id: str):
        """Remove a trusted key."""
        if signer_id in self._trusted_signers:
            del self._trusted_signers[signer_id]
        
        key_path = self.trusted_keys_dir / f"{signer_id}_public.pem"
        if key_path.exists():
            key_path.unlink()
        
        logger.info(f"Removed trusted signer: {signer_id}")
    
    def is_signer_trusted(self, signer_id: str) -> bool:
        """Check if signer is trusted."""
        return signer_id in self._trusted_signers
    
    def compute_plugin_hash(self, plugin_path: Path) -> str:
        """Compute hash of plugin contents (same as signer)."""
        plugin_path = Path(plugin_path)
        
        if plugin_path.is_file():
            with open(plugin_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        
        hasher = hashlib.sha256()
        all_files = sorted(plugin_path.rglob('*'))
        
        for file_path in all_files:
            if file_path.is_file():
                if file_path.name == 'signature.json':
                    continue
                
                rel_path = file_path.relative_to(plugin_path)
                hasher.update(str(rel_path).encode())
                
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def verify_plugin(self, plugin_path: Path) -> SignatureInfo:
        """
        Verify a plugin's signature.
        
        Args:
            plugin_path: Path to plugin file or directory
        
        Returns:
            SignatureInfo with verification result
        """
        plugin_path = Path(plugin_path)
        
        # Find signature file
        if plugin_path.is_dir():
            sig_path = plugin_path / "signature.json"
        else:
            sig_path = plugin_path.with_suffix('.sig.json')
        
        if not sig_path.exists():
            return SignatureInfo(
                plugin_id="unknown",
                plugin_version="unknown",
                signer_id="",
                signer_name="",
                timestamp="",
                signature="",
                certificate_chain=[],
                is_valid=False,
                verification_message="No signature file found"
            )
        
        # Load signature
        with open(sig_path) as f:
            sig_data = json.load(f)
        
        sig_info = SignatureInfo.from_dict(sig_data)
        
        # Check if signer is trusted
        if not self.is_signer_trusted(sig_info.signer_id):
            sig_info.is_valid = False
            sig_info.verification_message = f"Signer not trusted: {sig_info.signer_id}"
            return sig_info
        
        if not CRYPTO_AVAILABLE:
            sig_info.is_valid = False
            sig_info.verification_message = "cryptography library not available"
            return sig_info
        
        # Load public key
        public_key_pem = self._trusted_signers[sig_info.signer_id]
        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )
        
        # Compute current hash
        current_hash = self.compute_plugin_hash(plugin_path)
        
        # Recreate signing payload
        payload = {
            "plugin_id": sig_info.plugin_id,
            "plugin_version": sig_info.plugin_version,
            "hash": current_hash,
            "signer_id": sig_info.signer_id,
            "timestamp": sig_info.timestamp
        }
        
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        
        # Verify signature
        try:
            signature = base64.b64decode(sig_info.signature)
            public_key.verify(
                signature,
                payload_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            sig_info.is_valid = True
            sig_info.verification_message = "Signature valid"
            logger.info(f"Verified plugin {sig_info.plugin_id} v{sig_info.plugin_version}")
            
        except InvalidSignature:
            sig_info.is_valid = False
            sig_info.verification_message = "Invalid signature - plugin may be tampered"
            logger.warning(f"Invalid signature for {sig_info.plugin_id}")
            
        except Exception as e:
            sig_info.is_valid = False
            sig_info.verification_message = f"Verification error: {e}"
            logger.error(f"Verification error: {e}")
        
        return sig_info
    
    def verify_data(
        self,
        data: bytes,
        signature_b64: str,
        signer_id: str
    ) -> bool:
        """Verify signature on arbitrary data."""
        if not CRYPTO_AVAILABLE:
            return False
        
        if not self.is_signer_trusted(signer_id):
            return False
        
        public_key_pem = self._trusted_signers[signer_id]
        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )
        
        try:
            signature = base64.b64decode(signature_b64)
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


# Convenience functions
def get_key_manager(key_dir: str = None) -> KeyManager:
    """Get or create key manager."""
    if key_dir is None:
        key_dir = Path.home() / ".enigma_engine" / "keys"
    return KeyManager(Path(key_dir))


def get_verifier(trusted_keys_dir: str = None) -> PluginVerifier:
    """Get or create plugin verifier."""
    if trusted_keys_dir is None:
        trusted_keys_dir = Path.home() / ".enigma_engine" / "trusted_keys"
    return PluginVerifier(Path(trusted_keys_dir))


__all__ = [
    'PluginSigner',
    'PluginVerifier',
    'SignatureInfo',
    'KeyManager',
    'get_key_manager',
    'get_verifier'
]

"""
Network Encryption (TLS)

TLS/SSL configuration for secure API communications.
Handles certificate management and secure connections.

FILE: enigma_engine/security/tls.py
TYPE: Security
MAIN CLASSES: TLSConfig, CertificateManager, SecureServer
"""

import ipaddress
import logging
import ssl
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TLSVersion(Enum):
    """TLS protocol versions."""
    TLS_1_2 = "TLSv1.2"
    TLS_1_3 = "TLSv1.3"


class CertificateType(Enum):
    """Types of certificates."""
    SELF_SIGNED = "self_signed"
    LETS_ENCRYPT = "lets_encrypt"
    CUSTOM = "custom"


@dataclass
class CertificateInfo:
    """Information about a certificate."""
    path: Path
    key_path: Path
    cert_type: CertificateType
    domain: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    issuer: str = ""
    thumbprint: str = ""
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    @property
    def days_until_expiry(self) -> int:
        return int((self.expires_at - time.time()) / 86400)


@dataclass
class TLSConfig:
    """TLS configuration."""
    enabled: bool = True
    min_version: TLSVersion = TLSVersion.TLS_1_2
    cert_path: Optional[Path] = None
    key_path: Optional[Path] = None
    ca_path: Optional[Path] = None
    verify_client: bool = False
    ciphers: str = ""  # OpenSSL cipher string
    session_timeout: int = 3600
    
    # HSTS settings
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = False


class CertificateManager:
    """Manages TLS certificates."""
    
    def __init__(self, certs_dir: Path):
        """
        Initialize certificate manager.
        
        Args:
            certs_dir: Directory for storing certificates
        """
        self._certs_dir = Path(certs_dir)
        self._certs_dir.mkdir(parents=True, exist_ok=True)
        
        self._certificates: dict[str, CertificateInfo] = {}
    
    def generate_self_signed(self,
                             domain: str = "localhost",
                             days: int = 365,
                             key_size: int = 2048) -> CertificateInfo:
        """
        Generate a self-signed certificate.
        
        Args:
            domain: Domain name
            days: Validity period
            key_size: RSA key size
            
        Returns:
            Certificate info
        """
        cert_path = self._certs_dir / f"{domain}.crt"
        key_path = self._certs_dir / f"{domain}.key"
        
        # Check if openssl is available
        try:
            # Generate private key
            subprocess.run([
                "openssl", "genrsa",
                "-out", str(key_path),
                str(key_size)
            ], check=True, capture_output=True, timeout=60)
            
            # Generate self-signed certificate
            subprocess.run([
                "openssl", "req",
                "-new", "-x509",
                "-key", str(key_path),
                "-out", str(cert_path),
                "-days", str(days),
                "-subj", f"/CN={domain}"
            ], check=True, capture_output=True, timeout=60)
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"OpenSSL not available, using Python fallback: {e}")
            self._generate_python_cert(cert_path, key_path, domain, days)
        
        cert_info = CertificateInfo(
            path=cert_path,
            key_path=key_path,
            cert_type=CertificateType.SELF_SIGNED,
            domain=domain,
            expires_at=time.time() + days * 86400,
            issuer="Self-Signed"
        )
        
        self._certificates[domain] = cert_info
        return cert_info
    
    def _generate_python_cert(self,
                              cert_path: Path,
                              key_path: Path,
                              domain: str,
                              days: int):
        """Generate certificate using Python (requires cryptography package)."""
        try:
            import datetime

            from cryptography import x509
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.x509.oid import NameOID

            # Generate key
            key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            # Write key
            with open(key_path, 'wb') as f:
                f.write(key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Generate certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, domain),
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.datetime.utcnow()
            ).not_valid_after(
                datetime.datetime.utcnow() + datetime.timedelta(days=days)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(domain),
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))
                ]),
                critical=False
            ).sign(key, hashes.SHA256(), default_backend())
            
            # Write certificate
            with open(cert_path, 'wb') as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
                
        except ImportError:
            # Fallback: create minimal placeholder
            logger.error("Neither openssl nor cryptography available")
            raise RuntimeError("Cannot generate certificates: install openssl or cryptography package")
    
    def get_certificate(self, domain: str) -> Optional[CertificateInfo]:
        """Get certificate info for a domain."""
        return self._certificates.get(domain)
    
    def list_certificates(self) -> list[CertificateInfo]:
        """List all managed certificates."""
        return list(self._certificates.values())
    
    def check_expiring(self, days_threshold: int = 30) -> list[CertificateInfo]:
        """Get certificates expiring within threshold days."""
        return [
            cert for cert in self._certificates.values()
            if cert.days_until_expiry <= days_threshold
        ]
    
    def renew_certificate(self, domain: str) -> Optional[CertificateInfo]:
        """Renew a certificate."""
        existing = self._certificates.get(domain)
        if not existing:
            return None
        
        if existing.cert_type == CertificateType.SELF_SIGNED:
            return self.generate_self_signed(domain)
        
        # For Let's Encrypt or custom, would need additional implementation
        logger.warning(f"Renewal not implemented for {existing.cert_type.value}")
        return None


class SecureServer:
    """Wrapper for creating secure server connections."""
    
    def __init__(self, config: TLSConfig):
        """
        Initialize secure server.
        
        Args:
            config: TLS configuration
        """
        self._config = config
        self._ssl_context: Optional[ssl.SSLContext] = None
    
    def create_ssl_context(self) -> ssl.SSLContext:
        """
        Create an SSL context with the configured settings.
        
        Returns:
            Configured SSL context
        """
        if not self._config.enabled:
            raise ValueError("TLS is not enabled")
        
        # Select protocol
        if self._config.min_version == TLSVersion.TLS_1_3:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_3
        else:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # Load certificate
        if self._config.cert_path and self._config.key_path:
            context.load_cert_chain(
                certfile=str(self._config.cert_path),
                keyfile=str(self._config.key_path)
            )
        
        # Load CA for client verification
        if self._config.ca_path:
            context.load_verify_locations(str(self._config.ca_path))
        
        # Client verification
        if self._config.verify_client:
            context.verify_mode = ssl.CERT_REQUIRED
        else:
            context.verify_mode = ssl.CERT_NONE
        
        # Set ciphers
        if self._config.ciphers:
            context.set_ciphers(self._config.ciphers)
        else:
            # Use secure default ciphers
            context.set_ciphers(self._get_secure_ciphers())
        
        # Security options
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        
        self._ssl_context = context
        return context
    
    def _get_secure_ciphers(self) -> str:
        """Get a secure cipher string."""
        return (
            "ECDHE+AESGCM:"
            "DHE+AESGCM:"
            "ECDHE+CHACHA20:"
            "DHE+CHACHA20:"
            "!aNULL:"
            "!MD5:"
            "!DSS"
        )
    
    def get_hsts_header(self) -> str:
        """
        Get HSTS header value.
        
        Returns:
            HSTS header string
        """
        if not self._config.hsts_enabled:
            return ""
        
        parts = [f"max-age={self._config.hsts_max_age}"]
        
        if self._config.hsts_include_subdomains:
            parts.append("includeSubDomains")
        
        if self._config.hsts_preload:
            parts.append("preload")
        
        return "; ".join(parts)
    
    def wrap_socket(self, socket) -> ssl.SSLSocket:
        """
        Wrap a socket with TLS.
        
        Args:
            socket: Socket to wrap
            
        Returns:
            Wrapped SSL socket
        """
        if not self._ssl_context:
            self.create_ssl_context()
        
        return self._ssl_context.wrap_socket(socket, server_side=True)


def create_secure_flask_app(app, config: TLSConfig):
    """
    Configure a Flask app for TLS.
    
    Args:
        app: Flask application
        config: TLS configuration
    """
    @app.after_request
    def add_security_headers(response):
        # HSTS
        server = SecureServer(config)
        hsts = server.get_hsts_header()
        if hsts:
            response.headers['Strict-Transport-Security'] = hsts
        
        # Other security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        return response
    
    return app


def quick_setup_tls(domain: str = "localhost", certs_dir: Path = None) -> tuple[TLSConfig, CertificateInfo]:
    """
    Quick setup for TLS with self-signed certificate.
    
    Args:
        domain: Domain name
        certs_dir: Certificate directory
        
    Returns:
        Tuple of (TLSConfig, CertificateInfo)
    """
    certs_dir = certs_dir or Path("certs")
    
    manager = CertificateManager(certs_dir)
    cert = manager.generate_self_signed(domain)
    
    config = TLSConfig(
        enabled=True,
        cert_path=cert.path,
        key_path=cert.key_path
    )
    
    return config, cert


__all__ = [
    'TLSConfig',
    'TLSVersion',
    'CertificateManager',
    'CertificateInfo',
    'CertificateType',
    'SecureServer',
    'create_secure_flask_app',
    'quick_setup_tls'
]

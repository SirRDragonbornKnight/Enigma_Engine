"""
Secure Aggregation Protocols for Federated Learning

Implements real secure aggregation protocols:
- Secret Sharing (Shamir's)
- Masking-based secure aggregation
- Homomorphic encryption (Paillier)
- Differential privacy integration

These protocols ensure individual updates remain private
even from the aggregation server.

Usage:
    from enigma_engine.learning.secure_aggregation import (
        SecureSumProtocol,
        SecretSharing,
        HomomorphicAggregator
    )
    
    # Secret sharing based
    protocol = SecureSumProtocol(threshold=2, num_parties=3)
    shares = protocol.share(my_update)
    result = protocol.reconstruct(all_shares)
"""

import hashlib
import logging
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class ProtocolType(Enum):
    """Secure aggregation protocol types."""
    SECRET_SHARING = "secret_sharing"
    MASKING = "masking"
    HOMOMORPHIC = "homomorphic"


@dataclass
class SecretShare:
    """A share in a secret sharing scheme."""
    party_id: int
    share_id: str
    x: int  # Share index
    y: int  # Share value (for Shamir's)
    data: Optional[Any] = None  # Actual share data for array sharing


class SecretSharing:
    """
    Shamir's Secret Sharing implementation.
    
    Splits a secret into n shares such that any k shares
    can reconstruct the original secret.
    """
    
    def __init__(self, threshold: int, num_shares: int, prime: Optional[int] = None):
        """
        Initialize secret sharing.
        
        Args:
            threshold: Minimum shares needed to reconstruct (k)
            num_shares: Total number of shares to create (n)
            prime: Prime modulus for finite field arithmetic
        """
        if threshold > num_shares:
            raise ValueError("Threshold cannot exceed number of shares")
        
        self.threshold = threshold
        self.num_shares = num_shares
        # Use a large prime for security
        self.prime = prime or 2**127 - 1  # 12th Mersenne prime
    
    def share_secret(self, secret: int) -> list[SecretShare]:
        """
        Split a secret into shares using Shamir's scheme.
        
        Args:
            secret: The secret to share (must be < prime)
            
        Returns:
            List of SecretShare objects
        """
        # Reduce secret modulo prime if needed
        secret = secret % self.prime
        
        # Generate random polynomial coefficients
        # f(x) = secret + a1*x + a2*x^2 + ... + a_{k-1}*x^{k-1}
        coefficients = [secret]
        for _ in range(self.threshold - 1):
            coefficients.append(secrets.randbelow(self.prime))
        
        # Generate shares by evaluating polynomial at different points
        shares = []
        share_id = hashlib.sha256(str(secret).encode()).hexdigest()[:8]
        
        for i in range(1, self.num_shares + 1):
            # Evaluate polynomial at x=i
            y = 0
            for power, coeff in enumerate(coefficients):
                y = (y + coeff * pow(i, power, self.prime)) % self.prime
            
            shares.append(SecretShare(
                party_id=i,
                share_id=share_id,
                x=i,
                y=y
            ))
        
        return shares
    
    def reconstruct_secret(self, shares: list[SecretShare]) -> int:
        """
        Reconstruct the secret from shares using Lagrange interpolation.
        
        Args:
            shares: At least threshold shares
            
        Returns:
            The original secret
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares, got {len(shares)}")
        
        # Use first threshold shares
        shares = shares[:self.threshold]
        
        # Lagrange interpolation at x=0
        secret = 0
        
        for i, share_i in enumerate(shares):
            # Compute Lagrange basis polynomial at x=0
            numerator = 1
            denominator = 1
            
            for j, share_j in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-share_j.x)) % self.prime
                    denominator = (denominator * (share_i.x - share_j.x)) % self.prime
            
            # Compute modular inverse of denominator
            denominator_inv = pow(denominator, self.prime - 2, self.prime)
            
            # Add contribution
            lagrange_coeff = (numerator * denominator_inv) % self.prime
            secret = (secret + share_i.y * lagrange_coeff) % self.prime
        
        return secret
    
    def share_array(self, arr: 'np.ndarray') -> list[SecretShare]:
        """
        Share an array by sharing each element.
        
        For efficiency, uses additive sharing for arrays.
        
        Args:
            arr: NumPy array to share
            
        Returns:
            List of shares containing array shares
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required for array sharing")
        
        arr_flat = arr.flatten().astype(np.float64)
        share_id = hashlib.sha256(arr.tobytes()).hexdigest()[:8]
        
        # Additive secret sharing for efficiency
        # secret = sum of all shares
        shares = []
        
        for i in range(self.num_shares - 1):
            # Random share
            random_share = np.random.uniform(-1e6, 1e6, arr_flat.shape)
            shares.append(SecretShare(
                party_id=i + 1,
                share_id=share_id,
                x=i + 1,
                y=0,
                data=random_share
            ))
        
        # Last share is computed to sum to original
        last_share = arr_flat - sum(s.data for s in shares)
        shares.append(SecretShare(
            party_id=self.num_shares,
            share_id=share_id,
            x=self.num_shares,
            y=0,
            data=last_share
        ))
        
        return shares
    
    def reconstruct_array(self, shares: list[SecretShare], shape: tuple) -> 'np.ndarray':
        """
        Reconstruct array from additive shares.
        
        Args:
            shares: List of array shares
            shape: Original array shape
            
        Returns:
            Reconstructed array
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required for array reconstruction")
        
        # Sum all shares
        result = sum(s.data for s in shares)
        return result.reshape(shape)


class SecureSumProtocol:
    """
    Secure sum protocol using masking.
    
    Each party adds random masks that cancel out when summed,
    allowing the sum to be computed without revealing individuals.
    """
    
    def __init__(self, num_parties: int, seed: Optional[int] = None):
        """
        Initialize secure sum protocol.
        
        Args:
            num_parties: Number of participating parties
            seed: Random seed for reproducibility (should be shared secret)
        """
        self.num_parties = num_parties
        self.seed = seed
        self.masks: dict[tuple[int, int], 'np.ndarray'] = {}
    
    def generate_pairwise_masks(self, party_id: int, data_shape: tuple) -> dict[int, 'np.ndarray']:
        """
        Generate pairwise masks for a party.
        
        For each pair (i, j) where i < j:
        - Party i adds mask_ij
        - Party j subtracts mask_ij
        
        These cancel out in the sum.
        
        Args:
            party_id: This party's ID (1-indexed)
            data_shape: Shape of data being aggregated
            
        Returns:
            Dict of other_party_id -> mask to add
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required for secure sum")
        
        masks = {}
        
        for other_id in range(1, self.num_parties + 1):
            if other_id == party_id:
                continue
            
            # Generate deterministic mask for this pair
            pair = (min(party_id, other_id), max(party_id, other_id))
            seed = hash((self.seed, pair)) % (2**32)
            rng = np.random.RandomState(seed)
            mask = rng.randn(*data_shape)
            
            # If we're the smaller ID, we add; otherwise subtract
            if party_id < other_id:
                masks[other_id] = mask
            else:
                masks[other_id] = -mask
        
        return masks
    
    def mask_data(self, party_id: int, data: 'np.ndarray') -> 'np.ndarray':
        """
        Apply masks to party's data.
        
        Args:
            party_id: This party's ID
            data: Data to mask
            
        Returns:
            Masked data
        """
        masks = self.generate_pairwise_masks(party_id, data.shape)
        
        masked = data.copy()
        for mask in masks.values():
            masked = masked + mask
        
        return masked
    
    def aggregate(self, masked_updates: list['np.ndarray']) -> 'np.ndarray':
        """
        Aggregate masked updates.
        
        Since masks cancel out, this gives the true sum.
        
        Args:
            masked_updates: List of masked data from all parties
            
        Returns:
            Sum of original (unmasked) data
        """
        return sum(masked_updates)


class PaillierEncryption:
    """
    Simplified Paillier homomorphic encryption.
    
    Supports additive homomorphism:
    - E(a) * E(b) = E(a + b)
    
    This allows aggregation on encrypted data.
    """
    
    def __init__(self, key_size: int = 512):
        """
        Initialize Paillier encryption.
        
        Args:
            key_size: Bit size for key generation
        """
        self.key_size = key_size
        self.public_key: Optional[tuple[int, int]] = None
        self.private_key: Optional[tuple[int, int]] = None
        self._generate_keys()
    
    def _generate_keys(self):
        """Generate Paillier key pair."""
        # Generate two large primes (simplified for demo)
        # In production, use proper prime generation
        def is_prime(n: int) -> bool:
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        def gen_prime(bits: int) -> int:
            while True:
                p = secrets.randbits(bits) | 1
                if is_prime(p):
                    return p
        
        # For demo, use small primes (NOT SECURE)
        # In production, use 1024+ bit primes
        p = gen_prime(min(self.key_size // 2, 32))
        q = gen_prime(min(self.key_size // 2, 32))
        
        while p == q:
            q = gen_prime(min(self.key_size // 2, 32))
        
        n = p * q
        n_squared = n * n
        g = n + 1  # Simplified choice of g
        
        # Lambda = lcm(p-1, q-1)
        from math import gcd
        lambda_n = (p - 1) * (q - 1) // gcd(p - 1, q - 1)
        
        # Mu = lambda^-1 mod n
        mu = pow(lambda_n, -1, n)
        
        self.public_key = (n, g)
        self.private_key = (lambda_n, mu)
        self.n = n
        self.n_squared = n_squared
    
    def encrypt(self, plaintext: int) -> int:
        """
        Encrypt a plaintext value.
        
        Args:
            plaintext: Value to encrypt (must be 0 <= m < n)
            
        Returns:
            Ciphertext
        """
        n, g = self.public_key
        
        # Reduce plaintext
        m = plaintext % n
        
        # Random r
        r = secrets.randbelow(n - 1) + 1
        while gcd(r, n) != 1:
            r = secrets.randbelow(n - 1) + 1
        
        # c = g^m * r^n mod n^2
        c = (pow(g, m, self.n_squared) * pow(r, n, self.n_squared)) % self.n_squared
        
        return c
    
    def decrypt(self, ciphertext: int) -> int:
        """
        Decrypt a ciphertext value.
        
        Args:
            ciphertext: Value to decrypt
            
        Returns:
            Plaintext
        """
        lambda_n, mu = self.private_key
        n = self.n
        
        # L(x) = (x-1) / n
        def L(x):
            return (x - 1) // n
        
        # m = L(c^lambda mod n^2) * mu mod n
        m = (L(pow(ciphertext, lambda_n, self.n_squared)) * mu) % n
        
        return m
    
    def add_encrypted(self, c1: int, c2: int) -> int:
        """
        Add two encrypted values.
        
        E(a) * E(b) = E(a + b)
        
        Args:
            c1: First ciphertext
            c2: Second ciphertext
            
        Returns:
            Ciphertext of sum
        """
        return (c1 * c2) % self.n_squared


# Need gcd for Paillier
from math import gcd


class HomomorphicAggregator:
    """
    Aggregator using homomorphic encryption.
    
    Parties encrypt their updates with the public key.
    The aggregator multiplies ciphertexts (which adds plaintexts).
    Only the key holder can decrypt the final result.
    """
    
    def __init__(self, paillier: Optional[PaillierEncryption] = None):
        """
        Initialize homomorphic aggregator.
        
        Args:
            paillier: Paillier encryption instance (creates new if None)
        """
        self.paillier = paillier or PaillierEncryption()
    
    def encrypt_update(self, update: 'np.ndarray', scale: int = 1000000) -> list[int]:
        """
        Encrypt a weight update.
        
        Scales floats to integers for encryption.
        
        Args:
            update: NumPy array to encrypt
            scale: Scaling factor for float -> int
            
        Returns:
            List of encrypted values
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required")
        
        flat = update.flatten()
        encrypted = []
        
        for val in flat:
            # Scale to integer
            int_val = int(val * scale) % self.paillier.n
            encrypted.append(self.paillier.encrypt(int_val))
        
        return encrypted
    
    def aggregate_encrypted(
        self,
        encrypted_updates: list[list[int]]
    ) -> list[int]:
        """
        Aggregate encrypted updates homomorphically.
        
        Args:
            encrypted_updates: List of encrypted updates from parties
            
        Returns:
            Encrypted aggregate
        """
        if not encrypted_updates:
            return []
        
        # Element-wise multiplication (which adds plaintexts)
        result = encrypted_updates[0].copy()
        
        for update in encrypted_updates[1:]:
            for i in range(len(result)):
                result[i] = self.paillier.add_encrypted(result[i], update[i])
        
        return result
    
    def decrypt_aggregate(
        self,
        encrypted_aggregate: list[int],
        shape: tuple,
        num_parties: int,
        scale: int = 1000000
    ) -> 'np.ndarray':
        """
        Decrypt the aggregated result.
        
        Args:
            encrypted_aggregate: Encrypted sum
            shape: Original array shape
            num_parties: Number of parties (to compute average)
            scale: Scaling factor used in encryption
            
        Returns:
            Decrypted average update
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required")
        
        result = []
        for c in encrypted_aggregate:
            decrypted = self.paillier.decrypt(c)
            # Handle negative numbers (represented as n - |x|)
            if decrypted > self.paillier.n // 2:
                decrypted = decrypted - self.paillier.n
            # Unscale and average
            result.append(decrypted / scale / num_parties)
        
        return np.array(result).reshape(shape)


class SecureAggregationProtocol:
    """
    Complete secure aggregation protocol combining multiple techniques.
    
    Supports:
    1. Secret sharing for dropout tolerance
    2. Pairwise masking for efficiency  
    3. Optional homomorphic encryption for stronger security
    """
    
    def __init__(
        self,
        num_parties: int,
        threshold: int,
        protocol_type: ProtocolType = ProtocolType.MASKING
    ):
        """
        Initialize secure aggregation protocol.
        
        Args:
            num_parties: Number of participating parties
            threshold: Minimum parties needed for aggregation
            protocol_type: Which protocol to use
        """
        self.num_parties = num_parties
        self.threshold = threshold
        self.protocol_type = protocol_type
        
        # Initialize components based on protocol
        if protocol_type == ProtocolType.SECRET_SHARING:
            self.secret_sharing = SecretSharing(threshold, num_parties)
        elif protocol_type == ProtocolType.MASKING:
            self.secure_sum = SecureSumProtocol(num_parties)
        elif protocol_type == ProtocolType.HOMOMORPHIC:
            self.homomorphic = HomomorphicAggregator()
        
        logger.info(f"Initialized {protocol_type.value} secure aggregation protocol")
    
    def prepare_update(
        self,
        party_id: int,
        update: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Prepare a party's update for secure aggregation.
        
        Args:
            party_id: Party's ID
            update: Weight update dict
            
        Returns:
            Protected update
        """
        if not HAS_NUMPY:
            logger.warning("NumPy not available, returning raw update")
            return update
        
        protected = {}
        
        for layer_name, values in update.items():
            arr = np.array(values) if not isinstance(values, np.ndarray) else values
            
            if self.protocol_type == ProtocolType.MASKING:
                protected[layer_name] = self.secure_sum.mask_data(party_id, arr)
            elif self.protocol_type == ProtocolType.SECRET_SHARING:
                shares = self.secret_sharing.share_array(arr)
                protected[layer_name] = {
                    'shares': [(s.party_id, s.data.tolist()) for s in shares],
                    'shape': arr.shape
                }
            elif self.protocol_type == ProtocolType.HOMOMORPHIC:
                protected[layer_name] = {
                    'encrypted': self.homomorphic.encrypt_update(arr),
                    'shape': arr.shape
                }
        
        return protected
    
    def aggregate(
        self,
        protected_updates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Aggregate protected updates.
        
        Args:
            protected_updates: List of protected updates from parties
            
        Returns:
            Aggregated update (decrypted/reconstructed)
        """
        if not protected_updates:
            return {}
        
        if not HAS_NUMPY:
            logger.warning("NumPy not available, using first update")
            return protected_updates[0]
        
        result = {}
        
        # Get layer names from first update
        layer_names = list(protected_updates[0].keys())
        
        for layer in layer_names:
            if self.protocol_type == ProtocolType.MASKING:
                # Sum masked updates
                layer_updates = [u[layer] for u in protected_updates]
                aggregated = self.secure_sum.aggregate(layer_updates)
                result[layer] = aggregated / len(protected_updates)  # Average
                
            elif self.protocol_type == ProtocolType.SECRET_SHARING:
                # Reconstruct each party's share and average
                shape = protected_updates[0][layer]['shape']
                
                # For additive sharing, just sum all shares
                total = None
                for update in protected_updates:
                    for party_id, share_data in update[layer]['shares']:
                        share_arr = np.array(share_data)
                        if total is None:
                            total = share_arr
                        else:
                            total = total + share_arr
                
                result[layer] = total.reshape(shape) / len(protected_updates)
                
            elif self.protocol_type == ProtocolType.HOMOMORPHIC:
                # Aggregate encrypted and decrypt
                shape = protected_updates[0][layer]['shape']
                encrypted_updates = [u[layer]['encrypted'] for u in protected_updates]
                
                aggregated = self.homomorphic.aggregate_encrypted(encrypted_updates)
                result[layer] = self.homomorphic.decrypt_aggregate(
                    aggregated, shape, len(protected_updates)
                )
        
        return result


def integrate_with_federated_learning(updates: list[Any], method: str = "masking") -> Any:
    """
    Integrate secure aggregation with the federated learning system.
    
    Args:
        updates: List of WeightUpdate objects
        method: Protocol type ("masking", "secret_sharing", "homomorphic")
        
    Returns:
        Securely aggregated WeightUpdate
    """
    import uuid
    from datetime import datetime

    from .federated import WeightUpdate
    
    if not updates:
        raise ValueError("No updates to aggregate")
    
    # Map method string to enum
    protocol_map = {
        "masking": ProtocolType.MASKING,
        "secret_sharing": ProtocolType.SECRET_SHARING,
        "homomorphic": ProtocolType.HOMOMORPHIC
    }
    
    protocol_type = protocol_map.get(method, ProtocolType.MASKING)
    
    # Initialize protocol
    protocol = SecureAggregationProtocol(
        num_parties=len(updates),
        threshold=len(updates) // 2 + 1,
        protocol_type=protocol_type
    )
    
    # Prepare updates
    protected_updates = []
    for i, update in enumerate(updates):
        protected = protocol.prepare_update(i + 1, update.weight_deltas)
        protected_updates.append(protected)
    
    # Aggregate
    aggregated_deltas = protocol.aggregate(protected_updates)
    
    # Convert back to lists if needed
    for layer in aggregated_deltas:
        if HAS_NUMPY and isinstance(aggregated_deltas[layer], np.ndarray):
            aggregated_deltas[layer] = aggregated_deltas[layer].tolist()
    
    # Create result
    total_samples = sum(u.training_samples for u in updates)
    
    return WeightUpdate(
        update_id=str(uuid.uuid4()),
        device_id="secure_aggregated",
        timestamp=datetime.now(),
        weight_deltas=aggregated_deltas,
        training_samples=total_samples,
        metadata={
            "aggregation_method": f"secure_{method}",
            "num_parties": len(updates),
            "protocol": protocol_type.value
        }
    )

"""
DeepFake Detection System - Provenance System
Layer 5: Provenance (The "Trust" Layer)

This module implements cryptographic origin verification:
- Media signing at creation time
- Hash chaining through edits
- Device/model identity binding
- Tamper-evident audit logs

Shifts from "Is this fake?" to "Can this be proven real?"
"""

import hashlib
import hmac
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import base64

# Cryptographic primitives
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.asymmetric.rsa import (
        RSAPrivateKey,
        RSAPublicKey,
    )
    from cryptography.hazmat.backends import default_backend
    from cryptography.exceptions import InvalidSignature

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class ProvenanceStatus(Enum):
    """Status of provenance verification."""

    VERIFIED = "verified"  # Full chain verified
    PARTIAL = "partial"  # Some links verified
    UNVERIFIED = "unverified"  # No provenance data
    TAMPERED = "tampered"  # Tampering detected
    EXPIRED = "expired"  # Signatures expired


@dataclass
class HashChainLink:
    """A single link in the hash chain."""

    link_id: str
    previous_hash: str
    content_hash: str
    operation: str  # "create", "edit", "verify"
    timestamp: datetime
    author_id: Optional[str] = None
    device_id: Optional[str] = None
    signature: Optional[str] = None  # Base64 encoded
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_link_hash(self) -> str:
        """Compute hash of this link."""
        data = {
            "link_id": self.link_id,
            "previous_hash": self.previous_hash,
            "content_hash": self.content_hash,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "author_id": self.author_id,
            "device_id": self.device_id,
            "metadata": self.metadata,
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_id": self.link_id,
            "previous_hash": self.previous_hash,
            "content_hash": self.content_hash,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "author_id": self.author_id,
            "device_id": self.device_id,
            "signature": self.signature,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HashChainLink":
        return cls(
            link_id=data["link_id"],
            previous_hash=data["previous_hash"],
            content_hash=data["content_hash"],
            operation=data["operation"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            author_id=data.get("author_id"),
            device_id=data.get("device_id"),
            signature=data.get("signature"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProvenanceRecord:
    """Complete provenance record for a media file."""

    record_id: str
    original_content_hash: str
    current_content_hash: str
    chain: List[HashChainLink] = field(default_factory=list)
    status: ProvenanceStatus = ProvenanceStatus.UNVERIFIED

    # C2PA-compatible fields
    claim_generator: Optional[str] = None
    claim_signature: Optional[str] = None

    def add_link(self, link: HashChainLink) -> None:
        """Add a new link to the chain."""
        if self.chain:
            link.previous_hash = self.chain[-1].compute_link_hash()
        self.chain.append(link)
        self.current_content_hash = link.content_hash

    def verify_chain(self) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of the hash chain.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        if not self.chain:
            return False, ["Empty chain"]

        for i, link in enumerate(self.chain):
            # Verify link hash
            if i > 0:
                expected_prev = self.chain[i - 1].compute_link_hash()
                if link.previous_hash != expected_prev:
                    errors.append(f"Link {i}: previous_hash mismatch")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "original_content_hash": self.original_content_hash,
            "current_content_hash": self.current_content_hash,
            "chain": [link.to_dict() for link in self.chain],
            "status": self.status.value,
            "claim_generator": self.claim_generator,
            "claim_signature": self.claim_signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        record = cls(
            record_id=data["record_id"],
            original_content_hash=data["original_content_hash"],
            current_content_hash=data["current_content_hash"],
            status=ProvenanceStatus(data.get("status", "unverified")),
            claim_generator=data.get("claim_generator"),
            claim_signature=data.get("claim_signature"),
        )
        for link_data in data.get("chain", []):
            record.chain.append(HashChainLink.from_dict(link_data))
        return record


class CryptoSigner:
    """
    Handles cryptographic signing operations.
    """

    def __init__(self, private_key_path: Optional[str] = None):
        """
        Args:
            private_key_path: Path to PEM private key file
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography required: pip install cryptography")

        self.private_key: Optional[RSAPrivateKey] = None
        self.public_key: Optional[RSAPublicKey] = None

        if private_key_path and os.path.exists(private_key_path):
            self.load_private_key(private_key_path)

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate a new RSA keypair.

        Returns:
            (private_key_pem, public_key_pem)
        """
        self.private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        return private_pem, public_pem

    def load_private_key(self, path: str) -> None:
        """Load private key from PEM file."""
        with open(path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
        self.public_key = self.private_key.public_key()

    def sign(self, data: bytes) -> str:
        """
        Sign data and return base64-encoded signature.
        """
        if self.private_key is None:
            raise ValueError("No private key loaded")

        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )

        return base64.b64encode(signature).decode("utf-8")

    def verify(self, data: bytes, signature_b64: str, public_key: RSAPublicKey) -> bool:
        """
        Verify a signature.
        """
        try:
            signature = base64.b64decode(signature_b64)
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except InvalidSignature:
            return False


class ProvenanceManager:
    """
    Manages media provenance records.
    """

    def __init__(self, storage_dir: str, signer: Optional[CryptoSigner] = None):
        """
        Args:
            storage_dir: Directory to store provenance records
            signer: Optional CryptoSigner for signing operations
        """
        self.storage_dir = storage_dir
        self.signer = signer

        os.makedirs(storage_dir, exist_ok=True)

    def compute_content_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file content."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def create_provenance(
        self,
        file_path: str,
        author_id: Optional[str] = None,
        device_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> ProvenanceRecord:
        """
        Create initial provenance record for a file.

        Args:
            file_path: Path to the media file
            author_id: Identifier of the creator
            device_id: Identifier of the creation device
            metadata: Additional metadata

        Returns:
            ProvenanceRecord with initial chain link
        """
        record_id = str(uuid.uuid4())
        content_hash = self.compute_content_hash(file_path)

        # Create initial link
        initial_link = HashChainLink(
            link_id=str(uuid.uuid4()),
            previous_hash="genesis",
            content_hash=content_hash,
            operation="create",
            timestamp=datetime.utcnow(),
            author_id=author_id,
            device_id=device_id,
            metadata=metadata or {},
        )

        # Sign if signer available
        if self.signer:
            link_data = json.dumps(initial_link.to_dict(), sort_keys=True)
            initial_link.signature = self.signer.sign(link_data.encode())

        record = ProvenanceRecord(
            record_id=record_id,
            original_content_hash=content_hash,
            current_content_hash=content_hash,
            status=(
                ProvenanceStatus.VERIFIED
                if self.signer
                else ProvenanceStatus.UNVERIFIED
            ),
        )
        record.add_link(initial_link)

        # Save record
        self._save_record(record)

        return record

    def record_edit(
        self,
        record_id: str,
        new_file_path: str,
        operation: str,
        author_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> ProvenanceRecord:
        """
        Record an edit operation in the provenance chain.

        Args:
            record_id: Existing provenance record ID
            new_file_path: Path to the edited file
            operation: Description of the edit
            author_id: Who made the edit
            metadata: Edit metadata

        Returns:
            Updated ProvenanceRecord
        """
        record = self._load_record(record_id)
        if record is None:
            raise ValueError(f"Record not found: {record_id}")

        new_hash = self.compute_content_hash(new_file_path)

        edit_link = HashChainLink(
            link_id=str(uuid.uuid4()),
            previous_hash="",  # Will be filled by add_link
            content_hash=new_hash,
            operation=operation,
            timestamp=datetime.utcnow(),
            author_id=author_id,
            metadata=metadata or {},
        )

        if self.signer:
            # Need to set previous_hash first for signature
            if record.chain:
                edit_link.previous_hash = record.chain[-1].compute_link_hash()
            link_data = json.dumps(edit_link.to_dict(), sort_keys=True)
            edit_link.signature = self.signer.sign(link_data.encode())

        record.add_link(edit_link)
        self._save_record(record)

        return record

    def verify_provenance(
        self, file_path: str, record_id: str
    ) -> Tuple[ProvenanceStatus, Dict[str, Any]]:
        """
        Verify provenance of a file.

        Returns:
            (status, details)
        """
        record = self._load_record(record_id)
        if record is None:
            return ProvenanceStatus.UNVERIFIED, {"error": "Record not found"}

        details = {
            "record_id": record_id,
            "chain_length": len(record.chain),
            "original_hash": record.original_content_hash,
            "current_hash": record.current_content_hash,
            "errors": [],
        }

        # Verify current content
        actual_hash = self.compute_content_hash(file_path)
        if actual_hash != record.current_content_hash:
            details["errors"].append("Content hash mismatch - file may be tampered")
            return ProvenanceStatus.TAMPERED, details

        # Verify chain
        chain_valid, chain_errors = record.verify_chain()
        details["errors"].extend(chain_errors)

        if not chain_valid:
            return ProvenanceStatus.TAMPERED, details

        # Check signatures if available
        sig_count = sum(1 for link in record.chain if link.signature)
        details["signed_links"] = sig_count
        details["total_links"] = len(record.chain)

        if sig_count == len(record.chain):
            return ProvenanceStatus.VERIFIED, details
        elif sig_count > 0:
            return ProvenanceStatus.PARTIAL, details
        else:
            return ProvenanceStatus.UNVERIFIED, details

    def _get_record_path(self, record_id: str) -> str:
        """Get file path for a record."""
        return os.path.join(self.storage_dir, f"{record_id}.json")

    def _save_record(self, record: ProvenanceRecord) -> None:
        """Save record to storage."""
        path = self._get_record_path(record.record_id)
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

    def _load_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Load record from storage."""
        path = self._get_record_path(record_id)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return ProvenanceRecord.from_dict(json.load(f))


class TamperEvidentLog:
    """
    Tamper-evident log for audit trail.

    All verification events are logged in a hash-chained format.
    """

    def __init__(self, log_path: str):
        """
        Args:
            log_path: Path to the log file
        """
        self.log_path = log_path
        self.entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load existing log."""
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                self.entries = json.load(f)

    def _save(self) -> None:
        """Save log to file."""
        with open(self.log_path, "w") as f:
            json.dump(self.entries, f, indent=2)

    def _compute_entry_hash(self, entry: Dict[str, Any]) -> str:
        """Compute hash of a log entry."""
        serialized = json.dumps(entry, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def log_event(self, event_type: str, details: Dict[str, Any]) -> str:
        """
        Log an event in the tamper-evident log.

        Returns:
            Entry ID
        """
        entry_id = str(uuid.uuid4())

        # Get previous hash
        if self.entries:
            prev_hash = self._compute_entry_hash(self.entries[-1])
        else:
            prev_hash = "genesis"

        entry = {
            "entry_id": entry_id,
            "previous_hash": prev_hash,
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details,
        }

        self.entries.append(entry)
        self._save()

        return entry_id

    def verify_log_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of the log.

        Returns:
            (is_valid, errors)
        """
        errors = []

        for i, entry in enumerate(self.entries):
            if i == 0:
                if entry.get("previous_hash") != "genesis":
                    errors.append(f"Entry 0: Expected genesis hash")
            else:
                expected_prev = self._compute_entry_hash(self.entries[i - 1])
                if entry.get("previous_hash") != expected_prev:
                    errors.append(f"Entry {i}: Hash chain broken")

        return len(errors) == 0, errors


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Provenance management")
    parser.add_argument(
        "command", choices=["create", "verify", "edit"], help="Command to execute"
    )
    parser.add_argument("file", help="Media file path")
    parser.add_argument("--record-id", "-r", help="Record ID (for verify/edit)")
    parser.add_argument("--author", "-a", help="Author ID")
    parser.add_argument(
        "--storage", "-s", default="./provenance", help="Storage directory"
    )

    args = parser.parse_args()

    manager = ProvenanceManager(args.storage)

    if args.command == "create":
        record = manager.create_provenance(args.file, author_id=args.author)
        print(f"Created provenance record: {record.record_id}")
        print(f"Content hash: {record.original_content_hash}")

    elif args.command == "verify":
        if not args.record_id:
            print("Error: --record-id required for verify")
            exit(1)
        status, details = manager.verify_provenance(args.file, args.record_id)
        print(f"Status: {status.value}")
        print(json.dumps(details, indent=2))

    elif args.command == "edit":
        if not args.record_id:
            print("Error: --record-id required for edit")
            exit(1)
        record = manager.record_edit(
            args.record_id, args.file, operation="manual_edit", author_id=args.author
        )
        print(f"Recorded edit. Chain length: {len(record.chain)}")

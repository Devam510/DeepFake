"""
Test Provenance Module
tests/test_provenance.py

Validates:
- Hash chaining integrity
- Tamper detection
"""

import pytest
import sys
import os
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.provenance import (
    ProvenanceStatus,
    HashChainLink,
    ProvenanceRecord,
    ProvenanceManager,
    CryptoSigner,
    CRYPTO_AVAILABLE,
)
from datetime import datetime


class TestHashChainLink:
    """Test hash chain link functionality."""

    def test_link_has_required_fields(self):
        """Test that HashChainLink has all required fields."""
        link = HashChainLink(
            link_id="link_001",
            previous_hash="abc123",
            content_hash="def456",
            operation="create",
            timestamp=datetime.now(),
        )

        assert link.link_id == "link_001"
        assert link.previous_hash == "abc123"
        assert link.content_hash == "def456"
        assert link.operation == "create"

    def test_compute_link_hash_is_deterministic(self):
        """Test that link hash computation is deterministic."""
        ts = datetime(2026, 1, 1, 0, 0, 0)
        link = HashChainLink(
            link_id="link_001",
            previous_hash="abc123",
            content_hash="def456",
            operation="create",
            timestamp=ts,
        )

        hash1 = link.compute_link_hash()
        hash2 = link.compute_link_hash()

        assert hash1 == hash2

    def test_different_content_gives_different_hash(self):
        """Test that different content produces different hash."""
        ts = datetime(2026, 1, 1, 0, 0, 0)

        link1 = HashChainLink(
            link_id="link_001",
            previous_hash="abc123",
            content_hash="def456",
            operation="create",
            timestamp=ts,
        )

        link2 = HashChainLink(
            link_id="link_001",
            previous_hash="abc123",
            content_hash="xyz789",  # Different content hash
            operation="create",
            timestamp=ts,
        )

        assert link1.compute_link_hash() != link2.compute_link_hash()

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        ts = datetime(2026, 1, 1, 0, 0, 0)
        link = HashChainLink(
            link_id="link_001",
            previous_hash="abc123",
            content_hash="def456",
            operation="create",
            timestamp=ts,
        )

        data = link.to_dict()
        restored = HashChainLink.from_dict(data)

        assert restored.link_id == link.link_id
        assert restored.content_hash == link.content_hash


class TestProvenanceRecord:
    """Test provenance record functionality."""

    def test_chain_starts_empty(self):
        """Test that new record has empty chain."""
        record = ProvenanceRecord(
            record_id="rec_001",
            original_content_hash="abc123",
            current_content_hash="abc123",
        )

        assert len(record.chain) == 0

    def test_add_link_extends_chain(self):
        """Test that adding a link extends the chain."""
        record = ProvenanceRecord(
            record_id="rec_001",
            original_content_hash="abc123",
            current_content_hash="abc123",
        )

        link = HashChainLink(
            link_id="link_001",
            previous_hash="",
            content_hash="abc123",
            operation="create",
            timestamp=datetime.now(),
        )

        record.add_link(link)

        assert len(record.chain) == 1

    def test_verify_chain_empty_is_valid(self):
        """Test that empty chain is valid."""
        record = ProvenanceRecord(
            record_id="rec_001",
            original_content_hash="abc123",
            current_content_hash="abc123",
        )

        is_valid, errors = record.verify_chain()
        assert is_valid
        assert len(errors) == 0


class TestProvenanceManager:
    """Test provenance manager functionality."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def test_file(self):
        """Create a test file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for provenance")
            yield f.name
        os.unlink(f.name)

    def test_compute_content_hash_is_deterministic(self, temp_storage_dir, test_file):
        """Test content hash is deterministic."""
        manager = ProvenanceManager(temp_storage_dir)

        hash1 = manager.compute_content_hash(test_file)
        hash2 = manager.compute_content_hash(test_file)

        assert hash1 == hash2

    def test_create_provenance_returns_record(self, temp_storage_dir, test_file):
        """Test creating provenance returns a record."""
        manager = ProvenanceManager(temp_storage_dir)

        record = manager.create_provenance(test_file)

        assert isinstance(record, ProvenanceRecord)
        assert record.original_content_hash is not None

    def test_tamper_detection_modified_content(self, temp_storage_dir, test_file):
        """Test that modified file content is detected."""
        manager = ProvenanceManager(temp_storage_dir)

        # Create provenance
        record = manager.create_provenance(test_file)
        original_hash = record.original_content_hash

        # Modify file
        with open(test_file, "a") as f:
            f.write("\nTampered content!")

        # Hash should now be different
        new_hash = manager.compute_content_hash(test_file)

        assert new_hash != original_hash


class TestCryptoSigner:
    """Test cryptographic signing functionality."""

    @pytest.mark.skipif(
        not CRYPTO_AVAILABLE, reason="Cryptography library not installed"
    )
    def test_generate_keypair(self):
        """Test key pair generation."""
        signer = CryptoSigner()
        private_pem, public_pem = signer.generate_keypair()

        assert private_pem is not None
        assert public_pem is not None
        assert b"PRIVATE KEY" in private_pem
        assert b"PUBLIC KEY" in public_pem

    @pytest.mark.skipif(
        not CRYPTO_AVAILABLE, reason="Cryptography library not installed"
    )
    def test_sign_produces_signature(self):
        """Test that signing produces a signature."""
        signer = CryptoSigner()
        signer.generate_keypair()

        data = b"Test data to sign"
        signature = signer.sign(data)

        assert signature is not None
        assert len(signature) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

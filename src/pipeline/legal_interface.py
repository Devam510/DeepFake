"""
DeepFake Detection System - Legal & Societal Interface
Layer 9: Legal + Societal Interface

This module implements the legal safeguards:
- Audit logging for court admissibility
- Evidence-based language (not "truth" claims)
- Appeals and re-analysis support
- Explainability for legal review
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import uuid


class VerdictType(Enum):
    """
    Verdict types - explicitly NOT binary.

    Per Layer 9: Avoid claiming "truth", only "evidence".
    """

    STRONG_EVIDENCE_AUTHENTIC = "strong_evidence_authentic"
    MODERATE_EVIDENCE_AUTHENTIC = "moderate_evidence_authentic"
    INCONCLUSIVE = "inconclusive"
    MODERATE_EVIDENCE_SYNTHETIC = "moderate_evidence_synthetic"
    STRONG_EVIDENCE_SYNTHETIC = "strong_evidence_synthetic"
    REQUIRES_MANUAL_REVIEW = "requires_manual_review"


@dataclass
class AuditLogEntry:
    """
    Immutable audit log entry.

    Designed for court admissibility.
    """

    entry_id: str
    timestamp: datetime
    action: str
    actor_id: str  # Who performed the action
    media_hash: str  # SHA-256 of analyzed media
    request_data: Dict[str, Any]
    result_data: Dict[str, Any]
    previous_entry_hash: str  # Chain integrity
    entry_hash: str = ""

    def __post_init__(self):
        """Compute entry hash after initialization."""
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of entry contents."""
        data = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "actor_id": self.actor_id,
            "media_hash": self.media_hash,
            "request_data": self.request_data,
            "result_data": self.result_data,
            "previous_entry_hash": self.previous_entry_hash,
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "actor_id": self.actor_id,
            "media_hash": self.media_hash,
            "request_data": self.request_data,
            "result_data": self.result_data,
            "previous_entry_hash": self.previous_entry_hash,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditLogEntry":
        return cls(
            entry_id=data["entry_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action=data["action"],
            actor_id=data["actor_id"],
            media_hash=data["media_hash"],
            request_data=data["request_data"],
            result_data=data["result_data"],
            previous_entry_hash=data["previous_entry_hash"],
            entry_hash=data.get("entry_hash", ""),
        )


class ForensicAuditLog:
    """
    Forensic-grade audit log for legal compliance.

    Features:
    - Hash chain integrity
    - Immutable entries
    - Tamper detection
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.log_file = os.path.join(storage_dir, "audit_log.json")
        self.entries: List[AuditLogEntry] = []

        os.makedirs(storage_dir, exist_ok=True)
        self._load()

    def _load(self) -> None:
        """Load existing log."""
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                data = json.load(f)
                self.entries = [
                    AuditLogEntry.from_dict(e) for e in data.get("entries", [])
                ]

    def _save(self) -> None:
        """Save log to file."""
        data = {
            "version": "1.0",
            "created": self.entries[0].timestamp.isoformat() if self.entries else None,
            "last_modified": datetime.utcnow().isoformat(),
            "entries": [e.to_dict() for e in self.entries],
        }
        with open(self.log_file, "w") as f:
            json.dump(data, f, indent=2)

    def log_analysis(
        self,
        actor_id: str,
        media_path: str,
        request_data: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> str:
        """
        Log an analysis action.

        Returns:
            Entry ID
        """
        # Compute media hash
        media_hash = self._hash_file(media_path)

        # Get previous hash
        if self.entries:
            prev_hash = self.entries[-1].entry_hash
        else:
            prev_hash = "genesis"

        entry = AuditLogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            action="analysis",
            actor_id=actor_id,
            media_hash=media_hash,
            request_data=request_data,
            result_data=result_data,
            previous_entry_hash=prev_hash,
        )

        self.entries.append(entry)
        self._save()

        return entry.entry_id

    def log_appeal(
        self,
        original_entry_id: str,
        actor_id: str,
        appeal_reason: str,
        reanalysis_result: Dict[str, Any],
    ) -> str:
        """Log an appeal/re-analysis."""
        if self.entries:
            prev_hash = self.entries[-1].entry_hash
        else:
            prev_hash = "genesis"

        entry = AuditLogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            action="appeal",
            actor_id=actor_id,
            media_hash="",  # Reference original
            request_data={
                "original_entry_id": original_entry_id,
                "appeal_reason": appeal_reason,
            },
            result_data=reanalysis_result,
            previous_entry_hash=prev_hash,
        )

        self.entries.append(entry)
        self._save()

        return entry.entry_id

    def verify_integrity(self) -> tuple[bool, List[str]]:
        """
        Verify the integrity of the audit log.

        Returns:
            (is_valid, list of errors)
        """
        errors = []

        for i, entry in enumerate(self.entries):
            # Verify entry hash
            computed = entry._compute_hash()
            if computed != entry.entry_hash:
                errors.append(f"Entry {i}: Hash mismatch (tampering detected)")

            # Verify chain
            if i == 0:
                if entry.previous_entry_hash != "genesis":
                    errors.append(f"Entry 0: Expected genesis hash")
            else:
                if entry.previous_entry_hash != self.entries[i - 1].entry_hash:
                    errors.append(f"Entry {i}: Chain broken")

        return len(errors) == 0, errors

    def get_entry(self, entry_id: str) -> Optional[AuditLogEntry]:
        """Retrieve entry by ID."""
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def export_for_court(self, entry_id: str) -> Dict[str, Any]:
        """
        Export entry and context for legal proceedings.

        Includes chain verification and surrounding context.
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return {"error": "Entry not found"}

        # Find position in chain
        idx = next(i for i, e in enumerate(self.entries) if e.entry_id == entry_id)

        # Include context entries
        start = max(0, idx - 2)
        end = min(len(self.entries), idx + 3)
        context = [self.entries[i].to_dict() for i in range(start, end)]

        # Verify chain up to this point
        is_valid, errors = self.verify_integrity()

        return {
            "exported_at": datetime.utcnow().isoformat(),
            "target_entry": entry.to_dict(),
            "chain_context": context,
            "chain_verification": {
                "is_valid": is_valid,
                "errors": errors,
                "total_entries": len(self.entries),
                "position": idx + 1,
            },
            "legal_notice": (
                "This analysis provides evidence, not certainty. "
                "The scores indicate probability of synthetic generation, "
                "not definitive classification. Subject to appeal and re-analysis."
            ),
        }

    def _hash_file(self, path: str) -> str:
        """Compute SHA-256 hash of file."""
        if not os.path.exists(path):
            return "file_not_found"

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class EvidenceBasedReport:
    """
    Generate legally-appropriate evidence-based reports.

    Per Layer 9: Explain confidence, not certainty.
    """

    LANGUAGE_GUIDELINES = {
        # What NOT to say
        "forbidden": [
            "This is definitely fake",
            "This is 100% real",
            "This is a deepfake",
            "This is authentic",
            "The system determined",
            "The verdict is",
        ],
        # What TO say
        "recommended": [
            "The analysis indicates",
            "Evidence suggests",
            "With {confidence}% confidence",
            "Forensic signals show",
            "The probability of synthetic generation is",
            "Manual review is recommended",
        ],
    }

    @staticmethod
    def get_verdict(authenticity_score: float, confidence: float) -> VerdictType:
        """
        Convert score to evidence-based verdict.

        Never returns binary true/false.
        """
        if confidence < 60:
            return VerdictType.REQUIRES_MANUAL_REVIEW

        if authenticity_score >= 85:
            return VerdictType.STRONG_EVIDENCE_AUTHENTIC
        elif authenticity_score >= 65:
            return VerdictType.MODERATE_EVIDENCE_AUTHENTIC
        elif authenticity_score >= 35:
            return VerdictType.INCONCLUSIVE
        elif authenticity_score >= 15:
            return VerdictType.MODERATE_EVIDENCE_SYNTHETIC
        else:
            return VerdictType.STRONG_EVIDENCE_SYNTHETIC

    @staticmethod
    def generate_summary(analysis_result: Dict[str, Any]) -> str:
        """
        Generate evidence-based summary.

        Uses legally-appropriate language.
        """
        auth = analysis_result.get("authenticity", {})
        score = auth.get("score", 50)
        conf = auth.get("confidence_interval", {})
        conf_lower = conf.get("lower", 0)
        conf_upper = conf.get("upper", 100)

        verdict = EvidenceBasedReport.get_verdict(
            score, 100 - (conf_upper - conf_lower)
        )

        # Build evidence-based summary
        lines = [
            f"## Analysis Summary\n",
            f"**Evidence Level**: {verdict.value.replace('_', ' ').title()}\n",
            f"",
            f"### Findings",
            f"- Authenticity score: {score:.1f}/100",
            f"- Confidence interval: [{conf_lower:.1f}, {conf_upper:.1f}]",
            f"- Interpretation: {auth.get('interpretation', 'Not available')}",
            f"",
            f"### Legal Notice",
            f"This analysis provides probabilistic evidence based on forensic",
            f"signal analysis. It does not constitute a definitive determination",
            f"of authenticity or inauthenticity. The methodology is subject to",
            f"limitations and adversarial evasion. Appeal and re-analysis available.",
        ]

        return "\n".join(lines)

    @staticmethod
    def generate_appeal_form(
        entry_id: str, reason_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate appeal form template.

        Per Layer 9: Support appeals and re-analysis.
        """
        if reason_categories is None:
            reason_categories = [
                "technical_error",
                "new_evidence",
                "misidentified_source",
                "known_authentic_source",
                "context_not_considered",
                "other",
            ]

        return {
            "appeal_form_version": "1.0",
            "original_entry_id": entry_id,
            "appeal_categories": reason_categories,
            "required_fields": [
                "appellant_name",
                "appellant_contact",
                "appeal_reason_category",
                "detailed_explanation",
                "supporting_evidence_description",
            ],
            "instructions": (
                "To appeal this analysis, complete all required fields. "
                "If you have evidence of the media's authentic origin "
                "(e.g., original camera files, photographer testimony, "
                "chain of custody documentation), please describe it in "
                "supporting_evidence_description. The appeal will be "
                "reviewed by a human analyst within 5 business days."
            ),
        }


class ComplianceChecker:
    """
    Check outputs for legal compliance.
    """

    FORBIDDEN_PHRASES = [
        "definitely fake",
        "definitely real",
        "100%",
        "certainly",
        "without doubt",
        "proven fake",
        "proven real",
        "is a deepfake",
        "is authentic",
        "we determined",
        "the truth is",
    ]

    @staticmethod
    def check_output(text: str) -> tuple[bool, List[str]]:
        """
        Check if output text is legally compliant.

        Returns:
            (is_compliant, list of violations)
        """
        violations = []
        text_lower = text.lower()

        for phrase in ComplianceChecker.FORBIDDEN_PHRASES:
            if phrase in text_lower:
                violations.append(f"Contains forbidden phrase: '{phrase}'")

        return len(violations) == 0, violations

    @staticmethod
    def sanitize_output(text: str) -> str:
        """
        Attempt to sanitize output for compliance.
        """
        replacements = {
            "definitely fake": "shows strong evidence of synthetic generation",
            "definitely real": "shows strong evidence of authentic origin",
            "is a deepfake": "may be synthetically generated",
            "is authentic": "appears to be authentic",
            "100%": "with high probability",
            "certainly": "likely",
            "without doubt": "with high confidence",
        }

        result = text
        for old, new in replacements.items():
            result = result.replace(old, new)
            result = result.replace(old.capitalize(), new.capitalize())

        return result


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Legal interface tools")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--verify-log", help="Verify audit log at path")

    args = parser.parse_args()

    if args.demo:
        print("Running legal interface demo...")

        # Create audit log
        log = ForensicAuditLog("./demo_audit")

        # Simulate analysis
        entry_id = log.log_analysis(
            actor_id="analyst_001",
            media_path="test.jpg",
            request_data={"client": "demo"},
            result_data={
                "authenticity": {
                    "score": 75,
                    "confidence_interval": {"lower": 65, "upper": 85},
                }
            },
        )

        print(f"Created audit entry: {entry_id}")

        # Verify
        is_valid, errors = log.verify_integrity()
        print(f"Log integrity: {'VALID' if is_valid else 'INVALID'}")

        # Export
        export = log.export_for_court(entry_id)
        print("\nCourt export:")
        print(json.dumps(export, indent=2))

        # Generate report
        report = EvidenceBasedReport.generate_summary(
            export["target_entry"]["result_data"]
        )
        print("\nEvidence Report:")
        print(report)

    elif args.verify_log:
        log = ForensicAuditLog(args.verify_log)
        is_valid, errors = log.verify_integrity()

        if is_valid:
            print(f"✓ Log integrity verified ({len(log.entries)} entries)")
        else:
            print(f"✗ Log integrity FAILED:")
            for error in errors:
                print(f"  - {error}")
    else:
        parser.print_help()

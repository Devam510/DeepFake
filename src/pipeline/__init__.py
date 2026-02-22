"""
DeepFake Detection System - Pipeline Package
"""

from .integration import (
    IntegrationManager,
    PlatformConnector,
    WebhookConnector,
    BatchProcessor,
)

from .evaluation import (
    CoreMetrics,
    CalibrationMetrics,
    RobustnessMetrics,
    GeneralizationMetrics,
    TemporalDecayTracker,
    EvaluationSuite,
)

from .legal_interface import (
    ForensicAuditLog,
    AuditLogEntry,
    EvidenceBasedReport,
    ComplianceChecker,
    VerdictType,
)

__all__ = [
    # Integration
    "IntegrationManager",
    "PlatformConnector",
    "WebhookConnector",
    "BatchProcessor",
    # Evaluation
    "CoreMetrics",
    "CalibrationMetrics",
    "RobustnessMetrics",
    "GeneralizationMetrics",
    "TemporalDecayTracker",
    "EvaluationSuite",
    # Legal
    "ForensicAuditLog",
    "AuditLogEntry",
    "EvidenceBasedReport",
    "ComplianceChecker",
    "VerdictType",
]

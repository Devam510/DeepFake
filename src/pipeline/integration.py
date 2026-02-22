"""
DeepFake Detection System - Platform Integration
Layer 7: Platform Integration

Connectors for integrating with external platforms:
- Social media upload pipelines
- Messaging apps
- Newsroom CMS
- Courts/compliance systems

Latency targets: < 500ms for images, < 2s for short video
"""

import json
import time
import os
import hashlib
import queue
import threading
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum


class PlatformType(Enum):
    """Supported platform types."""

    SOCIAL_MEDIA = "social_media"
    MESSAGING = "messaging"
    NEWSROOM = "newsroom"
    LEGAL = "legal"
    CUSTOM = "custom"


@dataclass
class IntegrationConfig:
    """Configuration for a platform integration."""

    platform_type: PlatformType
    platform_name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    webhook_url: Optional[str] = None
    max_latency_ms: float = 500.0
    batch_size: int = 10
    retry_count: int = 3
    enabled: bool = True


@dataclass
class MediaSubmission:
    """Media submitted for analysis."""

    submission_id: str
    file_path: str
    source_platform: str
    upload_timestamp: datetime
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "file_path": self.file_path,
            "source_platform": self.source_platform,
            "upload_timestamp": self.upload_timestamp.isoformat(),
            "user_id": self.user_id,
            "metadata": self.metadata or {},
        }


@dataclass
class AnalysisResult:
    """Result to send back to platform."""

    submission_id: str
    authenticity_score: float
    confidence: float
    requires_review: bool
    processing_time_ms: float
    summary: str
    full_result: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "authenticity_score": self.authenticity_score,
            "confidence": self.confidence,
            "requires_review": self.requires_review,
            "processing_time_ms": self.processing_time_ms,
            "summary": self.summary,
            "full_result": self.full_result,
        }


class PlatformConnector(ABC):
    """
    Abstract base class for platform connectors.
    """

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to platform."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    def receive_submission(self) -> Optional[MediaSubmission]:
        """Receive a media submission from the platform."""
        pass

    @abstractmethod
    def send_result(self, result: AnalysisResult) -> bool:
        """Send analysis result back to platform."""
        pass

    def health_check(self) -> Dict[str, Any]:
        """Check connector health."""
        return {
            "platform": self.config.platform_name,
            "connected": self.connected,
            "enabled": self.config.enabled,
        }


class WebhookConnector(PlatformConnector):
    """
    Generic webhook-based connector.

    Receives media via webhook, sends results via callback.
    """

    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.submission_queue: queue.Queue = queue.Queue()
        self.result_callbacks: Dict[str, str] = {}

    def connect(self) -> bool:
        """Mark as connected (webhook is passive)."""
        self.connected = True
        return True

    def disconnect(self) -> None:
        """Mark as disconnected."""
        self.connected = False

    def receive_submission(self) -> Optional[MediaSubmission]:
        """Get next submission from queue."""
        try:
            return self.submission_queue.get_nowait()
        except queue.Empty:
            return None

    def enqueue_submission(
        self,
        file_path: str,
        callback_url: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Enqueue a submission for processing.

        Returns:
            Submission ID
        """
        submission_id = hashlib.md5(f"{file_path}{time.time()}".encode()).hexdigest()[
            :16
        ]

        submission = MediaSubmission(
            submission_id=submission_id,
            file_path=file_path,
            source_platform=self.config.platform_name,
            upload_timestamp=datetime.utcnow(),
            user_id=user_id,
            metadata=metadata,
        )

        self.submission_queue.put(submission)
        self.result_callbacks[submission_id] = callback_url

        return submission_id

    def send_result(self, result: AnalysisResult) -> bool:
        """Send result via webhook callback."""
        callback_url = self.result_callbacks.get(result.submission_id)
        if not callback_url:
            return False

        # In real implementation, would POST to callback_url
        # Here we just log it
        print(f"Would POST to {callback_url}: {json.dumps(result.to_dict())}")

        del self.result_callbacks[result.submission_id]
        return True


class BatchProcessor:
    """
    Processes media in batches for efficiency.

    Optimizes throughput while maintaining latency targets.
    """

    def __init__(
        self, analyze_fn: Callable, batch_size: int = 10, max_wait_ms: float = 100
    ):
        """
        Args:
            analyze_fn: Function to analyze a single media file
            batch_size: Maximum batch size
            max_wait_ms: Maximum wait time before processing partial batch
        """
        self.analyze_fn = analyze_fn
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms

        self.pending: List[MediaSubmission] = []
        self.lock = threading.Lock()
        self.last_batch_time = time.time()

    def add(self, submission: MediaSubmission) -> None:
        """Add submission to pending batch."""
        with self.lock:
            self.pending.append(submission)

    def should_process(self) -> bool:
        """Check if batch should be processed."""
        with self.lock:
            if len(self.pending) >= self.batch_size:
                return True

            elapsed_ms = (time.time() - self.last_batch_time) * 1000
            if self.pending and elapsed_ms >= self.max_wait_ms:
                return True

            return False

    def process_batch(self) -> List[AnalysisResult]:
        """Process current batch."""
        with self.lock:
            to_process = self.pending[: self.batch_size]
            self.pending = self.pending[self.batch_size :]
            self.last_batch_time = time.time()

        results = []
        for submission in to_process:
            start = time.time()

            try:
                full_result = self.analyze_fn(submission.file_path)

                auth_score = full_result.get("authenticity", {}).get("score", 50)
                conf_lower = (
                    full_result.get("authenticity", {})
                    .get("confidence_interval", {})
                    .get("lower", 30)
                )

                # Flag for review if low confidence or borderline
                requires_review = (
                    auth_score < 70 and auth_score > 30
                ) or conf_lower < 40

                summary = full_result.get("authenticity", {}).get(
                    "interpretation", "Analysis complete"
                )

            except Exception as e:
                full_result = {"error": str(e)}
                auth_score = 50
                requires_review = True
                summary = f"Error during analysis: {e}"

            processing_time = (time.time() - start) * 1000

            results.append(
                AnalysisResult(
                    submission_id=submission.submission_id,
                    authenticity_score=auth_score,
                    confidence=100 - abs(auth_score - 50) * 2,
                    requires_review=requires_review,
                    processing_time_ms=processing_time,
                    summary=summary,
                    full_result=full_result,
                )
            )

        return results


class IntegrationManager:
    """
    Manages all platform integrations.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to integration configuration file
        """
        self.connectors: Dict[str, PlatformConnector] = {}
        self.processor: Optional[BatchProcessor] = None
        self.running = False
        self._worker_thread: Optional[threading.Thread] = None

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, path: str) -> None:
        """Load integration configuration."""
        with open(path, "r") as f:
            config_data = json.load(f)

        for platform_config in config_data.get("platforms", []):
            config = IntegrationConfig(
                platform_type=PlatformType(platform_config["type"]),
                platform_name=platform_config["name"],
                api_key=platform_config.get("api_key"),
                webhook_url=platform_config.get("webhook_url"),
                max_latency_ms=platform_config.get("max_latency_ms", 500),
                enabled=platform_config.get("enabled", True),
            )

            connector = WebhookConnector(config)
            self.add_connector(config.platform_name, connector)

    def add_connector(self, name: str, connector: PlatformConnector) -> None:
        """Add a platform connector."""
        self.connectors[name] = connector

    def set_analyzer(self, analyze_fn: Callable) -> None:
        """Set the analysis function."""
        self.processor = BatchProcessor(analyze_fn)

    def start(self) -> None:
        """Start the integration manager."""
        if self.running:
            return

        self.running = True

        # Connect all connectors
        for name, connector in self.connectors.items():
            if connector.config.enabled:
                connector.connect()

        # Start worker thread
        self._worker_thread = threading.Thread(target=self._worker_loop)
        self._worker_thread.daemon = True
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop the integration manager."""
        self.running = False

        for connector in self.connectors.values():
            connector.disconnect()

        if self._worker_thread:
            self._worker_thread.join(timeout=5)

    def _worker_loop(self) -> None:
        """Main worker loop."""
        while self.running:
            # Collect submissions from all connectors
            for name, connector in self.connectors.items():
                if not connector.connected:
                    continue

                submission = connector.receive_submission()
                if submission and self.processor:
                    self.processor.add(submission)

            # Process batches
            if self.processor and self.processor.should_process():
                results = self.processor.process_batch()

                for result in results:
                    # Find the right connector to send result
                    for connector in self.connectors.values():
                        if result.submission_id in getattr(
                            connector, "result_callbacks", {}
                        ):
                            connector.send_result(result)
                            break

            time.sleep(0.01)  # Small sleep to prevent busy-waiting

    def get_status(self) -> Dict[str, Any]:
        """Get integration status."""
        return {
            "running": self.running,
            "connectors": {
                name: connector.health_check()
                for name, connector in self.connectors.items()
            },
        }


# ============================================================
# Quick integration helpers
# ============================================================


def create_social_media_connector(
    platform_name: str, webhook_url: str
) -> WebhookConnector:
    """Create a connector for social media platforms."""
    config = IntegrationConfig(
        platform_type=PlatformType.SOCIAL_MEDIA,
        platform_name=platform_name,
        webhook_url=webhook_url,
        max_latency_ms=500,  # Layer 7 spec
    )
    return WebhookConnector(config)


def create_newsroom_connector(platform_name: str, webhook_url: str) -> WebhookConnector:
    """Create a connector for newsroom CMS."""
    config = IntegrationConfig(
        platform_type=PlatformType.NEWSROOM,
        platform_name=platform_name,
        webhook_url=webhook_url,
        max_latency_ms=1000,  # Slightly more relaxed
    )
    return WebhookConnector(config)


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    print("Platform Integration Module")
    print("---------------------------")

    # Demo
    config = IntegrationConfig(
        platform_type=PlatformType.SOCIAL_MEDIA,
        platform_name="demo_platform",
        webhook_url="https://example.com/webhook",
    )

    connector = WebhookConnector(config)
    connector.connect()

    # Simulate submission
    submission_id = connector.enqueue_submission(
        file_path="test.jpg",
        callback_url="https://example.com/callback",
        user_id="user123",
    )

    print(f"Created submission: {submission_id}")
    print(f"Status: {connector.health_check()}")

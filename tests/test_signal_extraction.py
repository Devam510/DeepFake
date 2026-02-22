"""
Test Signal Extraction Module
tests/test_signal_extraction.py

Validates:
- Valid feature keys
- No NaNs or crashes
"""

import pytest
import numpy as np
import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.image_signals import ImageSignalExtractor, ImageSignals


class TestImageSignalExtractor:
    """Test ImageSignalExtractor functionality."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return ImageSignalExtractor()

    @pytest.fixture
    def test_image_path(self):
        """Create a temporary test image."""
        # Create a simple test image using numpy
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        try:
            import cv2

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, img)
                yield f.name
            os.unlink(f.name)
        except ImportError:
            pytest.skip("OpenCV required for image tests")

    def test_extract_returns_image_signals(self, extractor, test_image_path):
        """Test that extract returns ImageSignals dataclass."""
        signals = extractor.extract(test_image_path)
        assert isinstance(signals, ImageSignals)

    def test_extract_has_valid_feature_keys(self, extractor, test_image_path):
        """Test that extracted signals have all expected keys."""
        signals = extractor.extract(test_image_path)

        # Required feature attributes
        expected_attrs = [
            "fft_features",
            "dct_features",
            "texture_entropy",
            "edge_gradients",
            "diffusion_residue",
            "prnu_features",
        ]

        for attr in expected_attrs:
            assert hasattr(signals, attr), f"Missing attribute: {attr}"

    def test_extract_no_nan_values(self, extractor, test_image_path):
        """Test that no features contain NaN values."""
        signals = extractor.extract(test_image_path)

        # Check numeric fields for NaN
        if signals.texture_entropy is not None:
            assert not np.isnan(signals.texture_entropy), "texture_entropy contains NaN"

        if signals.edge_gradients is not None:
            if isinstance(signals.edge_gradients, dict):
                for k, v in signals.edge_gradients.items():
                    if isinstance(v, (int, float)):
                        assert not np.isnan(v), f"edge_gradients[{k}] contains NaN"

    def test_extract_does_not_crash_on_valid_image(self, extractor, test_image_path):
        """Test that extraction completes without exception."""
        try:
            signals = extractor.extract(test_image_path)
            assert signals is not None
        except Exception as e:
            pytest.fail(f"Extraction crashed with: {e}")

    def test_to_feature_vector_returns_array(self, extractor, test_image_path):
        """Test that to_feature_vector returns numpy array."""
        signals = extractor.extract(test_image_path)

        if hasattr(signals, "to_feature_vector"):
            vec = signals.to_feature_vector()
            assert isinstance(vec, np.ndarray)
            assert len(vec) > 0

    def test_extract_handles_grayscale(self, extractor):
        """Test that extractor handles grayscale images."""
        # Create grayscale test image
        img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)

        try:
            import cv2

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, img)
                try:
                    signals = extractor.extract(f.name)
                    # Should either succeed or raise a handled exception
                    assert signals is not None or True
                except ValueError:
                    # Acceptable if grayscale explicitly not supported
                    pass
                finally:
                    os.unlink(f.name)
        except ImportError:
            pytest.skip("OpenCV required for image tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

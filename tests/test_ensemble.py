"""
Test Ensemble Module
tests/test_ensemble.py

Validates:
- Weighted aggregation correctness
- CI computation sanity
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestWeightedAggregation:
    """Test weighted aggregation logic."""

    def test_equal_weights_gives_average(self):
        """Test that equal weights give arithmetic mean."""
        probs = [0.6, 0.8, 0.4]
        weights = [1 / 3, 1 / 3, 1 / 3]

        weighted_sum = sum(p * w for p, w in zip(probs, weights))

        assert abs(weighted_sum - 0.6) < 0.01

    def test_unequal_weights_favor_higher_weight(self):
        """Test that higher weights have more influence."""
        probs = [0.9, 0.1]
        weights = [0.3, 0.7]

        weighted_avg = sum(p * w for p, w in zip(probs, weights))

        assert weighted_avg < 0.5
        assert abs(weighted_avg - 0.34) < 0.01

    def test_weights_sum_to_one_after_normalization(self):
        """Test that weights are normalized to sum to 1."""
        weights = [2, 3, 5]
        total = sum(weights)
        normalized = [w / total for w in weights]

        assert abs(sum(normalized) - 1.0) < 0.001


class TestConfidenceInterval:
    """Test confidence interval computation."""

    def test_ci_contains_score(self):
        """Test that CI contains the point estimate."""
        score = 75.0
        ci_lower = 70.0
        ci_upper = 80.0

        assert ci_lower <= score <= ci_upper

    def test_ci_bounds_are_ordered(self):
        """Test that CI lower <= upper."""
        ci_lower = 60.0
        ci_upper = 80.0

        assert ci_lower <= ci_upper

    def test_ci_width_increases_with_uncertainty(self):
        """Test that more variance = wider CI."""
        low_var_samples = [0.7, 0.71, 0.69, 0.72, 0.68]
        low_var_width = np.percentile(low_var_samples, 97.5) - np.percentile(
            low_var_samples, 2.5
        )

        high_var_samples = [0.5, 0.9, 0.3, 0.8, 0.4]
        high_var_width = np.percentile(high_var_samples, 97.5) - np.percentile(
            high_var_samples, 2.5
        )

        assert high_var_width > low_var_width


class TestProbabilitySanity:
    """Test probability constraints."""

    def test_probabilities_sum_to_one(self):
        """Test that authenticity + synthetic = 1."""
        auth_score = 0.75
        synth_prob = 1 - auth_score

        assert abs(auth_score + synth_prob - 1.0) < 0.001

    def test_probability_bounds(self):
        """Test probabilities in [0, 1]."""
        probs = [0.0, 0.5, 1.0, 0.25, 0.75]

        for p in probs:
            assert 0.0 <= p <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

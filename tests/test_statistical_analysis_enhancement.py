"""Tests for enhancements to statistical analysis module"""
import pytest
import sys
import os

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

from statistical_analysis import KozaiLidovAnalyzer


def test_kozai_lidov_analyzer_init():
    """Test KozaiLidovAnalyzer initialization with new parameters"""

    # Test default initialization
    analyzer = KozaiLidovAnalyzer()
    assert analyzer.n_trials == 1000
    assert analyzer.inner_a0_AU == 1.0
    assert analyzer.horizon_Gyr == 1.0
    assert analyzer.rpcrit_Rs == 3.0

    # Test initialization with custom parameters
    analyzer_custom = KozaiLidovAnalyzer(
        n_trials=500,
        inner_a0_AU=0.5,
        horizon_Gyr=3.0,
        rpcrit_Rs=2.5
    )
    assert analyzer_custom.n_trials == 500
    assert analyzer_custom.inner_a0_AU == 0.5
    assert analyzer_custom.horizon_Gyr == 3.0
    assert analyzer_custom.rpcrit_Rs == 2.5

    # Test partial initialization (some defaults, some custom)
    analyzer_partial = KozaiLidovAnalyzer(inner_a0_AU=0.7)
    assert analyzer_partial.n_trials == 1000  # default
    assert analyzer_partial.inner_a0_AU == 0.7  # custom
    assert analyzer_partial.horizon_Gyr == 1.0  # default
    assert analyzer_partial.rpcrit_Rs == 3.0  # default


def test_kozai_lidov_analyzer_parameters_usage():
    """Test that the analyzer uses the new parameters correctly"""

    analyzer = KozaiLidovAnalyzer(
        inner_a0_AU=0.8,
        horizon_Gyr=2.0,
        rpcrit_Rs=4.0
    )

    # The parameters should be accessible
    assert hasattr(analyzer, 'inner_a0_AU')
    assert hasattr(analyzer, 'horizon_Gyr')
    assert hasattr(analyzer, 'rpcrit_Rs')

    # Values should match what was set
    assert analyzer.inner_a0_AU == 0.8
    assert analyzer.horizon_Gyr == 2.0
    assert analyzer.rpcrit_Rs == 4.0


if __name__ == "__main__":
    test_kozai_lidov_analyzer_init()
    test_kozai_lidov_analyzer_parameters_usage()
    print("All statistical analysis enhancement tests passed!")
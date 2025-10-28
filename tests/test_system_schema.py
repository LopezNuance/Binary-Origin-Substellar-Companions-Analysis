"""Tests for system schema module"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

from system_schema import build_system_table, _log10_safe


def test_log10_safe():
    """Test safe log10 function"""
    # Test normal values
    result = _log10_safe([1, 10, 100])
    expected = [0, 1, 2]
    np.testing.assert_array_almost_equal(result, expected)

    # Test zeros and negatives
    result = _log10_safe([0, -1, 1])
    assert np.isinf(result[0])  # log10(0) = -inf
    assert np.isnan(result[1])  # log10(-1) = nan
    assert result[2] == 0       # log10(1) = 0


def test_build_system_table():
    """Test system table building with synthetic data"""
    # Create temporary CSV files with synthetic data
    with tempfile.TemporaryDirectory() as tmpdir:
        nasa_csv = os.path.join(tmpdir, "test_nasa.csv")
        bd_csv = os.path.join(tmpdir, "test_bd.csv")
        out_csv = os.path.join(tmpdir, "test_systems.csv")

        # Create synthetic NASA data
        nasa_data = pd.DataFrame({
            'hostname': ['Star1', 'Star1', 'Star2'],
            'Mstar': [0.08, 0.08, 0.10],
            'Mj': [0.3, 0.5, 0.2],
            'a_AU': [0.05, 0.1, 0.2],
            'e': [0.0, 0.1, 0.2]
        })
        nasa_data.to_csv(nasa_csv, index=False)

        # Create synthetic BD data
        bd_data = pd.DataFrame({
            'host': ['Star3'],
            'Mstar': [0.12],
            'Mj': [15.0],  # Brown dwarf
            'a_AU': [0.3],
            'e': [0.05]
        })
        bd_data.to_csv(bd_csv, index=False)

        # Test the function
        systems = build_system_table(nasa_csv, bd_csv, None, out_csv)

        # Check results
        assert len(systems) == 3  # Three systems
        assert 'Star1' in systems['host'].values
        assert 'Star2' in systems['host'].values
        assert 'Star3' in systems['host'].values

        # Check aggregation for Star1 (has 2 companions)
        star1_row = systems[systems['host'] == 'Star1'].iloc[0]
        assert star1_row['n_comp'] == 2
        assert star1_row['q_max'] > 0  # Should have calculated mass ratios

        # Check that output file was created
        assert os.path.exists(out_csv)
        saved_systems = pd.read_csv(out_csv)
        assert len(saved_systems) == 3


if __name__ == "__main__":
    test_log10_safe()
    test_build_system_table()
    print("All system schema tests passed!")
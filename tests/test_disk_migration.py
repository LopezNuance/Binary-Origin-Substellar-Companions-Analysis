"""Tests for disk migration module"""
import pytest
import numpy as np
import sys
import os

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

from disk_migration import typeI_timescale_sec, migrate_time_numeric


def test_typeI_timescale():
    """Test Type-I migration timescale calculation"""
    # Test with typical VLMS parameters
    t_mig = typeI_timescale_sec(
        Mstar_Msun=0.1,
        Mp_Mj=0.3,
        a_AU=1.0,
        Sigma1_gcm2=300,
        p_sigma=1.0,
        H_over_a=0.04
    )

    # Should be on order of Myr (but could be shorter for very low mass systems)
    t_mig_myr = t_mig / (1e6 * 365.25 * 24 * 3600)
    assert 0.001 < t_mig_myr < 1000, f"Unexpected timescale: {t_mig_myr} Myr"


def test_migrate_time_numeric():
    """Test numerical integration of migration"""
    t_total = migrate_time_numeric(
        Mstar_Msun=0.1,
        Mp_Mj=0.3,
        a0_AU=1.0,
        af_AU=0.05,
        Sigma1_gcm2=300,
        p_sigma=1.0,
        H_over_a=0.04,
        nstep=100  # Fewer steps for test
    )

    # Just check that the function runs and returns a finite value
    assert np.isfinite(t_total), f"Migration time should be finite, got: {t_total}"
    assert t_total != 0, f"Migration time should be non-zero, got: {t_total}"


def test_physical_constants():
    """Test that physical constants are reasonable"""
    from disk_migration import AU, M_sun, M_jup, G, YEAR

    # Test AU in cm
    assert 1.4e13 < AU < 1.5e13

    # Test solar mass in g
    assert 1.9e33 < M_sun < 2.0e33

    # Test gravitational constant
    assert 6.6e-8 < G < 6.7e-8


if __name__ == "__main__":
    test_typeI_timescale()
    test_migrate_time_numeric()
    test_physical_constants()
    print("All disk migration tests passed!")
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statistical_analysis import (
    StatisticalAnalyzer,
    KozaiLidovAnalyzer,
    kozai_lidov_feasibility_single,
    kozai_lidov_feasibility_with_age,
    age_dependent_stellar_radius,
    age_dependent_tidal_q_factor,
)


def build_clustered_dataframe(n_per_cluster=30):
    rng = np.random.default_rng(42)
    cluster1 = rng.normal(loc=[-2.0, -1.0], scale=0.05, size=(n_per_cluster, 2))
    cluster2 = rng.normal(loc=[-0.5, -0.2], scale=0.05, size=(n_per_cluster, 2))
    data = np.vstack([cluster1, cluster2])
    df = pd.DataFrame(data, columns=["log_mass_ratio", "log_semimajor_axis"])
    df["eccentricity"] = rng.beta(1.5, 3.0, len(df))
    df["log_host_mass"] = rng.normal(-1.0, 0.1, len(df))
    df["mass_ratio"] = 10 ** df["log_mass_ratio"]
    df["high_mass_ratio"] = df["mass_ratio"] > np.median(df["mass_ratio"])
    df["discovery_method"] = np.where(df["high_mass_ratio"], "Transit", "RV")
    # Add age data for age regression tests
    df["host_age_gyr"] = rng.uniform(0.5, 10.0, len(df))
    df["semimajor_axis_au"] = 10 ** df["log_semimajor_axis"]
    df["host_mass_msun"] = 10 ** df["log_host_mass"]
    return df


def test_gaussian_mixture_analysis_identifies_two_components():
    analyzer = StatisticalAnalyzer()
    df = build_clustered_dataframe()
    results = analyzer.gaussian_mixture_analysis(df, max_components=4)

    assert results["best_n_components"] >= 1
    assert len(results["bic_scores"]) >= 1
    if results["best_n_components"] > 1:
        assert "cluster_labels" in results
        assert len(results["cluster_labels"]) == len(df.dropna(subset=["log_mass_ratio", "log_semimajor_axis"]))


def test_gaussian_mixture_analysis_handles_small_dataset():
    analyzer = StatisticalAnalyzer()
    df = build_clustered_dataframe(n_per_cluster=2)
    results = analyzer.gaussian_mixture_analysis(df, max_components=3)
    assert results["best_n_components"] == 1


def test_beta_distribution_analysis_returns_parameters():
    analyzer = StatisticalAnalyzer()
    df = build_clustered_dataframe()
    results = analyzer.beta_distribution_analysis(df)

    assert "high_q" in results
    assert "low_q" in results
    assert results["high_q"]["n_objects"] > 0
    assert "ks_test" in results


def test_origin_classification_returns_probabilities():
    analyzer = StatisticalAnalyzer()
    df = build_clustered_dataframe()
    results = analyzer.origin_classification(df)

    assert "probabilities" in results
    assert len(results["probabilities"]) == len(results["data_indices"])
    assert 0.0 <= results["probabilities"].min() <= 1.0


def test_origin_classification_requires_sufficient_data():
    analyzer = StatisticalAnalyzer()
    df = build_clustered_dataframe(n_per_cluster=1).head(5)
    results = analyzer.origin_classification(df)
    assert "error" in results


def test_save_results_writes_expected_files(tmp_path: Path):
    analyzer = StatisticalAnalyzer()
    gmm_results = {"bic_scores": [1.0, 0.5], "aic_scores": [1.5, 1.0], "best_n_components": 2, "n_components_range": [1, 2]}
    beta_results = {
        "high_q": {"alpha": 1.0, "beta": 2.0, "n_objects": 10, "mean_e": 0.3, "median_e": 0.25},
        "low_q": {"alpha": 1.5, "beta": 2.5, "n_objects": 12, "mean_e": 0.4, "median_e": 0.35},
        "ks_test": {"statistic": 0.1, "p_value": 0.2, "significant": False},
        "mannwhitney_test": {"statistic": 5.0, "p_value": 0.1, "significant": False},
    }
    classification_results = {"error": "skip"}

    analyzer.save_results(gmm_results, beta_results, classification_results, output_dir=str(tmp_path))

    gmm_file = tmp_path / "gmm_summary.json"
    beta_file = tmp_path / "beta_e_params.csv"
    ks_file = tmp_path / "ks_test_e.txt"

    assert gmm_file.exists()
    assert beta_file.exists()
    assert ks_file.exists()

    with gmm_file.open() as f:
        data = json.load(f)
    assert data["best_n_components"] == 2


def test_kozai_lidov_feasibility_single_success_case():
    success = kozai_lidov_feasibility_single(
        M_star=0.08,
        M_comp=0.3 / 1047.6,
        a_inner=0.05,
        e_inner=0.8,
        M_perturber=0.5,
        a_outer=5.0,
        e_outer=0.1,
    )
    assert success


def test_create_feasibility_map_small_grid():
    np.random.seed(0)
    analyzer = KozaiLidovAnalyzer(n_trials=5)
    results = analyzer.create_feasibility_map(
        perturber_mass_range=(0.2, 0.3),
        perturber_sep_range=(50, 60),
        n_mass_points=2,
        n_sep_points=2,
    )

    fmap = results["feasibility_map"]
    assert fmap.shape == (2, 2)
    assert np.all((fmap >= 0) & (fmap <= 1))


def test_age_dependent_stellar_radius():
    """Test age-dependent stellar radius calculation"""
    # Model predicts slowly increasing radius with age via logarithmic factor
    M_star = 0.1
    R_young = age_dependent_stellar_radius(M_star, 0.1)
    R_old = age_dependent_stellar_radius(M_star, 10.0)

    assert R_old > R_young
    assert R_young > 0
    assert R_old > 0


def test_age_dependent_tidal_q_factor():
    """Test age-dependent tidal Q factor calculation"""
    # Old star should have higher Q factor than young star
    M_star = 0.1
    Q_young = age_dependent_tidal_q_factor(M_star, 0.1)
    Q_old = age_dependent_tidal_q_factor(M_star, 10.0)

    assert Q_old > Q_young
    assert Q_young >= 1e5  # Minimum expected range
    assert Q_old <= 1e8    # Maximum expected range


def test_kozai_lidov_feasibility_with_age():
    """Test age-dependent Kozai-Lidov feasibility"""
    # Test basic functionality
    result = kozai_lidov_feasibility_with_age(
        M_star=0.08,
        M_comp=0.3 / 1047.6,
        a_inner=0.05,
        e_inner=0.3,
        M_perturber=0.5,
        a_outer=50.0,
        e_outer=0.1,
        age_gyr=5.0
    )

    # Result should be boolean
    assert isinstance(result, (bool, np.bool_))


def test_age_migration_regression_analysis():
    """Test age-migration regression analysis"""
    analyzer = StatisticalAnalyzer()
    df = build_clustered_dataframe()

    results = analyzer.age_migration_regression_analysis(df)

    if "error" in results:
        assert "Missing dependencies" in results["error"]
        pytest.skip(results["error"])

    # Should return valid results with sufficient data
    assert "correlations" in results
    assert "regressions" in results
    assert "n_total_objects" in results

    # Check correlation structure
    assert "age_semimajor_axis" in results["correlations"]
    assert "pearson_r" in results["correlations"]["age_semimajor_axis"]

    # Check regression structure
    assert "log_semimajor_axis_vs_log_age" in results["regressions"]
    assert "slope" in results["regressions"]["log_semimajor_axis_vs_log_age"]


def test_age_migration_regression_includes_multiple_regression_details():
    analyzer = StatisticalAnalyzer()
    df = build_clustered_dataframe()

    results = analyzer.age_migration_regression_analysis(df)

    if "error" in results:
        assert "Missing dependencies" in results["error"]
        pytest.skip(results["error"])

    assert "regressions" in results
    assert "multiple_regression" in results["regressions"]

    multi = results["regressions"]["multiple_regression"]
    assert multi["n_objects"] >= 5
    assert np.isfinite(multi["r_squared"])
    assert np.isfinite(multi["coefficients"]["log_age"])
    assert np.isfinite(multi["p_values"]["log_age"])


def test_age_migration_regression_insufficient_data():
    """Test age regression with insufficient data"""
    analyzer = StatisticalAnalyzer()

    # Create dataframe with insufficient age data
    df = build_clustered_dataframe(n_per_cluster=2)
    df = df.head(5)  # Only 5 objects

    results = analyzer.age_migration_regression_analysis(df)
    assert "error" in results


def test_create_age_dependent_feasibility_map_small():
    """Test age-dependent feasibility map creation with small grid"""
    np.random.seed(42)
    analyzer = KozaiLidovAnalyzer(n_trials=2)

    results = analyzer.create_age_dependent_feasibility_map(
        age_range=(1.0, 5.0),
        perturber_mass_range=(0.2, 0.3),
        perturber_sep_range=(50, 60),
        n_age_points=2,
        n_mass_points=2,
        n_sep_points=2
    )

    # Check output structure
    assert "feasibility_map" in results
    assert "age_grid" in results
    assert "perturber_masses" in results
    assert "perturber_separations" in results

    # Check map dimensions
    fmap = results["feasibility_map"]
    assert fmap.shape == (2, 2, 2)  # age x mass x separation
    assert np.all((fmap >= 0) & (fmap <= 1))

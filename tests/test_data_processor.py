import numpy as np
import pandas as pd

from data_processor import VLMSDataProcessor


def make_processor():
    return VLMSDataProcessor(min_stellar_mass=0.06, max_stellar_mass=0.20)


def test_process_nasa_data_converts_jupiter_masses_and_filters():
    df = pd.DataFrame({
        "pl_name": ["A", "B"],
        "hostname": ["HostA", "HostB"],
        "st_mass": [0.07, 0.25],
        "pl_masse": [np.nan, 100],
        "pl_massj": [0.5, 0.3],
        "pl_orbsmax": [0.05, 0.2],
    })

    processor = make_processor()
    processed = processor.process_nasa_data(df)

    # Only first row within mass window
    assert len(processed) == 1
    row = processed.iloc[0]
    assert np.isclose(row["companion_mass_mearth"], 0.5 * 317.8)
    assert row["data_source"] == "NASA"


def test_process_bd_data_maps_columns_and_converts_units():
    df = pd.DataFrame({
        "M_star": [0.08, 0.10],
        "M_comp": [1.0, 2.0],  # Jupiter masses, should convert to Earth masses
        "a_au": [0.05, 0.07],
        "e": [0.2, 0.1],
    })

    processor = make_processor()
    processed = processor.process_bd_data(df)

    assert len(processed) == 2
    assert (processed["companion_mass_mearth"] > 300).all()
    assert (processed["data_source"] == "BD_Catalogue").all()


def test_combine_datasets_preserves_union_of_columns():
    processor = make_processor()

    nasa = pd.DataFrame({
        "host_mass_msun": [0.1],
        "companion_mass_mearth": [100],
        "semimajor_axis_au": [0.05],
        "extra_nasa": [1],
    })
    bd = pd.DataFrame({
        "host_mass_msun": [0.12],
        "companion_mass_mearth": [120],
        "semimajor_axis_au": [0.06],
        "extra_bd": [2],
    })

    combined = processor.combine_datasets(nasa, bd)

    assert {
        "host_mass_msun",
        "companion_mass_mearth",
        "semimajor_axis_au",
        "extra_nasa",
        "extra_bd",
    }.issubset(combined.columns)
    assert len(combined) == 2
    assert pd.isna(combined.loc[0, "extra_bd"])
    assert pd.isna(combined.loc[1, "extra_nasa"])


def test_compute_derived_quantities_adds_expected_columns():
    processor = make_processor()
    df = pd.DataFrame({
        "host_mass_msun": [0.1],
        "companion_mass_mearth": [317.8],
        "semimajor_axis_au": [0.1],
        "eccentricity": [np.nan],
    })

    result = processor.compute_derived_quantities(df)

    expected_cols = {
        "companion_mass_mjup",
        "companion_mass_msun",
        "mass_ratio",
        "log_mass_ratio",
        "log_semimajor_axis",
        "log_host_mass",
        "above_deuterium_limit",
        "high_mass_ratio",
        "eccentricity",
    }
    assert expected_cols.issubset(result.columns)
    assert np.isclose(result.iloc[0]["companion_mass_mjup"], 1.0)
    assert result.iloc[0]["eccentricity"] == 0.0


def test_add_toi6894b_appends_entry():
    processor = make_processor()
    df = pd.DataFrame({
        "host_mass_msun": [0.1],
        "companion_mass_mearth": [317.8],
        "semimajor_axis_au": [0.1],
        "mass_ratio": [0.01],
        "log_mass_ratio": [-2],
        "log_semimajor_axis": [-1],
        "log_host_mass": [-1],
        "eccentricity": [0.1],
        "companion_mass_mjup": [1.0],
        "companion_mass_msun": [0.001],
        "above_deuterium_limit": [False],
        "high_mass_ratio": [True],
        "data_source": ["NASA"],
    })

    updated = processor.add_toi6894b(df)

    assert len(updated) == 2
    toi_row = updated[updated["data_source"] == "TOI"].iloc[0]
    assert toi_row["companion_name"] == "TOI-6894b"
    assert np.isclose(toi_row["companion_mass_mjup"], 0.3)

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import panoptic_vlms_project as pipeline


def test_setup_output_directory_creates_path(tmp_path: Path):
    outdir = tmp_path / "results"
    pipeline.setup_output_directory(str(outdir))
    assert outdir.exists()


def test_process_data_returns_dataset_with_toi_entry(tmp_path: Path):
    nasa_data = pd.DataFrame({
        "pl_name": ["Planet X"],
        "hostname": ["Star X"],
        "st_mass": [0.1],
        "pl_masse": [100],
        "pl_orbsmax": [0.05],
        "pl_orbeccen": [0.1],
        "discoverymethod": ["Transit"],
    })

    bd_data = pd.DataFrame({
        "M_star": [0.12],
        "M_comp": [120],
        "a_au": [0.07],
        "e": [0.2],
    })

    args = SimpleNamespace(
        toi_mstar=0.08,
        toi_mc_mj=0.3,
        toi_a_AU=0.05,
        toi_ecc=0.0,
        outdir=str(tmp_path),
    )

    final_df = pipeline.process_data(nasa_data, bd_data, args)

    assert "TOI" in final_df["data_source"].values
    toi_rows = final_df[final_df["data_source"] == "TOI"]
    assert np.isclose(toi_rows.iloc[0]["companion_mass_mjup"], 0.3)


def test_create_visualizations_uses_visualizer(monkeypatch, tmp_path: Path):
    calls = []

    class DummyViz:
        def plot_mass_mass_diagram(self, df, output_file):
            calls.append(("mass", output_file))

        def plot_architecture_diagram(self, df, output_file):
            calls.append(("arch", output_file))

    monkeypatch.setattr(pipeline, "VLMSVisualizer", lambda: DummyViz())

    df = pd.DataFrame({
        "host_mass_msun": [0.1],
        "companion_mass_mjup": [1.0],
        "semimajor_axis_au": [0.05],
        "eccentricity": [0.1],
        "log_mass_ratio": [-2.0],
        "high_mass_ratio": [True],
        "data_source": ["NASA"],
    })

    args = SimpleNamespace(outdir=str(tmp_path))
    fig1, fig2 = pipeline.create_visualizations(df, args)

    assert calls == [("mass", fig1), ("arch", fig2)]


def test_save_object_probabilities_writes_expected_file(tmp_path: Path):
    df = pd.DataFrame({
        "companion_name": ["Obj1", "Obj2", "TOI-6894b"],
        "host_name": ["Star1", "Star2", "TOI-6894"],
        "host_mass_msun": [0.1, 0.12, 0.08],
        "companion_mass_mjup": [1.0, 2.0, 0.3],
        "semimajor_axis_au": [0.05, 0.07, 0.05],
        "eccentricity": [0.1, 0.2, 0.0],
        "mass_ratio": [0.01, 0.02, 0.003],
        "data_source": ["NASA", "BD_Catalogue", "TOI"],
        "discovery_method": ["Transit", "RV", "Transit"],
        "high_mass_ratio": [True, False, False],
    })

    classification_results = {
        "probabilities": np.array([0.2, 0.8]),
        "data_indices": [0, 2],
    }

    args = SimpleNamespace(outdir=str(tmp_path))
    pipeline.save_object_probabilities(df, classification_results, args)

    output_file = Path(args.outdir) / "objects_with_probs.csv"
    assert output_file.exists()
    saved = pd.read_csv(output_file)
    assert "P_binary_like" in saved.columns
    saved_probs = saved["P_binary_like"].tolist()
    assert saved_probs[0] == pytest.approx(0.2)
    assert np.isnan(saved_probs[1])
    assert saved_probs[2] == pytest.approx(0.8)


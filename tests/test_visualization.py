import numpy as np
import pandas as pd

from visualization import VLMSVisualizer


def make_dataframe():
    return pd.DataFrame({
        "host_mass_msun": [0.1, 0.12, 0.08],
        "companion_mass_mjup": [1.0, 5.0, 0.3],
        "companion_mass_msun": [1.0 / 1047.6, 5.0 / 1047.6, 0.3 / 1047.6],
        "semimajor_axis_au": [0.05, 0.2, 0.03],
        "eccentricity": [0.1, 0.4, 0.0],
        "mass_ratio": [0.01, 0.03, 0.002],
        "log_mass_ratio": [-2.0, -1.5, -2.7],
        "log_semimajor_axis": [-1.3, -0.7, -1.5],
        "data_source": ["NASA", "BD_Catalogue", "TOI"],
        "high_mass_ratio": [True, True, False],
    })


def test_plot_mass_mass_diagram_creates_file(tmp_path):
    viz = VLMSVisualizer()
    df = make_dataframe()
    outfile = tmp_path / "mass_mass.png"
    viz.plot_mass_mass_diagram(df, str(outfile))
    assert outfile.exists()


def test_plot_architecture_diagram_creates_file(tmp_path):
    viz = VLMSVisualizer()
    df = make_dataframe()
    outfile = tmp_path / "architecture.png"
    viz.plot_architecture_diagram(df, str(outfile))
    assert outfile.exists()


def test_plot_feasibility_map_creates_file(tmp_path):
    viz = VLMSVisualizer()
    data = np.random.rand(3, 3)
    masses = np.array([0.1, 0.2, 0.3])
    seps = np.array([10, 50, 100])
    outfile = tmp_path / "feasibility.png"
    viz.plot_feasibility_map(data, masses, seps, str(outfile))
    assert outfile.exists()


def test_plot_gmm_analysis_handles_single_component(tmp_path):
    viz = VLMSVisualizer()
    df = make_dataframe()
    results = {
        "bic_scores": [100.0, 90.0],
        "cluster_labels": np.array([0, 0, 0]),
    }
    outfile = tmp_path / "gmm.png"
    viz.plot_gmm_analysis(df, results, str(outfile))
    assert outfile.exists()


def test_plot_classification_results_creates_file(tmp_path):
    viz = VLMSVisualizer()
    df = make_dataframe()
    probs = np.array([0.2, 0.8, 0.5])
    outfile = tmp_path / "class.png"
    viz.plot_classification_results(df, probs, str(outfile))
    assert outfile.exists()



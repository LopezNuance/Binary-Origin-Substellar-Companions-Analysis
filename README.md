# VLMS Companion Analysis System

A complete data analysis pipeline for studying very low mass star (VLMS) companions to test the hypothesis that objects like TOI-6894b originate from mass-asymmetric cloud fragmentation rather than traditional disk-based planet formation.

## Overview

This system implements the analysis described in `improved-small-star-big-planet-paper-with-code.md`, providing:

- **Data fetching** from NASA Exoplanet Archive and Brown Dwarf Companion Catalogue
- **Statistical analysis** including Gaussian Mixture Models, Beta distributions, and classification
- **Orbital dynamics** feasibility mapping for Kozai-Lidov + tides migration
- **Visualization** of companion demographics and orbital architecture
- **Quantitative assessment** of binary-like origin probabilities

## Installation

### Requirements

```bash
pip install numpy scipy pandas scikit-learn statsmodels matplotlib requests numba
```

### Quick Setup

```bash
git clone <repository>
cd small-star_big-planet
pip install -r requirements.txt
```

## Usage

### Fetch Fresh Data and Run Full Analysis

```bash
python panoptic_vlms_project.py --fetch --outdir results
```

### Use Local Data Files

```bash
python panoptic_vlms_project.py --ps pscomppars_lowM.csv --bd BD_catalogue.csv --outdir results
```

### Customize TOI-6894b Parameters

```bash
python panoptic_vlms_project.py --fetch --toi_mstar 0.08 --toi_mc_mj 0.3 --toi_a_AU 0.05 --outdir results
```

### Logging Options

```bash
python panoptic_vlms_project.py --fetch --outdir results \
  --logdir logs/run_outputs --errordir logs/run_errors \
  --log_basename vlms_pipeline --error_basename vlms_pipeline_error
```

- Each run mirrors stdout to a timestamped log file (default: `logs/panoptic_vlms_<timestamp>.log`).
- Errors are additionally captured in a paired file (default: `errors/panoptic_vlms_error_<timestamp>.log`).
- Omit the switches to accept the defaults or point them at project-specific directories.

## Outputs

The system generates:

### Figures
- `fig1_massmass.png` - Host vs companion mass diagram with deuterium/hydrogen burning limits
- `fig2_ae.png` - Eccentricity vs semi-major axis architecture plot
- `fig3_feasibility.png` - Kozai-Lidov + tides migration feasibility map
- Additional analysis plots (GMM clustering, classification results)

### Data Files
- `vlms_companions_stacked.csv` - Combined processed dataset
- `objects_with_probs.csv` - Objects with binary-like origin probabilities
- `feasibility_map.npz` - Migration feasibility simulation results

### Analysis Results
- `gmm_summary.json` - Gaussian Mixture Model BIC scores and components
- `beta_e_params.csv` - Beta distribution parameters for eccentricity groups
- `ks_test_e.txt` - Kolmogorov-Smirnov test results
- `SUMMARY.txt` - Complete analysis summary with key findings

### Structured Message Passing

Commit `abbc55e` introduced structured hand-offs between the statistical and visualization models by serializing Gaussian Mixture outputs to `gmm_summary.json` and companion metadata to CSV. Downstream tooling (including the plotting routines exercised in commit `004df6e`) consumes these JSON payloads to render figures and validate model behavior without re-running the full analysis, providing a stable message-passing contract between components.

## System Architecture

### Core Modules

- **`data_fetchers.py`** - NASA TAP API and Brown Dwarf Catalogue interfaces
- **`data_processor.py`** - Data cleaning, filtering, and feature engineering
- **`visualization.py`** - Scientific plotting and figure generation
- **`statistical_analysis.py`** - GMM, Beta distributions, classification, Kozai-Lidov modeling
- **`panoptic_vlms_project.py`** - Main pipeline with CLI interface

### Analysis Pipeline

1. **Data Acquisition** - Fetch VLMS host data (0.06-0.20 M☉) from catalogs
2. **Preprocessing** - Standardize columns, compute mass ratios, orbital parameters
3. **Statistical Analysis** - GMM clustering, eccentricity modeling, origin classification
4. **Migration Modeling** - Kozai-Lidov + tides feasibility mapping
5. **Visualization** - Generate publication-ready figures
6. **Reporting** - Compile quantitative summary with source URLs

## Key Scientific Results

The analysis provides evidence for mass-asymmetric cloud fragmentation:

- **Demographic continuity** across the deuterium burning limit in VLMS companions
- **Eccentricity structure** showing higher-e bias in high mass-ratio objects
- **Migration feasibility** demonstrating plausible routes to close orbits via dynamical processing
- **Origin classification** providing quantitative binary-like probabilities

## Performance Notes

- **Data fetching**: ~1-2 minutes depending on catalog sizes and connection
- **Statistical analysis**: ~2-5 minutes for full dataset
- **Kozai-Lidov mapping**: ~5-10 minutes for 25×25 grid with 2000 trials per point
- **Total runtime**: ~10-20 minutes for complete analysis

For better performance on multi-core systems:
```bash
export NUMBA_NUM_THREADS=<number_of_cores>
```

## Data Sources

- [NASA Exoplanet Archive (PSCompPars)](https://exoplanetarchive.ipac.caltech.edu/TAP)
- [Brown Dwarf Companion Catalogue](https://ordo.open.ac.uk/articles/dataset/Brown_Dwarf_Companion_Catalogue/24156393)

## Citation

If you use this code in your research, please cite:

```
Johnson, R.S. (2025). Binary-Origin Substellar Companions Around M Dwarfs: Evidence from Demographics, Orbital Architecture, and Migration Timescales.
```

## License

This project is designed for academic research. Please cite appropriately if used in publications.

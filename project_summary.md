# VLMS Companion Analysis System - Project Summary

## Overview

This project implements a comprehensive analysis pipeline for testing the **mass-asymmetric cloud fragmentation** hypothesis as an alternative origin mechanism for planetary-mass companions around very low-mass stars (VLMS). The system analyzes observational data to determine whether TOI-6894b and similar objects originated from turbulent cloud fragmentation followed by post-birth migration, rather than traditional disk-based planet formation.

## Scientific Motivation

The project addresses a fundamental tension in planetary science: close Saturn/Jupiter-mass companions around ultra-low-mass M dwarfs that are difficult to explain through conventional circumstellar disk formation models. The analysis tests whether these objects could instead be products of:

1. **Mass-asymmetric turbulent cloud fragmentation** ("failed binaries")
2. **Post-birth migration** via disk torques and/or Kozai-Lidov cycles with tidal evolution

## Repository Structure

```
small-star_big-planet/
├── source/                    # Core implementation modules
│   ├── panoptic_vlms_project.py   # Main analysis pipeline
│   ├── data_fetchers.py           # Data acquisition from external sources
│   ├── data_processor.py          # Data cleaning and preprocessing
│   ├── statistical_analysis.py    # Statistical modeling and analysis
│   ├── visualization.py           # Plotting and visualization
│   └── md2unicode_math.py         # Mathematical notation utilities
├── tests/                     # Comprehensive test suite
│   ├── conftest.py               # Test configuration
│   ├── test_panoptic_vlms_project.py  # Main pipeline tests
│   ├── test_data_fetchers.py     # Data acquisition tests
│   ├── test_data_processor.py    # Data processing tests
│   ├── test_statistical_analysis.py  # Statistical analysis tests
│   ├── test_visualization.py     # Visualization tests
│   ├── test_command_line_integration.py  # CLI integration tests
│   └── test_md2unicode_math.py   # Utility function tests
├── results/                   # Output directory for analysis artifacts
│   ├── gmm_summary.json          # Gaussian mixture model results
│   ├── beta_e_bootstrap_summary.json  # Eccentricity distribution analysis
│   └── age_regression_summary.json    # Age-orbit relationship analysis
├── README.md                  # Comprehensive project documentation
├── requirements.txt           # Python dependencies
└── design_enhancements.md     # Future enhancement specifications
```

## Core Implementation

### Data Acquisition (`data_fetchers.py`)

**NASAExoplanetArchiveFetcher Class:**
- Fetches data from NASA Exoplanet Archive TAP service
- Implements ADQL queries for VLMS hosts (0.06-0.20 M☉)
- Retrieves: stellar mass, companion mass, orbital parameters, discovery methods
- Includes data validation and temperature/metallicity filtering

**BrownDwarfCatalogueFetcher Class:**
- Fetches Brown Dwarf Companion Catalogue data
- Supports multiple data sources (GitHub, original repository)
- Filters for VLMS hosts using flexible column mapping
- Handles various naming conventions for stellar mass columns

### Data Processing (`data_processor.py`)

**VLMSDataProcessor Class:**
- Combines NASA and Brown Dwarf catalogue data
- Computes derived quantities: mass ratios, logarithmic scaling
- Implements age-based classifications (Young/Intermediate/Old/Unknown)
- Adds TOI-6894b reference object with customizable parameters
- Calculates age deltas relative to TOI-6894b
- Enhanced age analysis features including migration efficiency metrics

**Key Processing Steps:**
1. Data harmonization across different source formats
2. Quality filtering for physical plausibility
3. Mass ratio and semimajor axis logarithmic transformations
4. Age group classification and migration timescale calculations
5. TOI-6894b integration for comparative analysis

### Statistical Analysis (`statistical_analysis.py`)

**StatisticalAnalyzer Class:**
Implements four major statistical analyses:

**1. Gaussian Mixture Model Analysis:**
- Tests 1-5 component models in (log q, log a) space
- Uses BIC for model selection
- Identifies distinct populations suggesting different formation pathways
- Returns cluster labels and probability assignments

**2. Eccentricity Distribution Analysis:**
- Models eccentricity distributions using Beta distributions
- Splits data at q=0.01 threshold (high-q vs low-q subsets)
- Performs Kolmogorov-Smirnov two-sample tests
- Includes bootstrap validation (500 resamples, 80% sampling fraction)
- Tests for significant differences between formation pathways

**3. Age-Migration Regression Analysis:**
- Pearson and Spearman correlations between age and orbital parameters
- Linear regression models: log(a) ~ log(age) and e ~ log(age)
- Multiple regression incorporating stellar mass effects
- Statistical significance testing for migration trends

**4. Origin Classification:**
- Regularized logistic regression model
- Features: log q, log a, eccentricity, log M_star, metallicity, discovery method
- 5-fold cross-validation with AUROC scoring
- Provides P(binary-like) probabilities for each object
- Includes TOI-6894b classification

**KozaiLidovAnalyzer Class:**
- Monte Carlo simulation of Kozai-Lidov + tidal migration
- Explores perturber parameter space (mass: 0.1-1.0 M☉, separation: 10-1000 AU)
- Calculates migration timescales and feasibility fractions
- Uses 2000 trials per grid point for statistical robustness
- Accounts for orbital eccentricity and inclination distributions

### Visualization (`visualization.py`)

**VLMSVisualizer Class:**
Generates comprehensive figure set:

1. **Mass-Mass Diagram** (`fig1_massmass.png`): Host mass vs companion mass with deuterium limit
2. **Architecture Diagram** (`fig2_ae.png`): Eccentricity vs semimajor axis, colored by mass ratio
3. **Feasibility Map** (`fig3_feasibility.png`): Kozai-Lidov migration success rates
4. **GMM Analysis Plot**: Cluster visualization in (log q, log a) space
5. **Classification Results**: Binary-like probability distributions

### Main Pipeline (`panoptic_vlms_project.py`)

**Command-Line Interface:**
- Interactive candidate counting with percentage selection
- Non-interactive batch processing modes
- Local file processing capabilities
- Customizable TOI-6894b parameters
- Comprehensive logging system

**Analysis Workflow:**
1. Data fetching/loading with optional percentage sampling
2. Data processing and TOI-6894b integration
3. Statistical analysis suite execution
4. Kozai-Lidov feasibility mapping
5. Visualization generation
6. Summary report creation

## Testing Infrastructure

The project includes a comprehensive test suite with ~90% code coverage:

- **Unit Tests**: Individual component testing for all major classes
- **Integration Tests**: End-to-end pipeline validation
- **Mock Testing**: External API call simulation
- **Property Testing**: Edge case and boundary condition validation
- **Command-Line Testing**: Argument parsing and validation

## Dependencies and Requirements

**Core Scientific Stack:**
- **NumPy** (≥1.21.0): Numerical computations and array operations
- **SciPy** (≥1.7.0): Statistical distributions and optimization
- **Pandas** (≥1.3.0): Data manipulation and analysis
- **Scikit-learn** (≥1.0.0): Machine learning algorithms
- **Statsmodels** (≥0.13.0): Statistical modeling
- **Matplotlib** (≥3.5.0): Plotting and visualization

**Performance Optimization:**
- **Numba** (≥0.56.0): JIT compilation for Monte Carlo simulations
- **ThreadPoolCTL** (≥3.6.0): Thread management for BLAS operations

**Data Acquisition:**
- **Requests** (≥2.25.0): HTTP requests for external data sources

## Current Analysis Capabilities

### Demographics Analysis
- Companion mass distribution analysis across deuterium burning limit
- Host-companion mass relationship characterization
- Discovery method bias assessment

### Orbital Architecture
- Eccentricity distribution modeling by mass ratio regime
- Semimajor axis distribution analysis
- Statistical comparison of high-q vs low-q populations

### Migration Feasibility
- Kozai-Lidov cycle timescale calculations
- Tidal evolution modeling
- Perturber parameter space exploration
- Migration success rate quantification

### Age Analysis
- Host age correlation with orbital parameters
- Age-dependent migration efficiency assessment
- Comparative analysis relative to TOI-6894b

### Classification Framework
- Quantitative origin probability assignment
- Binary-like vs planet-like classification
- Cross-validated model performance metrics

## Output Artifacts

**Data Products:**
- `vlms_companions_stacked.csv`: Combined processed dataset
- `objects_with_probs.csv`: Classification probabilities for all objects
- `age_comparison.csv`: Age-orbit relationship data

**Statistical Results:**
- `gmm_summary.json`: Gaussian mixture model results
- `beta_e_params.csv`: Eccentricity distribution parameters
- `beta_e_bootstrap_summary.json`: Bootstrap validation results
- `age_regression_summary.json`: Age-migration analysis results
- `ks_test_e.txt`: Kolmogorov-Smirnov test results

**Visualizations:**
- `fig1_massmass.png`: Host-companion mass diagram
- `fig2_ae.png`: Orbital architecture visualization
- `fig3_feasibility.png`: Migration feasibility heatmap
- `gmm_analysis.png`: Cluster analysis results
- `classification_results.png`: Origin classification visualization

**Analysis Summary:**
- `SUMMARY.txt`: Comprehensive results summary
- `feasibility_map.npz`: Migration simulation raw data

## Key Scientific Findings

Based on current implementation and typical analysis runs:

1. **Population Structure**: GMM analysis reveals distinct populations in (log q, log a) space
2. **Eccentricity Differences**: High-q and low-q objects show different eccentricity distributions
3. **Migration Viability**: Kozai-Lidov + tidal migration can explain current orbital configurations
4. **Age Correlations**: Systematic relationships between stellar age and orbital architecture
5. **Classification Success**: Quantitative origin classification with cross-validated performance

## Technical Performance

- **Computational Efficiency**: Numba-accelerated Monte Carlo simulations
- **Memory Management**: Streaming data processing for large datasets
- **Scalability**: Percentage-based sampling for computational resource management
- **Reproducibility**: Fixed random seeds and comprehensive logging
- **Error Handling**: Robust exception handling with graceful degradation

## Current Limitations

1. **Sample Size**: Limited by available VLMS companion detections
2. **Selection Effects**: Discovery method biases not fully corrected
3. **Age Uncertainties**: Limited stellar age measurements in catalog
4. **Migration Physics**: Simplified analytical models vs full N-body evolution
5. **Metallicity Coverage**: Sparse metallicity measurements for host stars

This implementation provides a statistically rigorous, reproducible framework for testing the mass-asymmetric fragmentation hypothesis and represents a significant contribution to understanding companion formation around very low-mass stars.
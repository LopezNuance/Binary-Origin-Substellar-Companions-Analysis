# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hybrid scientific research project combining computational data analysis with academic paper writing. The project tests the hypothesis that TOI-6894b and similar substellar companions around very low-mass stars (VLMS) originate from mass-asymmetric cloud fragmentation (failed binary formation) rather than traditional disk-based planet formation.

The project consists of:
1. **Python analysis pipeline** - Comprehensive data processing and statistical analysis system
2. **LaTeX academic paper** - Scientific manuscript presenting the research findings

## Project Structure

### Python Components
- `source/panoptic_vlms_project.py` - Main analysis pipeline and CLI entry point
- `source/data_fetchers.py` - NASA Exoplanet Archive and Brown Dwarf Catalogue data fetchers
- `source/data_processor.py` - VLMS data processing and companion analysis
- `source/statistical_analysis.py` - Statistical methods including beta distribution analysis
- `source/visualization.py` - Plotting and visualization tools
- `source/md2unicode_math.py` - LaTeX/Unicode math conversion utilities
- `tests/` - Comprehensive test suite (38+ tests)
- `requirements.txt` - Python dependencies
- `README.md` - Detailed project documentation and usage instructions

### LaTeX Academic Paper
- `papers/TOI-6894b_as_a_Failed_Binary_Companion.tex` - Main LaTeX manuscript
- `papers/TOI-6894b as a Failed Binary Companion.bib` - Bibliography file
- `papers/toi6894.bib` - Symlink to bibliography file
- `papers/Matters Arising manuscript.pdf` - Compiled PDF output
- `papers/*.aux`, `papers/*.bbl`, etc. - LaTeX compilation artifacts

### Additional Files
- `Binary-Origin Substellar Companions Around M Dwarf - Evidence from Demographics, Orbital Architecture, and Migration Timescales - outline.md` - Paper outline
- `improved-small-star-big-planet-paper-with-code.md` - Project specifications

## Common Commands

### Python Analysis Pipeline
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete analysis pipeline
python source/panoptic_vlms_project.py --fetch --output-dir results/

# Run tests
pytest tests/

# Static analysis
python -m mypy source/
python -m pylint source/
```

### LaTeX Document Building
```bash
cd papers/
pdflatex "TOI-6894b_as_a_Failed_Binary_Companion.tex"
bibtex "TOI-6894b_as_a_Failed_Binary_Companion"
pdflatex "TOI-6894b_as_a_Failed_Binary_Companion.tex"
pdflatex "TOI-6894b_as_a_Failed_Binary_Companion.tex"
```

Or use latexmk for automated handling:
```bash
cd papers/
latexmk -pdf "TOI-6894b_as_a_Failed_Binary_Companion.tex"
```

### Cleaning Build Files
```bash
cd papers/
latexmk -c "TOI-6894b_as_a_Failed_Binary_Companion.tex"  # Clean auxiliary files
latexmk -C "TOI-6894b_as_a_Failed_Binary_Companion.tex"  # Clean all generated files including PDF
```

## Data Sources

The Python pipeline fetches data from:
- NASA Exoplanet Archive TAP service (PSCompPars table)
- Brown Dwarf Companion Catalogue (Open University dataset)

Analysis focuses on VLMS hosts (0.06-0.20 M☉) with companions having well-determined masses, orbits, and eccentricities.

## Scientific Methodology

The project implements:
1. **Demographics analysis** - Testing for bimodality in (log q, log a) space
2. **Orbital architecture studies** - Eccentricity distributions as a function of semi-major axis
3. **Migration feasibility** - Kozai-Lidov cycles + tidal evolution modeling
4. **Classification system** - Probabilistic origin assignment for individual systems

## LaTeX Configuration

The academic manuscript uses:
- 12pt article class with letter paper and 1-inch margins
- authblk package for author affiliations
- natbib for citation management with plainnat style
- Times font family
- Standard scientific packages (amsmath, amsfonts, graphicx, geometry)
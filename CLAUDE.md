# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LaTeX academic manuscript project for an astronomical research paper titled "Reconsidering the Origin of TOI-6894b as a Failed Binary Companion" by R. Scott Johnson. The project contains a scientific comment/response paper challenging the interpretation of TOI-6894b as a planet, instead proposing it as a failed binary companion.

## Project Structure

- `comment.tex` - Main LaTeX document containing the scientific manuscript
- `toi6894.bib` - Bibliography file with academic references
- `Matters Arising manuscript.pdf` - Compiled PDF output
- `comment.*` auxiliary files - LaTeX compilation artifacts (aux, bbl, blg, log, fls, fdb_latexmk)
- `toi6894-matters-arising.tar.gz` - Archive file (likely for journal submission)

## Common Commands

### Building the Document
- `pdflatex comment.tex` - Compile LaTeX to PDF
- `bibtex comment` - Process bibliography
- `pdflatex comment.tex` (run twice after bibtex for proper references)
- `latexmk -pdf comment.tex` - Automated compilation with proper dependency handling

### Complete Build Process
```bash
pdflatex comment.tex
bibtex comment
pdflatex comment.tex
pdflatex comment.tex
```

Or use latexmk for automated handling:
```bash
latexmk -pdf comment.tex
```

### Cleaning Build Files
```bash
latexmk -c comment.tex  # Clean auxiliary files
latexmk -C comment.tex  # Clean all generated files including PDF
```

## Document Structure

The manuscript follows standard academic paper format:
- Abstract summarizing the main argument
- Main text presenting the alternative interpretation
- Conclusion section
- Bibliography using natbib package with plainnat style

## LaTeX Configuration

The document uses:
- 12pt article class
- A4 paper with 1-inch margins
- authblk package for author affiliations
- natbib for citation management
- Times font
- Standard academic packages (graphicx, geometry)
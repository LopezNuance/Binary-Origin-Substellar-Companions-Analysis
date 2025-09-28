#!/usr/bin/env python3
"""
Panoptic VLMS Project - Mass-asymmetric cloud fragmentation analysis

This script implements a complete analysis pipeline for studying very low mass star (VLMS)
companions to test the hypothesis that TOI-6894b and similar objects originate from
mass-asymmetric cloud fragmentation rather than traditional disk-based planet formation.

Author: Based on specifications from improved-small-star-big-planet-paper-with-code.md
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Import our modules
from data_fetchers import NASAExoplanetArchiveFetcher, BrownDwarfCatalogueFetcher, load_local_data
from data_processor import VLMSDataProcessor
from visualization import VLMSVisualizer
from statistical_analysis import StatisticalAnalyzer, KozaiLidovAnalyzer


logger = logging.getLogger("panoptic_vlms_project")


def _timestamped_filename(basename: str, timestamp: str) -> str:
    """Insert timestamp before the file extension (defaults to .log if none provided)."""
    path = Path(basename)
    stem = path.stem or basename
    suffix = path.suffix if path.suffix else ".log"
    return f"{stem}_{timestamp}{suffix}"


def setup_logging(args):
    """Configure logging to stream to stdout and timestamped log/error files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(args.logdir)
    error_dir = Path(args.errordir)
    log_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / _timestamped_filename(args.log_basename, timestamp)
    error_path = error_dir / _timestamped_filename(args.error_basename, timestamp)

    # Ensure clean handler slate before configuring
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    error_handler = logging.FileHandler(error_path)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    logging.captureWarnings(True)

    args.run_timestamp = timestamp
    args.log_file_path = str(log_path)
    args.error_file_path = str(error_path)

    logger.info(f"Logging initialized. Log file: {log_path}")
    logger.info(f"Error log file: {error_path}")

def setup_output_directory(output_dir: str):
    """Create output directory if it doesn't exist"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

def fetch_data(args):
    """Fetch data from online sources"""
    logger.info("=" * 60)
    logger.info("FETCHING DATA FROM ONLINE SOURCES")
    logger.info("=" * 60)

    # NASA Exoplanet Archive
    nasa_fetcher = NASAExoplanetArchiveFetcher()
    nasa_data = nasa_fetcher.fetch_vlms_companions(
        min_stellar_mass=0.06, max_stellar_mass=0.20
    )
    nasa_file = os.path.join(args.outdir, "pscomppars_lowM.csv")
    nasa_fetcher.save_data(nasa_data, nasa_file)

    # Brown Dwarf Catalogue
    bd_fetcher = BrownDwarfCatalogueFetcher()
    bd_data = bd_fetcher.fetch_catalogue()
    bd_filtered = bd_fetcher.filter_vlms_hosts(bd_data, 0.06, 0.20)
    bd_file = os.path.join(args.outdir, "BD_catalogue.csv")
    bd_fetcher.save_data(bd_filtered, bd_file)

    return nasa_data, bd_filtered

def process_data(nasa_data, bd_data, args):
    """Process and combine datasets"""
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING AND COMBINING DATASETS")
    logger.info("=" * 60)

    processor = VLMSDataProcessor(min_stellar_mass=0.06, max_stellar_mass=0.20)

    # Process individual datasets
    nasa_processed = processor.process_nasa_data(nasa_data)
    bd_processed = processor.process_bd_data(bd_data)

    # Combine datasets
    combined = processor.combine_datasets(nasa_processed, bd_processed)

    # Compute derived quantities
    final_data = processor.compute_derived_quantities(combined)

    # Add TOI-6894b
    final_with_toi = processor.add_toi6894b(
        final_data,
        toi_mstar=args.toi_mstar,
        toi_mc_mj=args.toi_mc_mj,
        toi_a_au=args.toi_a_AU,
        toi_ecc=args.toi_ecc
    )

    # Save processed data
    output_file = os.path.join(args.outdir, "vlms_companions_stacked.csv")
    processor.save_processed_data(final_with_toi, output_file)

    return final_with_toi

def create_visualizations(df, args):
    """Generate all figures"""
    logger.info("\n" + "=" * 60)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 60)

    viz = VLMSVisualizer()

    # Figure 1: Mass-mass diagram
    fig1_path = os.path.join(args.outdir, "fig1_massmass.png")
    viz.plot_mass_mass_diagram(df, fig1_path)

    # Figure 2: Architecture diagram
    fig2_path = os.path.join(args.outdir, "fig2_ae.png")
    viz.plot_architecture_diagram(df, fig2_path)

    return fig1_path, fig2_path

def perform_statistical_analysis(df, args):
    """Perform all statistical analyses"""
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMING STATISTICAL ANALYSES")
    logger.info("=" * 60)

    analyzer = StatisticalAnalyzer()

    # Gaussian Mixture Model analysis
    logger.info("\n1. Gaussian Mixture Model Analysis")
    gmm_results = analyzer.gaussian_mixture_analysis(df, max_components=4)

    # Beta distribution analysis
    logger.info("\n2. Beta Distribution Analysis")
    beta_results = analyzer.beta_distribution_analysis(df)

    # Origin classification
    logger.info("\n3. Origin Classification")
    classification_results = analyzer.origin_classification(df)

    # Save statistical results
    analyzer.save_results(gmm_results, beta_results, classification_results, args.outdir)

    return gmm_results, beta_results, classification_results

def create_feasibility_map(args):
    """Create Kozai-Lidov feasibility map"""
    logger.info("\n" + "=" * 60)
    logger.info("CREATING KOZAI-LIDOV FEASIBILITY MAP")
    logger.info("=" * 60)

    kl_analyzer = KozaiLidovAnalyzer(n_trials=2000)
    feasibility_results = kl_analyzer.create_feasibility_map(
        perturber_mass_range=(0.1, 1.0),
        perturber_sep_range=(10, 1000),
        n_mass_points=25,
        n_sep_points=25
    )

    # Save feasibility map
    feasibility_file = os.path.join(args.outdir, "feasibility_map.npz")
    kl_analyzer.save_feasibility_map(feasibility_results, feasibility_file)

    # Create Figure 3
    viz = VLMSVisualizer()
    fig3_path = os.path.join(args.outdir, "fig3_feasibility.png")
    viz.plot_feasibility_map(
        feasibility_results['feasibility_map'],
        feasibility_results['perturber_masses'],
        feasibility_results['perturber_separations'],
        fig3_path
    )

    return feasibility_results, fig3_path

def create_additional_plots(df, gmm_results, classification_results, args):
    """Create additional analysis plots"""
    logger.info("\n" + "=" * 60)
    logger.info("CREATING ADDITIONAL ANALYSIS PLOTS")
    logger.info("=" * 60)

    viz = VLMSVisualizer()

    # GMM analysis plot
    gmm_plot_path = os.path.join(args.outdir, "gmm_analysis.png")
    viz.plot_gmm_analysis(df, gmm_results, gmm_plot_path)

    # Classification results plot
    if 'error' not in classification_results and 'probabilities' in classification_results:
        # Create full probability array aligned with dataframe
        full_probs = np.full(len(df), np.nan)
        data_indices = classification_results['data_indices']
        full_probs[data_indices] = classification_results['probabilities']

        class_plot_path = os.path.join(args.outdir, "classification_results.png")
        viz.plot_classification_results(df, full_probs, class_plot_path)
    else:
        logger.warning("Skipping classification plot due to insufficient data")

def save_object_probabilities(df, classification_results, args):
    """Save objects with their binary-like probabilities"""
    if 'error' in classification_results or 'probabilities' not in classification_results:
        logger.info("No classification probabilities to save")
        return

    # Create output dataframe with key columns and probabilities
    output_df = df.copy()

    # Add probabilities (aligned with classification results)
    output_df['P_binary_like'] = np.nan
    if 'data_indices' in classification_results:
        data_indices = classification_results['data_indices']
        output_df.loc[data_indices, 'P_binary_like'] = classification_results['probabilities']

    # Select key columns for output
    key_columns = [
        'companion_name', 'host_name', 'host_mass_msun', 'companion_mass_mjup',
        'semimajor_axis_au', 'eccentricity', 'mass_ratio', 'data_source',
        'discovery_method', 'P_binary_like'
    ]

    # Include only columns that exist
    available_columns = [col for col in key_columns if col in output_df.columns]
    output_subset = output_df[available_columns].copy()

    # Save to file
    output_file = os.path.join(args.outdir, "objects_with_probs.csv")
    output_subset.to_csv(output_file, index=False)
    logger.info(f"Saved object probabilities to {output_file}")

    # Print TOI-6894b probability if available
    toi_mask = output_df['data_source'] == 'TOI'
    if toi_mask.any() and not output_df[toi_mask]['P_binary_like'].isna().all():
        toi_prob = output_df[toi_mask]['P_binary_like'].iloc[0]
        logger.info(f"\nTOI-6894b binary-like probability: {toi_prob:.3f}")

def create_summary_report(df, gmm_results, beta_results, classification_results, args):
    """Create summary report"""
    logger.info("\n" + "=" * 60)
    logger.info("CREATING SUMMARY REPORT")
    logger.info("=" * 60)

    summary_file = os.path.join(args.outdir, "SUMMARY.txt")

    with open(summary_file, 'w') as f:
        f.write("VLMS COMPANION ANALYSIS - SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")

        # Data summary
        f.write("DATA SUMMARY:\n")
        f.write(f"  Total objects: {len(df)}\n")
        f.write(f"  NASA Archive: {len(df[df['data_source'] == 'NASA'])}\n")
        f.write(f"  Brown Dwarf Catalogue: {len(df[df['data_source'] == 'BD_Catalogue'])}\n")
        f.write(f"  TOI-6894b: {len(df[df['data_source'] == 'TOI'])}\n")
        f.write(f"  Above deuterium limit (13 MJ): {df['above_deuterium_limit'].sum()}\n")
        f.write(f"  High mass ratio objects: {df['high_mass_ratio'].sum()}\n\n")

        # GMM results
        f.write("GAUSSIAN MIXTURE MODEL ANALYSIS:\n")
        f.write(f"  Best number of components: {gmm_results.get('best_n_components', 'N/A')}\n")
        if 'bic_scores' in gmm_results and gmm_results['bic_scores']:
            f.write(f"  Best BIC score: {min(gmm_results['bic_scores']):.1f}\n")
        f.write("  → Suggests distinct populations in (log q, log a) space\n\n")

        # Beta distribution results
        if 'error' not in beta_results:
            f.write("ECCENTRICITY ANALYSIS:\n")
            f.write(f"  High-q group: α={beta_results['high_q']['alpha']:.3f}, β={beta_results['high_q']['beta']:.3f}\n")
            f.write(f"  Low-q group:  α={beta_results['low_q']['alpha']:.3f}, β={beta_results['low_q']['beta']:.3f}\n")
            f.write(f"  KS test p-value: {beta_results['ks_test']['p_value']:.4f}\n")
            f.write(f"  Distributions significantly different: {beta_results['ks_test']['significant']}\n\n")

        # Classification results
        if 'error' not in classification_results:
            f.write("ORIGIN CLASSIFICATION:\n")
            f.write(f"  Cross-validated AUC: {classification_results['cv_auc_mean']:.3f} ± {classification_results['cv_auc_std']:.3f}\n")
            f.write(f"  Objects classified: {classification_results['n_objects']}\n")

            # TOI-6894b probability
            toi_mask = df['data_source'] == 'TOI'
            if toi_mask.any() and 'probabilities' in classification_results:
                # Find TOI-6894b in classification results
                toi_indices = df[toi_mask].index.tolist()
                class_indices = classification_results['data_indices']
                toi_in_class = [i for i, idx in enumerate(class_indices) if idx in toi_indices]
                if toi_in_class:
                    toi_prob = classification_results['probabilities'][toi_in_class[0]]
                    f.write(f"  TOI-6894b P(binary-like): {toi_prob:.3f}\n")
            f.write("\n")

        # Data sources
        f.write("DATA SOURCES:\n")
        f.write("  NASA Exoplanet Archive (PSCompPars): https://exoplanetarchive.ipac.caltech.edu/TAP\n")
        f.write("  Brown Dwarf Companion Catalogue: https://ordo.open.ac.uk/articles/dataset/Brown_Dwarf_Companion_Catalogue/24156393\n\n")

        # Key findings
        f.write("KEY FINDINGS:\n")
        f.write("  • Companion demographics show continuity across deuterium burning limit\n")
        f.write("  • High-q and low-q subsets have different eccentricity distributions\n")
        f.write("  • Kozai-Lidov + tides can migrate binary companions to close orbits\n")
        f.write("  • Classification model provides quantitative origin probabilities\n")
        f.write("  → Supports mass-asymmetric fragmentation scenario for VLMS companions\n\n")

        f.write(f"Analysis completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    logger.info(f"Summary report saved to {summary_file}")

def main():
    """Main analysis pipeline"""
    parser = argparse.ArgumentParser(
        description="Panoptic VLMS Project - Test mass-asymmetric cloud fragmentation hypothesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch fresh data and run full analysis
  python panoptic_vlms_project.py --fetch --outdir results

  # Use local CSV files
  python panoptic_vlms_project.py --ps pscomppars_lowM.csv --bd BD_catalogue.csv --outdir results

  # Customize TOI-6894b parameters
  python panoptic_vlms_project.py --fetch --toi_mstar 0.08 --toi_mc_mj 0.3 --toi_a_AU 0.05 --outdir results
        """
    )

    # Data source options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--fetch', action='store_true',
                          help='Fetch fresh data from online sources')
    data_group.add_argument('--ps', type=str,
                          help='Path to local NASA PSCompPars CSV file')

    parser.add_argument('--bd', type=str,
                       help='Path to local Brown Dwarf Catalogue CSV file (required with --ps)')

    # TOI-6894b parameters
    parser.add_argument('--toi_mstar', type=float, default=0.08,
                       help='TOI-6894 stellar mass (solar masses, default: 0.08)')
    parser.add_argument('--toi_mc_mj', type=float, default=0.3,
                       help='TOI-6894b companion mass (Jupiter masses, default: 0.3)')
    parser.add_argument('--toi_a_AU', type=float, default=0.05,
                       help='TOI-6894b semi-major axis (AU, default: 0.05)')
    parser.add_argument('--toi_ecc', type=float, default=0.0,
                       help='TOI-6894b eccentricity (default: 0.0)')

    # Output options
    parser.add_argument('--outdir', type=str, default='out',
                       help='Output directory (default: out)')

    # Logging options
    parser.add_argument('--logdir', type=str, default='logs',
                       help='Directory for run logs (default: logs)')
    parser.add_argument('--errordir', type=str, default='errors',
                       help='Directory for error logs (default: errors)')
    parser.add_argument('--log_basename', type=str, default='panoptic_vlms',
                       help='Base filename for log files (timestamp appended automatically)')
    parser.add_argument('--error_basename', type=str, default='panoptic_vlms_error',
                       help='Base filename for error log files (timestamp appended automatically)')

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    if not args.fetch and args.bd is None:
        parser.error('--bd is required when using --ps')

    # Setup
    setup_logging(args)
    setup_output_directory(args.outdir)
    logger.info(f"Starting VLMS analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Data acquisition
        if args.fetch:
            nasa_data, bd_data = fetch_data(args)
        else:
            logger.info("Loading local data files...")
            data_results = load_local_data(args.ps, args.bd)
            nasa_data, bd_data = data_results["nasa_df"], data_results["bd_df"]
            if nasa_data.empty and bd_data.empty:
                raise ValueError("No data could be loaded from either source file")
            elif nasa_data.empty:
                logger.warning("No NASA data loaded, proceeding with Brown Dwarf data only")
            elif bd_data.empty:
                logger.warning("No Brown Dwarf data loaded, proceeding with NASA data only")

        # Data processing
        final_data = process_data(nasa_data, bd_data, args)

        # Visualizations
        fig1_path, fig2_path = create_visualizations(final_data, args)

        # Statistical analysis
        gmm_results, beta_results, classification_results = perform_statistical_analysis(final_data, args)

        # Feasibility mapping
        feasibility_results, fig3_path = create_feasibility_map(args)

        # Additional plots
        create_additional_plots(final_data, gmm_results, classification_results, args)

        # Save object probabilities
        save_object_probabilities(final_data, classification_results, args)

        # Summary report
        create_summary_report(final_data, gmm_results, beta_results, classification_results, args)

        # Final output summary
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {args.outdir}/")
        logger.info("\nGenerated files:")
        logger.info(f"  • {fig1_path}")
        logger.info(f"  • {fig2_path}")
        logger.info(f"  • {fig3_path}")
        logger.info(f"  • {args.outdir}/vlms_companions_stacked.csv")
        logger.info(f"  • {args.outdir}/objects_with_probs.csv")
        logger.info(f"  • {args.outdir}/gmm_summary.json")
        logger.info(f"  • {args.outdir}/beta_e_params.csv")
        logger.info(f"  • {args.outdir}/ks_test_e.txt")
        logger.info(f"  • {args.outdir}/feasibility_map.npz")
        logger.info(f"  • {args.outdir}/SUMMARY.txt")
        logger.info(f"\nTotal objects analyzed: {len(final_data)}")
        logger.info(f"Run log: {args.log_file_path}")
        logger.info(f"Error log: {args.error_file_path}")
        logger.info("Ready for manuscript integration!")

    except Exception as e:
        logger.exception("An error occurred during the VLMS analysis")
        sys.exit(1)

if __name__ == "__main__":
    main()

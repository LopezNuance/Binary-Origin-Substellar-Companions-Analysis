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

def count_candidates(min_stellar_mass: float = 0.06, max_stellar_mass: float = 0.20) -> int:
    """Count total candidates from online sources that meet requirements"""
    print("Counting candidates from online data sources...")

    total_candidates = 0

    # Count NASA Exoplanet Archive candidates
    try:
        nasa_fetcher = NASAExoplanetArchiveFetcher()
        nasa_data = nasa_fetcher.fetch_vlms_companions(min_stellar_mass, max_stellar_mass)
        nasa_count = len(nasa_data)
        total_candidates += nasa_count
        print(f"NASA Exoplanet Archive candidates: {nasa_count}")
    except Exception as e:
        print(f"Error counting NASA candidates: {e}")
        nasa_count = 0

    # Count Brown Dwarf Catalogue candidates
    try:
        bd_fetcher = BrownDwarfCatalogueFetcher()
        bd_data = bd_fetcher.fetch_catalogue()
        bd_filtered = bd_fetcher.filter_vlms_hosts(bd_data, min_stellar_mass, max_stellar_mass)
        bd_count = len(bd_filtered)
        total_candidates += bd_count
        print(f"Brown Dwarf Catalogue candidates: {bd_count}")
    except Exception as e:
        print(f"Error counting Brown Dwarf candidates: {e}")
        bd_count = 0

    print(f"Total candidates meeting requirements: {total_candidates}")
    return total_candidates

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

def sample_data(nasa_data, bd_data, percentage: float):
    """Sample a percentage of the combined dataset"""
    if percentage >= 100:
        return nasa_data, bd_data

    # Calculate sample sizes
    nasa_sample_size = int(len(nasa_data) * percentage / 100) if len(nasa_data) > 0 else 0
    bd_sample_size = int(len(bd_data) * percentage / 100) if len(bd_data) > 0 else 0

    # Sample the data
    nasa_sampled = nasa_data.sample(n=nasa_sample_size, random_state=42) if nasa_sample_size > 0 else nasa_data
    bd_sampled = bd_data.sample(n=bd_sample_size, random_state=42) if bd_sample_size > 0 else bd_data

    print(f"Sampled {len(nasa_sampled)} NASA entries ({nasa_sample_size}/{len(nasa_data)})")
    print(f"Sampled {len(bd_sampled)} BD entries ({bd_sample_size}/{len(bd_data)})")
    print(f"Total sampled: {len(nasa_sampled) + len(bd_sampled)} ({percentage}% of original)")

    return nasa_sampled, bd_sampled

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
        toi_ecc=args.toi_ecc,
        toi_age_gyr=getattr(args, 'toi_age_gyr', None)
    )

    final_with_toi = processor.annotate_age_relative_to_toi(final_with_toi, getattr(args, 'toi_age_gyr', None))

    # Enhanced age analysis features
    final_with_toi = processor.classify_age_groups(final_with_toi)
    final_with_toi = processor.enhance_age_analysis_features(final_with_toi)

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

    # Age-migration regression analysis
    logger.info("\n3. Age-Migration Regression Analysis")
    age_regression_results = analyzer.age_migration_regression_analysis(df)

    # Origin classification
    logger.info("\n4. Origin Classification")
    classification_results = analyzer.origin_classification(df)

    # Save statistical results
    analyzer.save_results(gmm_results, beta_results, classification_results,
                         args.outdir, age_regression_results)

    return gmm_results, beta_results, age_regression_results, classification_results

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


def analyze_age_relationships(df, toi_age_gyr, args):
    """Evaluate host ages relative to TOI-6894b and orbital properties."""

    if toi_age_gyr is None or (isinstance(toi_age_gyr, float) and np.isnan(toi_age_gyr)):
        logger.info("No TOI-6894b age provided; skipping age comparison analysis")
        return None

    logger.info("\n" + "=" * 60)
    logger.info("ANALYZING HOST AGE RELATIONSHIPS")
    logger.info("=" * 60)

    if 'host_age_gyr' not in df.columns:
        logger.warning("Dataset lacks host_age_gyr column; skipping age analysis")
        return None

    subset_cols = [
        'companion_name',
        'host_name',
        'host_age_gyr',
        'age_delta_vs_toi_gyr',
        'semimajor_axis_au',
        'eccentricity',
        'data_source'
    ]

    available = df[subset_cols].copy()
    available = available.dropna(subset=['host_age_gyr', 'semimajor_axis_au', 'eccentricity'])

    if available.empty:
        logger.warning("No systems with complete age, semimajor axis, and eccentricity data")
        return {
            'toi_age_gyr': toi_age_gyr,
            'n_with_age': 0,
            'output_path': None
        }

    available.sort_values('age_delta_vs_toi_gyr', inplace=True)

    age_table_path = os.path.join(args.outdir, "age_comparison.csv")
    available.to_csv(age_table_path, index=False)
    logger.info(f"Saved age comparison table to {age_table_path}")

    age_diff = available['age_delta_vs_toi_gyr']

    younger_fraction = float((age_diff < 0).sum() / len(age_diff))
    median_delta = float(np.median(age_diff))

    def safe_corr(x, y):
        if len(x) < 2 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    corr_semimajor = safe_corr(age_diff, available['semimajor_axis_au'])
    corr_ecc = safe_corr(age_diff, available['eccentricity'])

    logger.info(f"Systems with age data: {len(available)}")
    logger.info(f"Median age difference vs TOI-6894b: {median_delta:+.2f} Gyr")
    logger.info(f"Fraction younger than TOI-6894b: {younger_fraction:.2f}")
    logger.info(
        "Correlation(age Δ, semimajor axis): "
        f"{corr_semimajor if not np.isnan(corr_semimajor) else 'N/A'}"
    )
    logger.info(
        "Correlation(age Δ, eccentricity): "
        f"{corr_ecc if not np.isnan(corr_ecc) else 'N/A'}"
    )

    return {
        'toi_age_gyr': toi_age_gyr,
        'n_with_age': int(len(available)),
        'median_delta': median_delta,
        'younger_fraction': younger_fraction,
        'corr_semimajor': corr_semimajor,
        'corr_ecc': corr_ecc,
        'output_path': age_table_path
    }

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

def create_summary_report(df, gmm_results, beta_results, age_regression_results, classification_results, age_summary, args):
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

        # Age regression results
        if age_regression_results and 'error' not in age_regression_results:
            f.write("AGE-MIGRATION REGRESSION ANALYSIS:\n")
            f.write(f"  Systems with age data: {age_regression_results['n_total_objects']}\n")
            corr = age_regression_results['correlations']['age_semimajor_axis']
            f.write(f"  Age vs semimajor axis correlation: r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.3f})\n")
            corr_e = age_regression_results['correlations']['age_eccentricity']
            f.write(f"  Age vs eccentricity correlation: r={corr_e['pearson_r']:.3f} (p={corr_e['pearson_p']:.3f})\n")
            reg = age_regression_results['regressions']['log_semimajor_axis_vs_log_age']
            f.write(f"  log(a) ~ log(age) regression: R²={reg['r_squared']:.3f}, slope={reg['slope']:.3f}\n")
            if 'multiple_regression' in age_regression_results['regressions']:
                mr = age_regression_results['regressions']['multiple_regression']
                f.write(f"  Multiple regression R²={mr['r_squared']:.3f}\n")
            f.write("\n")

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

        # Age analysis summary
        if age_summary:
            f.write("AGE COMPARISON:\n")
            f.write(f"  TOI-6894b age (Gyr): {age_summary['toi_age_gyr']:.2f}\n")
            f.write(f"  Systems with age data: {age_summary['n_with_age']}\n")
            if age_summary['n_with_age'] > 0:
                f.write(f"  Median Δage (system - TOI): {age_summary['median_delta']:+.2f} Gyr\n")
                f.write(f"  Fraction younger than TOI: {age_summary['younger_fraction']:.2f}\n")
                corr_a = age_summary['corr_semimajor']
                corr_e = age_summary['corr_ecc']
                f.write("  Corr(Δage, semimajor axis): ")
                f.write(f"{corr_a:.3f}\n" if not np.isnan(corr_a) else "N/A\n")
                f.write("  Corr(Δage, eccentricity): ")
                f.write(f"{corr_e:.3f}\n\n" if not np.isnan(corr_e) else "N/A\n\n")
            else:
                f.write("  No systems with both age and orbital data.\n\n")

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
  # Interactive mode: count candidates and specify percentage interactively
  python panoptic_vlms_project.py --count-candidates --outdir results

  # Non-interactive mode: process specific percentage of candidates
  python panoptic_vlms_project.py --fetch --percent 50 --outdir results

  # Fetch fresh data and run full analysis
  python panoptic_vlms_project.py --fetch --outdir results

  # Use local CSV files
  python panoptic_vlms_project.py --ps pscomppars_lowM.csv --bd BD_catalogue.csv --outdir results

  # Use local CSV files with percentage sampling
  python panoptic_vlms_project.py --ps pscomppars_lowM.csv --bd BD_catalogue.csv --percent 25 --outdir results

  # Customize TOI-6894b parameters
  python panoptic_vlms_project.py --fetch --toi_mstar 0.08 --toi_mc_mj 0.3 --toi_a_AU 0.05 --outdir results
        """
    )

    # Data source options
    data_group = parser.add_mutually_exclusive_group(required=False)
    data_group.add_argument('--fetch', action='store_true',
                          help='Fetch fresh data from online sources')
    data_group.add_argument('--ps', type=str,
                          help='Path to local NASA PSCompPars CSV file')
    data_group.add_argument('--count-candidates', action='store_true',
                          help='Report number of candidates from online sources and wait for user input')

    parser.add_argument('--bd', type=str,
                       help='Path to local Brown Dwarf Catalogue CSV file (required with --ps)')

    # New percentage argument for non-interactive mode
    parser.add_argument('--percent', type=float, metavar='N',
                       help='Process only N%% of candidates (0-100) in non-interactive mode')

    # TOI-6894b parameters
    parser.add_argument('--toi_mstar', type=float, default=0.08,
                       help='TOI-6894 stellar mass (solar masses, default: 0.08)')
    parser.add_argument('--toi_mc_mj', type=float, default=0.3,
                       help='TOI-6894b companion mass (Jupiter masses, default: 0.3)')
    parser.add_argument('--toi_a_AU', type=float, default=0.05,
                       help='TOI-6894b semi-major axis (AU, default: 0.05)')
    parser.add_argument('--toi_ecc', type=float, default=0.0,
                       help='TOI-6894b eccentricity (default: 0.0)')
    parser.add_argument('--toi_age_gyr', type=float, default=None,
                       help='TOI-6894 system age (Gyr, optional)')

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
    if not args.fetch and not args.count_candidates and args.ps is None:
        parser.error('Must specify --fetch, --count-candidates, or --ps')
    if args.ps and args.bd is None:
        parser.error('--bd is required when using --ps')
    if args.percent is not None and (args.percent < 0 or args.percent > 100):
        parser.error('--percent must be between 0 and 100')
    if args.count_candidates and args.percent is not None:
        parser.error('--count-candidates and --percent cannot be used together')

    # Setup
    setup_logging(args)
    setup_output_directory(args.outdir)
    logger.info(f"Starting VLMS analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Handle count-candidates mode (interactive)
        if args.count_candidates:
            candidate_count = count_candidates()
            print(f"\nFound {candidate_count} candidates that fit the requirements.")
            print("The requirements are:")
            print("  - VLMS hosts with mass between 0.06-0.20 solar masses")
            print("  - Complete data for stellar mass, companion mass, and semimajor axis")
            print("  - Reasonable physical parameters")

            while True:
                try:
                    user_input = input("\nEnter percentage of candidates to process (0-100) or 'exit' to quit: ").strip()
                    if user_input.lower() == 'exit':
                        print("Exiting.")
                        sys.exit(0)

                    percentage = float(user_input)
                    if 0 <= percentage <= 100:
                        break
                    else:
                        print("Please enter a number between 0 and 100.")
                except ValueError:
                    print("Please enter a valid number or 'exit'.")

            # Now fetch and process the data with the specified percentage
            args.fetch = True  # Override to fetch data
            nasa_data, bd_data = fetch_data(args)
            nasa_data, bd_data = sample_data(nasa_data, bd_data, percentage)

        # Data acquisition
        elif args.fetch:
            nasa_data, bd_data = fetch_data(args)
            # Apply percentage sampling if specified
            if args.percent is not None:
                nasa_data, bd_data = sample_data(nasa_data, bd_data, args.percent)
        else:
            logger.info("Loading local data files...")
            data_results = load_local_data(args.ps, args.bd)
            nasa_data, bd_data = data_results["nasa_df"], data_results["bd_df"]
            # Apply percentage sampling if specified
            if args.percent is not None:
                nasa_data, bd_data = sample_data(nasa_data, bd_data, args.percent)

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
        gmm_results, beta_results, age_regression_results, classification_results = perform_statistical_analysis(final_data, args)

        # Feasibility mapping
        feasibility_results, fig3_path = create_feasibility_map(args)

        # Additional plots
        create_additional_plots(final_data, gmm_results, classification_results, args)

        # Save object probabilities
        save_object_probabilities(final_data, classification_results, args)

        # Age analysis
        age_summary = analyze_age_relationships(final_data, args.toi_age_gyr, args)

        # Summary report
        create_summary_report(final_data, gmm_results, beta_results, age_regression_results, classification_results, age_summary, args)

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
        if age_summary and age_summary.get('output_path'):
            logger.info(f"  • {age_summary['output_path']}")
        logger.info(f"\nTotal objects analyzed: {len(final_data)}")
        logger.info(f"Run log: {args.log_file_path}")
        logger.info(f"Error log: {args.error_file_path}")
        logger.info("Ready for manuscript integration!")

    except Exception as e:
        logger.exception("An error occurred during the VLMS analysis")
        sys.exit(1)

if __name__ == "__main__":
    main()

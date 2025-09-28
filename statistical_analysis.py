import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats
from scipy.optimize import minimize
import warnings
import json

# Try to import numba, fall back to pure Python if not available
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    print("Warning: numba not available, using pure Python (slower)")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Handle @jit without parentheses
            return args[0]
        return decorator
    prange = range
    NUMBA_AVAILABLE = False

class StatisticalAnalyzer:
    """Perform statistical analyses for VLMS companion study"""

    def __init__(self):
        self.scaler = StandardScaler()

    def gaussian_mixture_analysis(self, df: pd.DataFrame, max_components: int = 5) -> dict:
        """
        Perform Gaussian Mixture Model analysis in (log q, log a) space

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with log_mass_ratio and log_semimajor_axis
        max_components : int
            Maximum number of components to test

        Returns:
        --------
        dict with GMM results including BIC scores and best model
        """

        print("Performing Gaussian Mixture Model analysis...")

        # Prepare data
        X = df[['log_mass_ratio', 'log_semimajor_axis']].dropna()

        if len(X) < 10:
            print("Warning: Too few data points for reliable GMM analysis")
            return {'bic_scores': [], 'best_n_components': 1}

        # Test different numbers of components
        n_components_range = range(1, min(max_components + 1, len(X) // 2))
        bic_scores = []
        aic_scores = []
        models = []

        for n in n_components_range:
            try:
                gmm = GaussianMixture(n_components=n, covariance_type='full',
                                    random_state=42, max_iter=200)
                gmm.fit(X)
                bic_scores.append(gmm.bic(X))
                aic_scores.append(gmm.aic(X))
                models.append(gmm)
            except Exception as e:
                print(f"Warning: GMM with {n} components failed: {e}")
                bic_scores.append(np.inf)
                aic_scores.append(np.inf)
                models.append(None)

        # Find best model
        best_n = np.argmin(bic_scores) + 1
        best_model = models[np.argmin(bic_scores)]

        results = {
            'bic_scores': bic_scores,
            'aic_scores': aic_scores,
            'best_n_components': best_n,
            'best_model': best_model,
            'n_components_range': list(n_components_range)
        }

        # Add cluster assignments if best model has > 1 component
        if best_model is not None and best_n > 1:
            try:
                cluster_labels = best_model.predict(X)
                results['cluster_labels'] = cluster_labels
                results['cluster_probs'] = best_model.predict_proba(X)

                print(f"Best model: {best_n} components (BIC = {min(bic_scores):.1f})")
                for i in range(best_n):
                    n_in_cluster = np.sum(cluster_labels == i)
                    print(f"  Component {i}: {n_in_cluster} objects ({n_in_cluster/len(X)*100:.1f}%)")

            except Exception as e:
                print(f"Warning: Could not generate cluster assignments: {e}")

        else:
            print("Best model: 1 component (single population)")

        return results

    def beta_distribution_analysis(self, df: pd.DataFrame) -> dict:
        """
        Fit Beta distributions to eccentricity data for different mass ratio groups

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with eccentricity and high_mass_ratio columns

        Returns:
        --------
        dict with Beta distribution parameters and KS test results
        """

        print("Performing Beta distribution analysis of eccentricities...")

        results = {}

        # Split by mass ratio
        high_q = df[df['high_mass_ratio']]['eccentricity'].dropna()
        low_q = df[~df['high_mass_ratio']]['eccentricity'].dropna()

        if len(high_q) < 3 or len(low_q) < 3:
            print("Warning: Too few data points for reliable Beta distribution fitting")
            return {'error': 'Insufficient data'}

        # Fit Beta distributions
        try:
            # Clip eccentricities to (0, 1) range for Beta distribution
            high_q_clipped = np.clip(high_q, 1e-6, 1 - 1e-6)
            low_q_clipped = np.clip(low_q, 1e-6, 1 - 1e-6)

            # Maximum likelihood estimation
            high_q_params = stats.beta.fit(high_q_clipped, floc=0, fscale=1)
            low_q_params = stats.beta.fit(low_q_clipped, floc=0, fscale=1)

            results['high_q'] = {
                'alpha': high_q_params[0],
                'beta': high_q_params[1],
                'n_objects': len(high_q),
                'mean_e': np.mean(high_q),
                'median_e': np.median(high_q)
            }

            results['low_q'] = {
                'alpha': low_q_params[0],
                'beta': low_q_params[1],
                'n_objects': len(low_q),
                'mean_e': np.mean(low_q),
                'median_e': np.median(low_q)
            }

            # Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = stats.ks_2samp(high_q, low_q)

            results['ks_test'] = {
                'statistic': ks_statistic,
                'p_value': ks_p_value,
                'significant': ks_p_value < 0.05
            }

            # Mann-Whitney U test (non-parametric alternative)
            u_statistic, u_p_value = stats.mannwhitneyu(high_q, low_q, alternative='two-sided')

            results['mannwhitney_test'] = {
                'statistic': u_statistic,
                'p_value': u_p_value,
                'significant': u_p_value < 0.05
            }

            print(f"Beta distribution fits:")
            print(f"  High-q: α={high_q_params[0]:.3f}, β={high_q_params[1]:.3f} (n={len(high_q)})")
            print(f"  Low-q:  α={low_q_params[0]:.3f}, β={low_q_params[1]:.3f} (n={len(low_q)})")
            print(f"KS test: D={ks_statistic:.3f}, p={ks_p_value:.3f}")

        except Exception as e:
            print(f"Error in Beta distribution analysis: {e}")
            results['error'] = str(e)

        return results

    def origin_classification(self, df: pd.DataFrame) -> dict:
        """
        Train logistic regression classifier for binary-like origin prediction

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with features for classification

        Returns:
        --------
        dict with classifier results including probabilities
        """

        print("Training origin classification model...")

        # Define features
        feature_cols = ['log_mass_ratio', 'log_semimajor_axis', 'eccentricity', 'log_host_mass']

        # Add metallicity if available
        if 'metallicity' in df.columns:
            feature_cols.append('metallicity')

        # One-hot encode discovery method if available
        if 'discovery_method' in df.columns:
            method_dummies = pd.get_dummies(df['discovery_method'], prefix='method')
            feature_df = pd.concat([df[feature_cols], method_dummies], axis=1)
            feature_cols.extend(method_dummies.columns)
        else:
            feature_df = df[feature_cols].copy()

        # Remove rows with missing features
        feature_df = feature_df.dropna()

        if len(feature_df) < 10:
            print("Warning: Too few complete observations for classification")
            return {'error': 'Insufficient data for classification'}

        # Create pseudo-labels based on mass ratio and eccentricity
        # High mass ratio + high eccentricity = more likely binary-like
        q_threshold = np.percentile(df['mass_ratio'].dropna(), 75)  # Top quartile
        e_threshold = np.percentile(df['eccentricity'].dropna(), 50)  # Median

        # Create binary labels (this is a heuristic for demonstration)
        binary_like_labels = ((df.loc[feature_df.index, 'mass_ratio'] > q_threshold) |
                             (df.loc[feature_df.index, 'eccentricity'] > e_threshold)).astype(int)

        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(feature_df)

            # Train logistic regression with regularization
            classifier = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
            classifier.fit(X_scaled, binary_like_labels)

            # Predict probabilities
            probabilities = classifier.predict_proba(X_scaled)[:, 1]

            # Cross-validation score
            cv_scores = cross_val_score(classifier, X_scaled, binary_like_labels,
                                       cv=min(5, len(feature_df) // 2), scoring='roc_auc')

            # Feature importance (coefficients)
            feature_importance = dict(zip(feature_cols, classifier.coef_[0]))

            results = {
                'model': classifier,
                'scaler': self.scaler,
                'probabilities': probabilities,
                'feature_columns': feature_cols,
                'feature_importance': feature_importance,
                'cv_auc_mean': np.mean(cv_scores),
                'cv_auc_std': np.std(cv_scores),
                'n_objects': len(feature_df),
                'data_indices': feature_df.index.tolist()
            }

            print(f"Classification model trained on {len(feature_df)} objects")
            print(f"Cross-validated AUC: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            print("Top feature importances:")
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            for feat, coef in sorted_features[:5]:
                print(f"  {feat}: {coef:.3f}")

        except Exception as e:
            print(f"Error in classification: {e}")
            results = {'error': str(e)}

        return results

    def save_results(self, gmm_results: dict, beta_results: dict,
                    classification_results: dict, output_dir: str = "."):
        """Save statistical analysis results to files"""

        # Save GMM results
        gmm_summary = {
            'bic_scores': gmm_results.get('bic_scores', []),
            'aic_scores': gmm_results.get('aic_scores', []),
            'best_n_components': gmm_results.get('best_n_components', 1),
            'n_components_range': gmm_results.get('n_components_range', [])
        }

        with open(f"{output_dir}/gmm_summary.json", 'w') as f:
            json.dump(gmm_summary, f, indent=2)
        print(f"Saved GMM summary to {output_dir}/gmm_summary.json")

        # Save Beta distribution parameters
        if 'error' not in beta_results:
            beta_df = pd.DataFrame({
                'group': ['high_q', 'low_q'],
                'alpha': [beta_results['high_q']['alpha'], beta_results['low_q']['alpha']],
                'beta': [beta_results['high_q']['beta'], beta_results['low_q']['beta']],
                'n_objects': [beta_results['high_q']['n_objects'], beta_results['low_q']['n_objects']],
                'mean_e': [beta_results['high_q']['mean_e'], beta_results['low_q']['mean_e']],
                'median_e': [beta_results['high_q']['median_e'], beta_results['low_q']['median_e']]
            })
            beta_df.to_csv(f"{output_dir}/beta_e_params.csv", index=False)
            print(f"Saved Beta parameters to {output_dir}/beta_e_params.csv")

            # Save KS test results
            ks_text = f"""Kolmogorov-Smirnov Test for Eccentricity Distributions
High-q vs Low-q mass ratio groups:
  KS statistic: {beta_results['ks_test']['statistic']:.4f}
  p-value: {beta_results['ks_test']['p_value']:.4f}
  Significant (p < 0.05): {beta_results['ks_test']['significant']}

Mann-Whitney U Test:
  U statistic: {beta_results['mannwhitney_test']['statistic']:.4f}
  p-value: {beta_results['mannwhitney_test']['p_value']:.4f}
  Significant (p < 0.05): {beta_results['mannwhitney_test']['significant']}
"""
            with open(f"{output_dir}/ks_test_e.txt", 'w') as f:
                f.write(ks_text)
            print(f"Saved statistical tests to {output_dir}/ks_test_e.txt")


@jit(nopython=True)
def kozai_lidov_feasibility_single(M_star, M_comp, a_inner, e_inner,
                                  M_perturber, a_outer, e_outer,
                                  target_a=0.05, t_max_gyr=1.0):
    """
    Single realization of Kozai-Lidov + tides evolution (simplified)

    This is a very simplified model - real implementation would need
    proper orbital mechanics integration
    """

    # Kozai-Lidov timescale (simplified)
    # Real formula is much more complex
    if a_outer <= a_inner * 3:  # Hill sphere check
        return False

    # Approximate KL timescale
    P_inner = np.sqrt(a_inner**3 / M_star)  # Years
    P_outer = np.sqrt(a_outer**3 / (M_star + M_perturber))

    if P_outer <= P_inner * 3:  # Stability check
        return False

    t_kl_yr = P_outer * (M_star / M_perturber) * (a_outer / a_inner)**3

    # Can it complete cycles within t_max?
    if t_kl_yr > t_max_gyr * 1e9:
        return False

    # Maximum eccentricity from KL (simplified)
    e_max = np.sqrt(1.0 - 5.0/3.0 * (1.0 - e_inner**2))
    e_max = min(e_max, 0.95)  # Physical limit

    # Periapsis at maximum eccentricity
    periapsis = a_inner * (1.0 - e_max)

    # If periapsis gets small enough, tides can circularize
    # This is very approximate
    if periapsis < 0.02:  # AU, roughly where tides become strong
        return periapsis <= target_a * 1.5

    return False


class KozaiLidovAnalyzer:
    """Perform Kozai-Lidov + tides feasibility analysis"""

    def __init__(self, n_trials: int = 1000):
        self.n_trials = n_trials

    def create_feasibility_map(self, perturber_mass_range: tuple = (0.1, 1.0),
                              perturber_sep_range: tuple = (10, 1000),
                              n_mass_points: int = 20,
                              n_sep_points: int = 20) -> dict:
        """
        Create feasibility map for Kozai-Lidov + tides migration

        Parameters:
        -----------
        perturber_mass_range : tuple
            Range of perturber masses (solar masses)
        perturber_sep_range : tuple
            Range of perturber separations (AU)
        n_mass_points : int
            Number of mass grid points
        n_sep_points : int
            Number of separation grid points

        Returns:
        --------
        dict with feasibility map and parameters
        """

        print(f"Creating Kozai-Lidov feasibility map ({n_mass_points}×{n_sep_points} grid, {self.n_trials} trials each)...")

        # Create grids
        mass_grid = np.logspace(np.log10(perturber_mass_range[0]),
                               np.log10(perturber_mass_range[1]), n_mass_points)
        sep_grid = np.logspace(np.log10(perturber_sep_range[0]),
                              np.log10(perturber_sep_range[1]), n_sep_points)

        # Initialize results
        feasibility_map = np.zeros((n_mass_points, n_sep_points))

        # TOI-6894b-like system parameters
        M_star = 0.08  # Solar masses
        M_comp = 0.3 / 1047.6  # Jupiter to solar masses
        a_initial = 1.0  # Start at ~1 AU

        # Grid search
        for i, M_pert in enumerate(mass_grid):
            for j, a_pert in enumerate(sep_grid):
                successes = 0

                # Monte Carlo trials
                for trial in range(self.n_trials):
                    # Random initial conditions
                    e_inner = np.random.uniform(0.1, 0.7)  # Initial eccentricity
                    e_outer = np.random.uniform(0.0, 0.3)  # Outer orbit eccentricity

                    # Test if this configuration can migrate to target
                    if kozai_lidov_feasibility_single(M_star, M_comp, a_initial, e_inner,
                                                    M_pert, a_pert, e_outer):
                        successes += 1

                feasibility_map[i, j] = successes / self.n_trials

        results = {
            'feasibility_map': feasibility_map,
            'perturber_masses': mass_grid,
            'perturber_separations': sep_grid,
            'n_trials': self.n_trials
        }

        max_feasibility = np.max(feasibility_map)
        print(f"Feasibility map completed. Maximum success rate: {max_feasibility:.3f}")

        return results

    def save_feasibility_map(self, results: dict, filename: str):
        """Save feasibility map to compressed numpy file"""
        np.savez_compressed(filename, **results)
        print(f"Saved feasibility map to {filename}")


if __name__ == "__main__":
    # Test the statistical analyzer
    analyzer = StatisticalAnalyzer()

    # Create dummy data
    np.random.seed(42)
    n_objects = 100

    dummy_data = pd.DataFrame({
        'log_mass_ratio': np.random.normal(-2, 0.5, n_objects),
        'log_semimajor_axis': np.random.normal(-1, 0.5, n_objects),
        'eccentricity': np.random.beta(0.867, 3.03, n_objects),
        'log_host_mass': np.random.normal(np.log10(0.1), 0.3, n_objects),
        'mass_ratio': 10**np.random.normal(-2, 0.5, n_objects),
        'high_mass_ratio': np.random.choice([True, False], n_objects, p=[0.3, 0.7]),
        'discovery_method': np.random.choice(['Transit', 'RV', 'Imaging'], n_objects)
    })

    # Test analyses
    print("Testing statistical analyses...")

    # GMM analysis
    gmm_results = analyzer.gaussian_mixture_analysis(dummy_data)

    # Beta distribution analysis
    beta_results = analyzer.beta_distribution_analysis(dummy_data)

    # Classification
    classification_results = analyzer.origin_classification(dummy_data)

    # Kozai-Lidov analysis
    kl_analyzer = KozaiLidovAnalyzer(n_trials=100)  # Reduced for testing
    feasibility_results = kl_analyzer.create_feasibility_map(n_mass_points=10, n_sep_points=10)

    # Save results
    analyzer.save_results(gmm_results, beta_results, classification_results, "test_output")
    kl_analyzer.save_feasibility_map(feasibility_results, "test_feasibility_map.npz")

    print("Statistical analysis tests completed!")
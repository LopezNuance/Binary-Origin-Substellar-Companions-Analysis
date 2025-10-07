import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, Any, Optional
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

    def beta_distribution_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
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

        results: Dict[str, Any] = {}

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

        # Always compute bootstrap bagging summary alongside the parametric fit
        try:
            bagging_results = self.bagged_beta_distribution_analysis(df)
        except Exception as e:  # Defensive: bagging should not block pipeline
            warnings.warn(f"Bagged beta analysis failed: {e}")
            bagging_results = {'error': str(e)}

        results['bagging'] = bagging_results

        return results

    def bagged_beta_distribution_analysis(
        self,
        df: pd.DataFrame,
        n_bootstrap: int = 500,
        sample_fraction: float = 0.8,
        min_group_size: int = 3,
        random_state: Optional[int] = 42
    ) -> Dict[str, Any]:
        """
        Apply bootstrap bagging to the eccentricity-based statistical tests.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with eccentricity and high_mass_ratio columns
        n_bootstrap : int
            Number of bootstrap resamples to draw
        sample_fraction : float
            Fraction of each group to sample (with replacement) per bootstrap
        min_group_size : int
            Minimum number of samples per group in each bootstrap
        random_state : Optional[int]
            Seed for the random number generator

        Returns:
        --------
        dict with aggregated bagging statistics for the Beta-based tests
        """

        bagger = StatisticalBagging(
            n_bootstrap=n_bootstrap,
            sample_fraction=sample_fraction,
            min_group_size=min_group_size,
            random_state=random_state
        )

        return bagger.bootstrap_beta_distribution(df)

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

    def age_migration_regression_analysis(self, df: pd.DataFrame) -> dict:
        """
        Perform regression analysis of age vs orbital parameters

        This provides an introductory statistical analysis before the more
        sophisticated physics-based migration modeling.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with age, semimajor axis, and eccentricity data

        Returns:
        --------
        dict with regression results and correlations
        """

        print("Performing age-migration regression analysis...")

        # Filter to systems with complete age and orbital data
        complete_data = df.dropna(subset=['host_age_gyr', 'semimajor_axis_au', 'eccentricity']).copy()

        if len(complete_data) < 10:
            print("Warning: Too few systems with complete age/orbital data for regression")
            return {'error': 'Insufficient data for age regression analysis'}

        results = {}

        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            from scipy.stats import pearsonr, spearmanr
            import statsmodels.api as sm

            # Log-transform variables for better linear relationships
            complete_data['log_age'] = np.log10(complete_data['host_age_gyr'])
            complete_data['log_semimajor_axis'] = np.log10(complete_data['semimajor_axis_au'])

            # Simple correlations
            correlations = {}

            # Age vs semimajor axis
            age_a_pearson, age_a_p_pearson = pearsonr(complete_data['host_age_gyr'],
                                                     complete_data['semimajor_axis_au'])
            age_a_spearman, age_a_p_spearman = spearmanr(complete_data['host_age_gyr'],
                                                        complete_data['semimajor_axis_au'])

            # Age vs eccentricity
            age_e_pearson, age_e_p_pearson = pearsonr(complete_data['host_age_gyr'],
                                                     complete_data['eccentricity'])
            age_e_spearman, age_e_p_spearman = spearmanr(complete_data['host_age_gyr'],
                                                        complete_data['eccentricity'])

            # Log age vs log semimajor axis
            log_age_log_a_pearson, log_age_log_a_p = pearsonr(complete_data['log_age'],
                                                             complete_data['log_semimajor_axis'])

            correlations = {
                'age_semimajor_axis': {
                    'pearson_r': age_a_pearson,
                    'pearson_p': age_a_p_pearson,
                    'spearman_r': age_a_spearman,
                    'spearman_p': age_a_p_spearman
                },
                'age_eccentricity': {
                    'pearson_r': age_e_pearson,
                    'pearson_p': age_e_p_pearson,
                    'spearman_r': age_e_spearman,
                    'spearman_p': age_e_p_spearman
                },
                'log_age_log_semimajor_axis': {
                    'pearson_r': log_age_log_a_pearson,
                    'pearson_p': log_age_log_a_p
                }
            }

            # Linear regression models
            regressions = {}

            # Model 1: log(a) ~ log(age)
            X_log = complete_data[['log_age']].values
            y_log_a = complete_data['log_semimajor_axis'].values

            reg_log_a = LinearRegression().fit(X_log, y_log_a)
            y_pred_log_a = reg_log_a.predict(X_log)
            r2_log_a = r2_score(y_log_a, y_pred_log_a)

            regressions['log_semimajor_axis_vs_log_age'] = {
                'slope': reg_log_a.coef_[0],
                'intercept': reg_log_a.intercept_,
                'r_squared': r2_log_a,
                'n_objects': len(complete_data)
            }

            # Model 2: e ~ log(age)
            y_ecc = complete_data['eccentricity'].values
            reg_ecc = LinearRegression().fit(X_log, y_ecc)
            y_pred_ecc = reg_ecc.predict(X_log)
            r2_ecc = r2_score(y_ecc, y_pred_ecc)

            regressions['eccentricity_vs_log_age'] = {
                'slope': reg_ecc.coef_[0],
                'intercept': reg_ecc.intercept_,
                'r_squared': r2_ecc,
                'n_objects': len(complete_data)
            }

            # Multiple regression: log(a) ~ log(age) + e + log(M_star)
            if 'log_host_mass' in complete_data.columns:
                X_multi = complete_data[['log_age', 'eccentricity', 'log_host_mass']].dropna()
                if len(X_multi) >= 5:
                    y_multi = complete_data.loc[X_multi.index, 'log_semimajor_axis']

                    # Add constant for statsmodels
                    X_multi_sm = sm.add_constant(X_multi)

                    # Fit with statsmodels for more detailed statistics
                    model_multi = sm.OLS(y_multi, X_multi_sm).fit()

                    regressions['multiple_regression'] = {
                        'coefficients': {
                            'intercept': model_multi.params[0],
                            'log_age': model_multi.params[1],
                            'eccentricity': model_multi.params[2],
                            'log_host_mass': model_multi.params[3]
                        },
                        'r_squared': model_multi.rsquared,
                        'adjusted_r_squared': model_multi.rsquared_adj,
                        'f_statistic': model_multi.fvalue,
                        'f_p_value': model_multi.f_pvalue,
                        'p_values': {
                            'intercept': model_multi.pvalues[0],
                            'log_age': model_multi.pvalues[1],
                            'eccentricity': model_multi.pvalues[2],
                            'log_host_mass': model_multi.pvalues[3]
                        },
                        'n_objects': len(X_multi)
                    }

            results = {
                'correlations': correlations,
                'regressions': regressions,
                'n_total_objects': len(complete_data),
                'age_range_gyr': (float(complete_data['host_age_gyr'].min()),
                                 float(complete_data['host_age_gyr'].max())),
                'semimajor_axis_range_au': (float(complete_data['semimajor_axis_au'].min()),
                                          float(complete_data['semimajor_axis_au'].max())),
                'data_indices': complete_data.index.tolist()
            }

            # Print summary
            print(f"Age regression analysis on {len(complete_data)} systems:")
            print(f"  Age vs semimajor axis correlation: r = {age_a_pearson:.3f} (p = {age_a_p_pearson:.3f})")
            print(f"  Age vs eccentricity correlation: r = {age_e_pearson:.3f} (p = {age_e_p_pearson:.3f})")
            print(f"  log(a) ~ log(age) regression: R² = {r2_log_a:.3f}, slope = {reg_log_a.coef_[0]:.3f}")

            if 'multiple_regression' in regressions:
                mr = regressions['multiple_regression']
                print(f"  Multiple regression R² = {mr['r_squared']:.3f}")
                print(f"    log(age) coefficient: {mr['coefficients']['log_age']:.3f} (p = {mr['p_values']['log_age']:.3f})")

        except ImportError as e:
            print(f"Warning: Missing dependencies for regression analysis: {e}")
            results = {'error': f'Missing dependencies: {e}'}
        except Exception as e:
            print(f"Error in age regression analysis: {e}")
            results = {'error': str(e)}

        return results

    def save_results(self, gmm_results: dict, beta_results: dict,
                    classification_results: dict, output_dir: str = ".",
                    age_regression_results: dict = None,
                    bagged_beta_results: Optional[Dict[str, Any]] = None):
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

        # Save bagged beta bootstrap summary and raw draws
        def _to_builtin(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_builtin(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_builtin(v) for v in obj]
            return obj

        effective_bagging = bagged_beta_results or beta_results.get('bagging')
        if effective_bagging:
            if 'error' in effective_bagging:
                print(f"Bagged beta analysis reported error: {effective_bagging['error']}")
            else:
                summary_only = {k: v for k, v in effective_bagging.items()
                                if k != 'bootstrap_distributions'}
                summary_path = f"{output_dir}/beta_e_bootstrap_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(_to_builtin(summary_only), f, indent=2)
                print(f"Saved bagged Beta summary to {summary_path}")

                distributions = effective_bagging.get('bootstrap_distributions')
                if distributions:
                    dist_df = pd.DataFrame(_to_builtin(distributions))
                    dist_path = f"{output_dir}/beta_e_bootstrap_distributions.csv"
                    dist_df.to_csv(dist_path, index=False)
                    print(f"Saved bagged Beta samples to {dist_path}")

        # Save age regression results
        if age_regression_results and 'error' not in age_regression_results:
            # Save regression summary as JSON
            with open(f"{output_dir}/age_regression_summary.json", 'w') as f:
                json.dump(age_regression_results, f, indent=2)
            print(f"Saved age regression results to {output_dir}/age_regression_summary.json")

            # Create detailed regression report
            regression_text = f"""Age-Migration Regression Analysis Report

CORRELATIONS:
Age vs Semimajor Axis:
  Pearson r = {age_regression_results['correlations']['age_semimajor_axis']['pearson_r']:.4f}
  Pearson p-value = {age_regression_results['correlations']['age_semimajor_axis']['pearson_p']:.4f}
  Spearman r = {age_regression_results['correlations']['age_semimajor_axis']['spearman_r']:.4f}
  Spearman p-value = {age_regression_results['correlations']['age_semimajor_axis']['spearman_p']:.4f}

Age vs Eccentricity:
  Pearson r = {age_regression_results['correlations']['age_eccentricity']['pearson_r']:.4f}
  Pearson p-value = {age_regression_results['correlations']['age_eccentricity']['pearson_p']:.4f}
  Spearman r = {age_regression_results['correlations']['age_eccentricity']['spearman_r']:.4f}
  Spearman p-value = {age_regression_results['correlations']['age_eccentricity']['spearman_p']:.4f}

Log(Age) vs Log(Semimajor Axis):
  Pearson r = {age_regression_results['correlations']['log_age_log_semimajor_axis']['pearson_r']:.4f}
  Pearson p-value = {age_regression_results['correlations']['log_age_log_semimajor_axis']['pearson_p']:.4f}

REGRESSION MODELS:
Log(Semimajor Axis) ~ Log(Age):
  Slope = {age_regression_results['regressions']['log_semimajor_axis_vs_log_age']['slope']:.4f}
  Intercept = {age_regression_results['regressions']['log_semimajor_axis_vs_log_age']['intercept']:.4f}
  R² = {age_regression_results['regressions']['log_semimajor_axis_vs_log_age']['r_squared']:.4f}

Eccentricity ~ Log(Age):
  Slope = {age_regression_results['regressions']['eccentricity_vs_log_age']['slope']:.4f}
  Intercept = {age_regression_results['regressions']['eccentricity_vs_log_age']['intercept']:.4f}
  R² = {age_regression_results['regressions']['eccentricity_vs_log_age']['r_squared']:.4f}
"""

            # Add multiple regression if available
            if 'multiple_regression' in age_regression_results['regressions']:
                mr = age_regression_results['regressions']['multiple_regression']
                regression_text += f"""
Multiple Regression: Log(Semimajor Axis) ~ Log(Age) + Eccentricity + Log(Host Mass):
  R² = {mr['r_squared']:.4f}
  Adjusted R² = {mr['adjusted_r_squared']:.4f}
  F-statistic = {mr['f_statistic']:.4f} (p = {mr['f_p_value']:.4f})

  Coefficients:
    Log(Age): {mr['coefficients']['log_age']:.4f} (p = {mr['p_values']['log_age']:.4f})
    Eccentricity: {mr['coefficients']['eccentricity']:.4f} (p = {mr['p_values']['eccentricity']:.4f})
    Log(Host Mass): {mr['coefficients']['log_host_mass']:.4f} (p = {mr['p_values']['log_host_mass']:.4f})
    Intercept: {mr['coefficients']['intercept']:.4f} (p = {mr['p_values']['intercept']:.4f})
"""

            regression_text += f"""
DATASET SUMMARY:
  Total objects with complete age/orbital data: {age_regression_results['n_total_objects']}
  Age range: {age_regression_results['age_range_gyr'][0]:.2f} - {age_regression_results['age_range_gyr'][1]:.2f} Gyr
  Semimajor axis range: {age_regression_results['semimajor_axis_range_au'][0]:.4f} - {age_regression_results['semimajor_axis_range_au'][1]:.4f} AU

INTERPRETATION:
This regression analysis provides a preliminary statistical examination of age-orbital parameter
relationships before the detailed physics-based migration modeling. Significant correlations
may indicate evolutionary processes affecting companion orbits over stellar lifetimes.
"""

            with open(f"{output_dir}/age_regression_report.txt", 'w') as f:
                f.write(regression_text)
            print(f"Saved detailed age regression report to {output_dir}/age_regression_report.txt")


class StatisticalBagging:
    """Bootstrap bagging utilities for the VLMS statistical analyses."""

    def __init__(
        self,
        n_bootstrap: int = 500,
        sample_fraction: float = 0.8,
        min_group_size: int = 3,
        random_state: Optional[int] = None
    ) -> None:
        if n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive")
        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be in (0, 1]")
        if min_group_size < 1:
            raise ValueError("min_group_size must be at least 1")

        self.n_bootstrap = int(n_bootstrap)
        self.sample_fraction = float(sample_fraction)
        self.min_group_size = int(min_group_size)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def _sample_group(self, df: pd.DataFrame) -> pd.DataFrame:
        size = max(self.min_group_size, int(np.ceil(len(df) * self.sample_fraction)))
        indices = self._rng.choice(len(df), size=size, replace=True)
        return df.iloc[indices].copy()

    @staticmethod
    def _summarize(values: list[float]) -> Dict[str, float]:
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return {
                'mean': float('nan'),
                'std': float('nan'),
                'median': float('nan'),
                'ci95_low': float('nan'),
                'ci95_high': float('nan')
            }

        return {
            'mean': float(array.mean()),
            'std': float(array.std(ddof=1)) if array.size > 1 else 0.0,
            'median': float(np.median(array)),
            'ci95_low': float(np.quantile(array, 0.025)),
            'ci95_high': float(np.quantile(array, 0.975))
        }

    def bootstrap_beta_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run bootstrap bagging on the eccentricity-based statistical tests.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with `eccentricity` and `high_mass_ratio` columns

        Returns:
        --------
        dict summarizing bootstrap distributions of test statistics
        """

        required_columns = {'eccentricity', 'high_mass_ratio'}
        missing = required_columns - set(df.columns)
        if missing:
            return {'error': f'Missing required columns for bagging: {sorted(missing)}'}

        high_group = df[df['high_mass_ratio']].dropna(subset=['eccentricity'])
        low_group = df[~df['high_mass_ratio']].dropna(subset=['eccentricity'])

        if len(high_group) < self.min_group_size or len(low_group) < self.min_group_size:
            return {'error': 'Insufficient data for bootstrapped beta analysis'}

        ks_stats: list[float] = []
        ks_p_values: list[float] = []
        u_stats: list[float] = []
        u_p_values: list[float] = []
        alpha_high: list[float] = []
        beta_high: list[float] = []
        alpha_low: list[float] = []
        beta_low: list[float] = []

        successes = 0

        for _ in range(self.n_bootstrap):
            high_sample = self._sample_group(high_group)
            low_sample = self._sample_group(low_group)

            high_vals = np.clip(high_sample['eccentricity'].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
            low_vals = np.clip(low_sample['eccentricity'].to_numpy(dtype=float), 1e-6, 1 - 1e-6)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    high_params = stats.beta.fit(high_vals, floc=0, fscale=1)
                    low_params = stats.beta.fit(low_vals, floc=0, fscale=1)

                ks_statistic, ks_p = stats.ks_2samp(high_vals, low_vals)
                u_statistic, u_p = stats.mannwhitneyu(high_vals, low_vals, alternative='two-sided')

            except Exception:
                continue

            successes += 1

            alpha_high.append(float(high_params[0]))
            beta_high.append(float(high_params[1]))
            alpha_low.append(float(low_params[0]))
            beta_low.append(float(low_params[1]))

            ks_stats.append(float(ks_statistic))
            ks_p_values.append(float(ks_p))
            u_stats.append(float(u_statistic))
            u_p_values.append(float(u_p))

        if successes == 0:
            return {'error': 'Bootstrap bagging failed for all resamples'}

        ks_p_array = np.asarray(ks_p_values)
        u_p_array = np.asarray(u_p_values)

        summary: Dict[str, Any] = {
            'n_bootstrap': self.n_bootstrap,
            'n_successful': successes,
            'n_failed': self.n_bootstrap - successes,
            'sample_fraction': self.sample_fraction,
            'min_group_size': self.min_group_size,
            'ks_test': {
                'statistic': self._summarize(ks_stats),
                'p_value': self._summarize(ks_p_values),
                'significant_rate': float(np.mean(ks_p_array < 0.05))
            },
            'mannwhitney_test': {
                'statistic': self._summarize(u_stats),
                'p_value': self._summarize(u_p_values),
                'significant_rate': float(np.mean(u_p_array < 0.05))
            },
            'beta_parameters': {
                'high_q': {
                    'alpha': self._summarize(alpha_high),
                    'beta': self._summarize(beta_high)
                },
                'low_q': {
                    'alpha': self._summarize(alpha_low),
                    'beta': self._summarize(beta_low)
                }
            },
            'bootstrap_distributions': {
                'ks_p_values': ks_p_values,
                'ks_statistics': ks_stats,
                'mannwhitney_p_values': u_p_values,
                'mannwhitney_statistics': u_stats,
                'alpha_high': alpha_high,
                'beta_high': beta_high,
                'alpha_low': alpha_low,
                'beta_low': beta_low
            }
        }

        print(
            "Bootstrapped Beta analysis: "
            f"{successes}/{self.n_bootstrap} successful resamples; "
            f"mean KS p-value = {summary['ks_test']['p_value']['mean']:.3f}, "
            f"significant fraction = {summary['ks_test']['significant_rate']:.2f}"
        )

        return summary


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
    kl_argument = 1.0 - 5.0/3.0 * (1.0 - e_inner**2)
    if kl_argument <= 0:
        # KL feasibility rejected: non-positive argument
        return False

    e_max = np.sqrt(kl_argument)
    e_max = min(e_max, 0.95)  # Physical limit

    # Periapsis at maximum eccentricity
    periapsis = a_inner * (1.0 - e_max)

    # If periapsis gets small enough, tides can circularize
    # This is very approximate
    if periapsis < 0.02:  # AU, roughly where tides become strong
        return periapsis <= target_a * 1.5

    return False


@jit(nopython=True)
def age_dependent_stellar_radius(M_star, age_gyr):
    """
    Age-dependent stellar radius for VLMS stars

    Parameters:
    -----------
    M_star : float
        Stellar mass (solar masses)
    age_gyr : float
        Stellar age (Gyr)

    Returns:
    --------
    float : Stellar radius in solar radii
    """
    # Empirical relation for VLMS: R/R_sun ~ M_star^0.8 * (1 + 0.1*log10(age/1Gyr))
    # Young stars are larger, contract with age
    if age_gyr <= 0:
        age_gyr = 0.1  # Minimum age

    R_main_sequence = M_star**0.8
    age_factor = 1.0 + 0.1 * np.log10(age_gyr / 1.0)

    return R_main_sequence * age_factor


@jit(nopython=True)
def age_dependent_tidal_q_factor(M_star, age_gyr):
    """
    Age-dependent tidal Q factor for VLMS stars

    Parameters:
    -----------
    M_star : float
        Stellar mass (solar masses)
    age_gyr : float
        Stellar age (Gyr)

    Returns:
    --------
    float : Tidal Q factor (dimensionless)
    """
    # Q increases with age as stars become less active
    # Young stars: Q ~ 10^5-10^6, Old stars: Q ~ 10^7-10^8
    if age_gyr <= 0:
        age_gyr = 0.1

    Q_young = 1e5
    Q_old = 1e7

    # Logarithmic increase with age
    log_Q = np.log10(Q_young) + (np.log10(Q_old) - np.log10(Q_young)) * np.tanh(age_gyr / 5.0)

    return 10**log_Q


@jit(nopython=True)
def kozai_lidov_feasibility_with_age(M_star, M_comp, a_inner, e_inner,
                                   M_perturber, a_outer, e_outer,
                                   age_gyr, target_a=0.05, t_max_gyr=1.0):
    """
    Age-dependent Kozai-Lidov + tides feasibility analysis

    Parameters:
    -----------
    M_star : float
        Primary mass (solar masses)
    M_comp : float
        Companion mass (solar masses)
    a_inner : float
        Inner orbit semimajor axis (AU)
    e_inner : float
        Inner orbit initial eccentricity
    M_perturber : float
        Perturber mass (solar masses)
    a_outer : float
        Perturber semimajor axis (AU)
    e_outer : float
        Perturber eccentricity
    age_gyr : float
        System age (Gyr)
    target_a : float
        Target migration distance (AU)
    t_max_gyr : float
        Maximum evolution time (Gyr)

    Returns:
    --------
    bool : Whether migration to target_a is feasible
    """

    # Basic stability checks
    if a_outer <= a_inner * 3:  # Hill sphere check
        return False

    # Kozai-Lidov timescale
    P_inner = np.sqrt(a_inner**3 / M_star)  # Years
    P_outer = np.sqrt(a_outer**3 / (M_star + M_perturber))

    if P_outer <= P_inner * 3:  # Stability check
        return False

    t_kl_yr = P_outer * (M_star / M_perturber) * (a_outer / a_inner)**3

    # Can it complete cycles within available time?
    available_time = min(t_max_gyr, age_gyr) * 1e9  # Convert to years
    if t_kl_yr > available_time:
        return False

    # Maximum eccentricity from KL
    kl_argument = 1.0 - 5.0/3.0 * (1.0 - e_inner**2)
    if kl_argument <= 0:
        return False

    e_max = np.sqrt(kl_argument)
    e_max = min(e_max, 0.95)

    # Age-dependent stellar properties
    R_star = age_dependent_stellar_radius(M_star, age_gyr)
    Q_star = age_dependent_tidal_q_factor(M_star, age_gyr)

    # Periapsis at maximum eccentricity
    periapsis = a_inner * (1.0 - e_max)

    # Tidal evolution timescale at periapsis
    # t_tidal ~ (Q/9) * (M_star/M_comp) * (a/R_star)^5 / n
    R_star_au = R_star * 0.00465  # Convert solar radii to AU
    if periapsis <= 3.0 * R_star_au:  # Strong tidal regime
        # Simplified tidal timescale
        n = np.sqrt((M_star + M_comp) / periapsis**3)  # Mean motion
        t_tidal_yr = (Q_star / 9.0) * (M_star / M_comp) * (periapsis / R_star_au)**5 / n
        t_tidal_yr *= 365.25 * 24 * 3600  # Convert to years

        # Can tides shrink orbit within available time?
        if t_tidal_yr < available_time:
            # Estimate final semimajor axis after tidal evolution
            # Very simplified - real calculation requires integration
            shrink_factor = np.exp(-available_time / t_tidal_yr)
            final_a = periapsis * (1.0 + shrink_factor) / 2.0  # Circularized orbit

            return final_a <= target_a * 1.2  # Allow some tolerance

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

    def create_age_dependent_feasibility_map(self, age_range: tuple = (0.1, 10.0),
                                           perturber_mass_range: tuple = (0.1, 1.0),
                                           perturber_sep_range: tuple = (10, 1000),
                                           n_age_points: int = 10,
                                           n_mass_points: int = 20,
                                           n_sep_points: int = 20) -> dict:
        """
        Create age-dependent feasibility map for Kozai-Lidov + tides migration

        Parameters:
        -----------
        age_range : tuple
            Range of system ages (Gyr)
        perturber_mass_range : tuple
            Range of perturber masses (solar masses)
        perturber_sep_range : tuple
            Range of perturber separations (AU)
        n_age_points : int
            Number of age grid points
        n_mass_points : int
            Number of mass grid points
        n_sep_points : int
            Number of separation grid points

        Returns:
        --------
        dict with age-dependent feasibility map and parameters
        """

        print(f"Creating age-dependent KL feasibility map ({n_age_points}×{n_mass_points}×{n_sep_points} grid, {self.n_trials} trials each)...")

        # Create grids
        age_grid = np.logspace(np.log10(age_range[0]), np.log10(age_range[1]), n_age_points)
        mass_grid = np.logspace(np.log10(perturber_mass_range[0]),
                               np.log10(perturber_mass_range[1]), n_mass_points)
        sep_grid = np.logspace(np.log10(perturber_sep_range[0]),
                              np.log10(perturber_sep_range[1]), n_sep_points)

        # Initialize results
        feasibility_map = np.zeros((n_age_points, n_mass_points, n_sep_points))

        # TOI-6894b-like system parameters
        M_star = 0.08  # Solar masses
        M_comp = 0.3 / 1047.6  # Jupiter to solar masses
        a_initial = 1.0  # Start at ~1 AU

        # Grid search
        for i, age_gyr in enumerate(age_grid):
            for j, M_pert in enumerate(mass_grid):
                for k, a_pert in enumerate(sep_grid):
                    successes = 0

                    # Monte Carlo trials
                    for trial in range(self.n_trials):
                        # Random initial conditions
                        e_inner = np.random.uniform(0.1, 0.7)  # Initial eccentricity
                        e_outer = np.random.uniform(0.0, 0.3)  # Outer orbit eccentricity

                        # Test age-dependent feasibility
                        if kozai_lidov_feasibility_with_age(M_star, M_comp, a_initial, e_inner,
                                                          M_pert, a_pert, e_outer, age_gyr):
                            successes += 1

                    feasibility_map[i, j, k] = successes / self.n_trials

        results = {
            'feasibility_map': feasibility_map,
            'age_grid': age_grid,
            'perturber_masses': mass_grid,
            'perturber_separations': sep_grid,
            'n_trials': self.n_trials
        }

        max_feasibility = np.max(feasibility_map)
        optimal_age_idx = np.unravel_index(np.argmax(feasibility_map), feasibility_map.shape)[0]
        optimal_age = age_grid[optimal_age_idx]

        print(f"Age-dependent feasibility map completed.")
        print(f"Maximum success rate: {max_feasibility:.3f} at age {optimal_age:.2f} Gyr")

        return results


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

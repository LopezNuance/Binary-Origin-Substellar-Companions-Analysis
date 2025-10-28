import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import matplotlib.patches as patches
from matplotlib.colors import LogNorm

class VLMSVisualizer:
    """Create visualizations for VLMS companion analysis"""

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize
        plt.style.use('default')  # Ensure clean style

    def plot_mass_mass_diagram(self, df: pd.DataFrame, output_file: str = "fig1_massmass.png",
                              toi_params: Optional[dict] = None) -> None:
        """
        Create M_star vs M_companion plot (Figure 1)

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with stellar and companion masses
        output_file : str
            Output filename
        toi_params : dict, optional
            TOI-6894b parameters for highlighting
        """

        fig, ax = plt.subplots(figsize=self.figsize)

        # Convert to Jupiter masses for plotting
        host_mass = df['host_mass_msun']
        comp_mass_mjup = df['companion_mass_mjup']

        # Separate data sources for different markers
        nasa_mask = df['data_source'] == 'NASA'
        bd_mask = df['data_source'] == 'BD_Catalogue'
        toi_mask = df['data_source'] == 'TOI'

        # Plot different data sources
        if nasa_mask.any():
            ax.scatter(host_mass[nasa_mask], comp_mass_mjup[nasa_mask],
                      alpha=0.7, s=50, c='blue', marker='o', label='NASA Archive')

        if bd_mask.any():
            ax.scatter(host_mass[bd_mask], comp_mass_mjup[bd_mask],
                      alpha=0.7, s=50, c='red', marker='^', label='Brown Dwarf Catalogue')

        # Highlight TOI-6894b
        if toi_mask.any():
            ax.scatter(host_mass[toi_mask], comp_mass_mjup[toi_mask],
                      s=200, c='gold', marker='*', edgecolors='black',
                      linewidth=2, label='TOI-6894b', zorder=10)

        # Add deuterium burning limit line (13 MJ)
        ax.axhline(y=13.0, color='green', linestyle='--', linewidth=2,
                  alpha=0.8, label='Deuterium burning limit (13 M$_J$)')

        # Add hydrogen burning limit line (~75-80 MJ)
        ax.axhline(y=78.0, color='orange', linestyle='--', linewidth=2,
                  alpha=0.8, label='Hydrogen burning limit (~78 M$_J$)')

        # Formatting
        ax.set_xlabel('Host Mass (M$_☉$)', fontsize=14)
        ax.set_ylabel('Companion Mass (M$_J$)', fontsize=14)
        ax.set_title('Very Low Mass Stars and Their Companions', fontsize=16)

        # Log scale for y-axis to show full range
        ax.set_yscale('log')
        ax.set_xlim(0.05, 0.21)
        ax.set_ylim(0.1, 200)

        # Grid
        ax.grid(True, alpha=0.3)

        # Legend
        ax.legend(loc='upper left', fontsize=10)

        # Add text annotation
        ax.text(0.52, 0.95, f'VLMS hosts: {len(df[~toi_mask])} objects',
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Figure 1 to {output_file}")

    def plot_architecture_diagram(self, df: pd.DataFrame, output_file: str = "fig2_ae.png") -> None:
        """
        Create eccentricity vs semi-major axis plot (Figure 2)

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with orbital parameters
        output_file : str
            Output filename
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left panel: e vs a
        # Color by mass ratio
        scatter = ax1.scatter(df['semimajor_axis_au'], df['eccentricity'],
                            c=df['log_mass_ratio'], cmap='viridis',
                            alpha=0.7, s=50)

        # Highlight TOI-6894b
        toi_mask = df['data_source'] == 'TOI'
        if toi_mask.any():
            ax1.scatter(df[toi_mask]['semimajor_axis_au'], df[toi_mask]['eccentricity'],
                       s=200, c='red', marker='*', edgecolors='black',
                       linewidth=2, label='TOI-6894b', zorder=10)

        ax1.set_xlabel('Semi-major axis (AU)', fontsize=14)
        ax1.set_ylabel('Eccentricity', fontsize=14)
        ax1.set_title('Orbital Architecture', fontsize=14)
        ax1.set_xscale('log')
        ax1.set_xlim(0.01, 1.0)
        ax1.set_ylim(-0.05, 1.0)
        ax1.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('log$_{10}$(q)', fontsize=12)

        if toi_mask.any():
            ax1.legend(loc='upper right')

        # Right panel: e distribution by mass ratio
        high_q = df[df['high_mass_ratio']]
        low_q = df[~df['high_mass_ratio']]

        ax2.hist(low_q['eccentricity'], bins=15, alpha=0.7, label='Low q',
                density=True, color='blue')
        ax2.hist(high_q['eccentricity'], bins=15, alpha=0.7, label='High q',
                density=True, color='red')

        ax2.set_xlabel('Eccentricity', fontsize=14)
        ax2.set_ylabel('Density', fontsize=14)
        ax2.set_title('Eccentricity Distribution', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Figure 2 to {output_file}")

    def plot_feasibility_map(self, feasibility_data: np.ndarray,
                           perturber_masses: np.ndarray,
                           perturber_separations: np.ndarray,
                           output_file: str = "fig3_feasibility.png") -> None:
        """
        Create Kozai-Lidov + tides feasibility map (Figure 3)

        Parameters:
        -----------
        feasibility_data : np.ndarray
            2D array of success fractions
        perturber_masses : np.ndarray
            Perturber masses (solar masses)
        perturber_separations : np.ndarray
            Perturber separations (AU)
        output_file : str
            Output filename
        """

        fig, ax = plt.subplots(figsize=self.figsize)

        # Create meshgrid for contour plot
        X, Y = np.meshgrid(perturber_separations, perturber_masses)

        # Contour plot
        contour = ax.contourf(X, Y, feasibility_data, levels=20, cmap='plasma')
        contour_lines = ax.contour(X, Y, feasibility_data, levels=[0.1, 0.3, 0.5, 0.7],
                                  colors='white', linewidths=1.5, alpha=0.8)
        ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%0.1f')

        # Formatting
        ax.set_xlabel('Perturber Separation (AU)', fontsize=14)
        ax.set_ylabel('Perturber Mass (M$_☉$)', fontsize=14)
        ax.set_title('Kozai-Lidov + Tides Migration Feasibility', fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Fraction reaching a ~ 0.05 AU within 1 Gyr', fontsize=12)

        # Add text
        ax.text(0.02, 0.95, 'Higher values indicate\nmore feasible migration',
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Figure 3 to {output_file}")

    def plot_gmm_analysis(self, df: pd.DataFrame, gmm_results: dict,
                         output_file: str = "gmm_analysis.png") -> None:
        """
        Plot Gaussian Mixture Model results in (log q, log a) space

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
        gmm_results : dict
            GMM fitting results
        output_file : str
            Output filename
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left: Data with cluster assignments
        if 'cluster_labels' in gmm_results:
            labels = gmm_results['cluster_labels']
            unique_labels = np.unique(labels)
            colors = ['blue', 'red', 'green', 'purple']

            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax1.scatter(df[mask]['log_mass_ratio'], df[mask]['log_semimajor_axis'],
                           alpha=0.7, c=colors[i % len(colors)], s=50,
                           label=f'Component {label}')

        else:
            ax1.scatter(df['log_mass_ratio'], df['log_semimajor_axis'],
                       alpha=0.7, c='blue', s=50)

        # Highlight TOI-6894b
        toi_mask = df['data_source'] == 'TOI'
        if toi_mask.any():
            ax1.scatter(df[toi_mask]['log_mass_ratio'], df[toi_mask]['log_semimajor_axis'],
                       s=200, c='gold', marker='*', edgecolors='black',
                       linewidth=2, label='TOI-6894b', zorder=10)

        ax1.set_xlabel('log$_{10}$(q)', fontsize=14)
        ax1.set_ylabel('log$_{10}$(a [AU])', fontsize=14)
        ax1.set_title('GMM Clustering in (log q, log a) Space', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Right: BIC comparison
        if 'bic_scores' in gmm_results:
            n_components = range(1, len(gmm_results['bic_scores']) + 1)
            ax2.plot(n_components, gmm_results['bic_scores'], 'bo-', linewidth=2)
            ax2.set_xlabel('Number of Components', fontsize=14)
            ax2.set_ylabel('BIC Score', fontsize=14)
            ax2.set_title('Model Selection (Lower BIC = Better)', fontsize=14)
            ax2.grid(True, alpha=0.3)

            # Mark best model
            best_n = np.argmin(gmm_results['bic_scores']) + 1
            ax2.axvline(x=best_n, color='red', linestyle='--', alpha=0.7,
                       label=f'Best: {best_n} components')
            ax2.legend()

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved GMM analysis to {output_file}")

    def plot_classification_results(self, df: pd.DataFrame, probabilities: np.ndarray,
                                  output_file: str = "classification_results.png") -> None:
        """
        Plot classification results

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
        probabilities : np.ndarray
            Binary-like origin probabilities
        output_file : str
            Output filename
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left: Scatter plot colored by probability
        scatter = ax1.scatter(df['log_mass_ratio'], df['log_semimajor_axis'],
                            c=probabilities, cmap='RdYlBu_r',
                            alpha=0.7, s=50, vmin=0, vmax=1)

        # Highlight TOI-6894b
        toi_mask = df['data_source'] == 'TOI'
        if toi_mask.any():
            ax1.scatter(df[toi_mask]['log_mass_ratio'], df[toi_mask]['log_semimajor_axis'],
                       s=200, c='black', marker='*', edgecolors='white',
                       linewidth=2, label='TOI-6894b', zorder=10)

        ax1.set_xlabel('log$_{10}$(q)', fontsize=14)
        ax1.set_ylabel('log$_{10}$(a [AU])', fontsize=14)
        ax1.set_title('Binary-like Origin Probability', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('P(binary-like)', fontsize=12)

        if toi_mask.any():
            ax1.legend()

        # Right: Histogram of probabilities
        ax2.hist(probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')

        # Mark TOI-6894b probability if available
        if toi_mask.any() and len(probabilities[toi_mask]) > 0:
            toi_prob = probabilities[toi_mask][0]
            ax2.axvline(x=toi_prob, color='red', linestyle='--', linewidth=3,
                       label=f'TOI-6894b: P = {toi_prob:.3f}')
            ax2.legend()

        ax2.set_xlabel('P(binary-like)', fontsize=14)
        ax2.set_ylabel('Number of Objects', fontsize=14)
        ax2.set_title('Distribution of Origin Probabilities', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved classification results to {output_file}")

    def compose_migration_vs_kl(self, disk_png, kl_png, out_png):
        """
        Create side-by-side comparison of disk migration and KL feasibility

        Parameters:
        -----------
        disk_png : str
            Path to disk migration figure
        kl_png : str
            Path to KL feasibility figure
        out_png : str
            Output path for combined figure
        """
        import matplotlib.image as mpimg

        # Load images
        img_disk = mpimg.imread(disk_png)
        img_kl = mpimg.imread(kl_png)

        # Create composite figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                      constrained_layout=True)

        ax1.imshow(img_disk)
        ax1.axis('off')
        ax1.set_title("Disk Migration Timescales", fontsize=12)

        ax2.imshow(img_kl)
        ax2.axis('off')
        ax2.set_title("Kozai-Lidov + Tides Feasibility", fontsize=12)

        fig.suptitle("Inward Hardening Pathways for VLMS Companions",
                    fontsize=14, fontweight='bold')

        fig.savefig(out_png, dpi=200, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    # Test the visualizer
    viz = VLMSVisualizer()

    # Create dummy data for testing
    np.random.seed(42)
    n_objects = 50

    dummy_data = pd.DataFrame({
        'host_mass_msun': np.random.uniform(0.06, 0.20, n_objects),
        'companion_mass_mjup': np.random.lognormal(np.log(5), 1, n_objects),
        'semimajor_axis_au': np.random.lognormal(np.log(0.1), 1, n_objects),
        'eccentricity': np.random.beta(0.867, 3.03, n_objects),  # From literature
        'data_source': np.random.choice(['NASA', 'BD_Catalogue'], n_objects),
        'high_mass_ratio': np.random.choice([True, False], n_objects)
    })

    # Add derived quantities
    dummy_data['companion_mass_msun'] = dummy_data['companion_mass_mjup'] / 1047.6
    dummy_data['mass_ratio'] = dummy_data['companion_mass_msun'] / dummy_data['host_mass_msun']
    dummy_data['log_mass_ratio'] = np.log10(dummy_data['mass_ratio'])
    dummy_data['log_semimajor_axis'] = np.log10(dummy_data['semimajor_axis_au'])

    # Test visualization functions
    print("Testing visualization functions...")
    viz.plot_mass_mass_diagram(dummy_data, "test_fig1.png")
    viz.plot_architecture_diagram(dummy_data, "test_fig2.png")

    # Test feasibility map
    test_feas = np.random.rand(20, 20)
    test_masses = np.logspace(-1, 0, 20)
    test_seps = np.logspace(1, 3, 20)
    viz.plot_feasibility_map(test_feas, test_masses, test_seps, "test_fig3.png")

    print("Visualization tests completed!")
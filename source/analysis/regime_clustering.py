"""
Statistical regime discovery using clustering and mixture models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from pathlib import Path


def run_hdbscan_and_gmm(systems_csv="results/combined_systems.csv",
                       out_prefix="results/regimes",
                       min_cluster_size=5,
                       random_state=42):
    """
    Perform HDBSCAN clustering and GMM validation

    Returns:
    --------
    tuple : (labeled_dataframe, bic_info_dict)
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("Please install hdbscan: pip install hdbscan")

    # Load system data
    df = pd.read_csv(systems_csv).dropna(subset=["logq_max", "loga_min"])

    # Feature matrix
    feature_cols = ["logq_max", "loga_min", "e_max"]
    X = df[feature_cols].copy()

    # Handle missing values
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    # Standardize features
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X.values)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(Xz)
    df["hdbscan_label"] = labels

    # GMM with BIC selection
    Ks = list(range(1, 7))
    bics = []
    gmms = []

    for k in Ks:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=random_state)
        gmm.fit(Xz)
        bics.append(gmm.bic(Xz))
        gmms.append(gmm)

    # Select best model
    k_best = Ks[int(np.argmin(bics))]
    gmm_best = gmms[int(np.argmin(bics))]
    df["gmm_label"] = gmm_best.predict(Xz)

    # Save results
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out_prefix}_labels.csv", index=False)

    # Create visualization
    fig, ax = plt.subplots(figsize=(7, 6))

    for lab in sorted(np.unique(labels)):
        if lab == -1:  # Noise points
            continue
        sel = df["hdbscan_label"] == lab
        ax.scatter(df.loc[sel, "logq_max"], df.loc[sel, "loga_min"],
                  s=50, label=f"Cluster {lab}", alpha=0.7)

    # Plot noise points
    noise = df["hdbscan_label"] == -1
    if noise.any():
        ax.scatter(df.loc[noise, "logq_max"], df.loc[noise, "loga_min"],
                  s=20, c='gray', alpha=0.3, label="Unclustered")

    ax.set_xlabel(r"$\log_{10}\,q_{\rm max}$")
    ax.set_ylabel(r"$\log_{10}\,a_{\rm min}\,{\rm [AU]}$")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.set_title("HDBSCAN Regime Discovery")

    fig.tight_layout()
    fig.savefig(f"{out_prefix}_hdbscan_logq_loga.png", dpi=180)
    plt.close(fig)

    # BIC plot
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(Ks, bics, marker='o')
    ax2.set_xlabel("GMM Components K")
    ax2.set_ylabel("BIC (lower is better)")
    ax2.axvline(k_best, ls='--', color='k', alpha=0.5,
               label=f"Best K={k_best}")
    ax2.legend()

    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_gmm_bic.png", dpi=180)
    plt.close(fig2)

    return df, {"K_best": k_best, "BICs": bics}
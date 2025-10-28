"""
Segmented regression analysis for phase-shift detection
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path
import json


def fit_one_break(logq, loga):
    """
    Fit piecewise linear regression with one break point

    Returns:
    --------
    tuple : (break_index, order, model_left, model_right, bic0, bic1, is_better)
    """
    try:
        import ruptures as rpt
    except ImportError:
        raise ImportError("Please install ruptures: pip install ruptures")

    x = np.asarray(logq).reshape(-1, 1)
    y = np.asarray(loga)

    # Sort by x for meaningful change-point detection
    order = np.argsort(x[:, 0])
    x, y = x[order], y[order]

    # Fit single line (0 breaks)
    ols0 = LinearRegression().fit(x, y)
    rss0 = np.sum((y - ols0.predict(x))**2)
    n, p0 = len(y), 2  # slope + intercept
    bic0 = n * np.log(rss0/n) + p0 * np.log(n)

    # Detect 1 break using PELT
    algo = rpt.Pelt(model="rbf").fit(y)
    cp = algo.predict(pen=np.log(n) * 5)  # Mild penalty

    # Check if valid break found
    k = None
    if len(cp) >= 2:
        k = cp[0]

    if k is not None and 2 <= k <= n-2:  # Keep interior breaks only
        xl, yl = x[:k], y[:k]
        xr, yr = x[k:], y[k:]

        olsL = LinearRegression().fit(xl, yl)
        olsR = LinearRegression().fit(xr, yr)

        rss1 = np.sum((yl - olsL.predict(xl))**2) + \
               np.sum((yr - olsR.predict(xr))**2)
        p1 = 4  # 2 params per segment
        bic1 = n * np.log(rss1/n) + p1 * np.log(n)

        better = (bic1 + 2.0) < bic0  # Small safety margin

        return (k, order, olsL, olsR, bic0, bic1, better)

    return (None, order, ols0, None, bic0, None, False)


def run_segmented_plot(systems_csv="results/combined_systems.csv",
                      out_png="results/segmented_logq_loga.png"):
    """
    Create segmented regression visualization

    Returns:
    --------
    dict : Summary statistics
    """
    df = pd.read_csv(systems_csv).dropna(subset=["logq_max", "loga_min"])

    res = fit_one_break(df["logq_max"].values, df["loga_min"].values)
    k, order, mL, mR, bic0, bic1, better = res

    x = df["logq_max"].values[order]
    y = df["loga_min"].values[order]

    # Create visualization
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=40, color="tab:blue", alpha=0.8, label="Systems")

    if k is not None and better:
        # Plot two segments
        xx = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)

        # Left segment
        mask_left = xx[:, 0] <= x[k]
        if mask_left.any():
            ax.plot(xx[mask_left], mL.predict(xx[mask_left]),
                   color="tab:red", lw=2, label="Regime 1")

        # Right segment
        mask_right = xx[:, 0] >= x[k]
        if mask_right.any():
            ax.plot(xx[mask_right], mR.predict(xx[mask_right]),
                   color="tab:green", lw=2, label="Regime 2")

        # Mark break point
        ax.axvline(x[k], ls="--", color="k", alpha=0.5,
                  label=f"Break at log q={x[k]:.2f}")
        ax.set_title(f"Segmented fit preferred (ΔBIC={bic0-bic1:.1f})")
    else:
        # Single line
        xx = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
        ax.plot(xx, mL.predict(xx), color="tab:red", lw=2, label="Single trend")
        ax.set_title("No significant break detected")

    ax.set_xlabel(r"$\log_{10}\,q_{\rm max}$")
    ax.set_ylabel(r"$\log_{10}\,a_{\rm min}\,{\rm [AU]}$")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    # Save summary
    summary = {
        "break_supported": bool(better),
        "break_logq": float(x[k]) if (k is not None and better) else None,
        "bic_single": float(bic0),
        "bic_two": float(bic1) if bic1 is not None else None,
    }

    json_path = out_png.replace(".png", ".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
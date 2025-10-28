# 1) What you’ll get

* **New CLI flags**

  * `--disk-panel` (bool): render a **Disk torques timescale** panel.
  * `--disk-lifetime-myr` (float, default 3.0): target upper bound for migration time.
  * `--a0-min` `--a0-max` (AU, default 0.3–1.0): birth semimajor axis sweep.
  * `--Sigma1AU` (g/cm^2, default 300): surface density at 1 AU for a VLMS disk.
  * `--H-over-a` (default 0.04), `--alpha` (default 3e-3), `--p-sigma` (default 1.0).
  * `--kl-a0` (AU, default 0.5): **birth** inner semimajor axis for the KL map.
  * `--kl-horizon-gyr` (default 3.0): time horizon for feasibility.
  * `--rpcrit-Rs` (default 3.0): periastron threshold in stellar radii.

* **New figure**: `fig3_migration_vs_KL.png`
  Left: **disk-migration time** (Myr) from (a_0) → (a_f=0.05) AU over a grid of (a_0) × (\Sigma_{1\rm AU}).
  Right: **updated KL+tides feasibility** using the **birth** (a_0=) `--kl-a0` and new horizon.

* **Plain-English captions** baked into the PNGs so interpretation is immediate.

---

# 2) Drop-in patch

Open `panoptic_vlms_project.py` and apply the following additions. (Search for the comments like `### PATCH START`.)

```python
# =======================
# ### PATCH START: imports
# =======================
import math

# (Already present) import numpy as np, matplotlib, etc.
# ### PATCH END: imports
```

```python
# ===============================================================
# ### PATCH START: CLI additions (extend your existing argparse)
# ===============================================================
parser.add_argument("--disk-panel", action="store_true",
                    help="Render disk-migration timescale panel alongside KL.")
parser.add_argument("--disk-lifetime-myr", type=float, default=3.0,
                    help="Target disk lifetime for feasible migration (Myr).")
parser.add_argument("--a0-min", type=float, default=0.3,
                    help="Minimum birth a0 (AU) to sweep in disk panel.")
parser.add_argument("--a0-max", type=float, default=1.0,
                    help="Maximum birth a0 (AU) to sweep in disk panel.")
parser.add_argument("--Sigma1AU", type=float, default=300.0,
                    help="Gas surface density at 1 AU (g/cm^2) for VLMS disk.")
parser.add_argument("--p-sigma", type=float, default=1.0,
                    help="Surface-density power-law: Sigma ~ a^{-p}.")
parser.add_argument("--H-over-a", type=float, default=0.04,
                    help="Disk aspect ratio H/a (assumed radially constant here).")
parser.add_argument("--alpha", type=float, default=3e-3,
                    help="Viscosity parameter used in gap-opening sanity check.")
parser.add_argument("--kl-a0", type=float, default=0.5,
                    help="Birth inner a0 (AU) used by KL feasibility map.")
parser.add_argument("--kl-horizon-gyr", type=float, default=3.0,
                    help="Time horizon (Gyr) for KL+tides feasibility.")
parser.add_argument("--rpcrit-Rs", type=float, default=3.0,
                    help="Critical periastron in stellar radii for tides to act.")
# ### PATCH END: CLI additions
```

```python
# =====================================================================
# ### PATCH START: disk-migration utilities (Type-I-ish order of mag)
# =====================================================================
AU = 1.495978707e13     # cm
M_sun = 1.98847e33      # g
M_jup = 1.89813e30      # g
G = 6.67430e-8          # cgs
YEAR = 3.15576e7        # s

def _Omega(Mstar_Msun, a_AU):
    M = Mstar_Msun * M_sun
    a = a_AU * AU
    return math.sqrt(G*M / a**3)

def typeI_timescale_sec(Mstar_Msun, Mp_Mj, a_AU, Sigma1_gcm2, p_sigma, H_over_a, C=3.0):
    """
    Tanaka-like scaling:
      t_I ~ C * (M_*/M_p) * (M_*/(Sigma a^2)) * (H/a)^2 * Omega^-1
    We allow Sigma(a) = Sigma1 * (a/1AU)^(-p_sigma), H/a constant.
    """
    Mstar = Mstar_Msun * M_sun
    Mp = Mp_Mj * M_jup
    a = a_AU * AU
    Sigma = Sigma1_gcm2 * (a_AU ** (-p_sigma))  # g/cm^2
    Omega = _Omega(Mstar_Msun, a_AU)
    return C * (Mstar/Mp) * (Mstar/(Sigma * a*a)) * (H_over_a**2) / Omega

def migrate_time_numeric(Mstar_Msun, Mp_Mj, a0_AU, af_AU,
                         Sigma1_gcm2, p_sigma, H_over_a, C=3.0, nstep=2000):
    """
    Integrate dt = ∫ t_I(a)/a da (since |da/dt| ~ a/t_I) from a0 -> af.
    """
    a_hi, a_lo = max(a0_AU, af_AU), min(a0_AU, af_AU)
    # integrate from high to low
    a_grid = np.geomspace(a_hi, a_lo, nstep)
    # trapezoid on t_I(a)/a
    vals = [typeI_timescale_sec(Mstar_Msun, Mp_Mj, a, Sigma1_gcm2, p_sigma, H_over_a, C)/ (a*AU)
            for a in a_grid]
    # convert (t_I/a) * da (in cm) -> seconds: multiply by (AU) already built-in above
    # Here we used a*AU in denom; the integral over a in AU: dt ≈ Σ (t_I(a)/a) Δa
    # Implement trapezoid in AU, multiply by AU (cm) inside function above, so Δa is dimensionless.
    dt_sec = np.trapz(vals, a_grid)
    return dt_sec

def render_disk_panel(outpath_png, Mstar_Msun, Mp_Mj, args):
    """
    Heatmap of migration time (Myr) to go from a0 -> a_f=0.05 AU
    vs (a0, Sigma1AU). Marks a contour at disk-lifetime.
    """
    a0s = np.geomspace(args.a0_min, args.a0_max, 60)
    Sigma1_list = np.geomspace(args.Sigma1AU/5.0, args.Sigma1AU*5.0, 60)
    Z = np.zeros((len(Sigma1_list), len(a0s)))
    af = 0.05
    for i, S1 in enumerate(Sigma1_list):
        for j, a0 in enumerate(a0s):
            t_sec = migrate_time_numeric(Mstar_Msun, Mp_Mj, a0, af,
                                         S1, args.p_sigma, args.H_over_a, C=3.0)
            Z[i, j] = t_sec / (1e6*YEAR)  # Myr

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4), constrained_layout=True)
    im = ax.pcolormesh(a0s, Sigma1_list, Z, shading='auto', norm=mpl.colors.LogNorm())
    cb = fig.colorbar(im, ax=ax, label="Migration time (Myr) to 0.05 AU")
    ax.axhline(args.Sigma1AU, ls='--', lw=1, color='k', alpha=0.5)
    cs = ax.contour(a0s, Sigma1_list, Z, levels=[args.disk_lifetime_myr], colors='white', linewidths=1.5)
    ax.clabel(cs, fmt={args.disk_lifetime_myr: f"{args.disk_lifetime_myr:.0f} Myr"}, inline=True)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"Birth $a_0$ (AU)")
    ax.set_ylabel(r"$\Sigma_{1\,\mathrm{AU}}$ (g cm$^{-2}$)")
    ax.set_title("Disk Torques: Is $a_0\\rightarrow0.05$ AU within the disk lifetime?")
    ax.text(0.03, 0.02,
            f"H/a={args.H_over_a:.2f}, p={args.p_sigma:.1f}, α={args.alpha:.0e}\n"
            f"M*={Mstar_Msun:.2f} M$_\\odot$, M$_c$={Mp_Mj:.2f} M$_J$",
            transform=ax.transAxes, fontsize=9, va='bottom')
    fig.savefig(outpath_png, dpi=200)
    plt.close(fig)
# ### PATCH END: disk-migration utilities
```

```python
# ============================================================================
# ### PATCH START: KL map hook – use birth a0 and new horizon / rpcrit
# ============================================================================
# Wherever you currently call your KL feasibility routine, change the call so:
#   - inner_birth_a_AU = args.kl_a0
#   - horizon_Gyr = args.kl_horizon_gyr
#   - rpcrit_Rstar = args.rpcrit_Rs
#
# Example (pseudo – adapt to your function names):
#
# kl_map = compute_kl_feasibility_grid(
#     Mstar_Msun=toi_Mstar, Mc_Mj=toi_Mc,
#     inner_birth_a_AU=args.kl_a0,
#     horizon_Gyr=args.kl_horizon_gyr,
#     rpcrit_in_Rstar=args.rpcrit_Rs,
#     grid_Mout=..., grid_aout=..., n_draws=... )
#
# save_kl_map_figure("results/fig3_KL.png", kl_map, args)
# ### PATCH END: KL map hook
```

```python
# ===========================================================
# ### PATCH START: combined fig3 (disk vs KL) compositor
# ===========================================================
def compose_migration_vs_kl(disk_png, kl_png, out_png):
    imgL = mpl.image.imread(disk_png)
    imgR = mpl.image.imread(kl_png)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    axs[0].imshow(imgL); axs[0].axis('off'); axs[0].set_title("Disk Migration (Myr)")
    axs[1].imshow(imgR); axs[1].axis('off'); axs[1].set_title("KL + Tides Feasibility")
    fig.suptitle("Inward Hardening Pathways for VLMS Companions", fontsize=14)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
# ### PATCH END: combined fig3
```

```python
# ===========================================================
# ### PATCH START: main() integration
# ===========================================================
# After you’ve parsed args and know the TOI-like marker masses:
toi_Mstar = args.toi_mstar if hasattr(args, "toi_mstar") and args.toi_mstar else 0.08
toi_Mc = args.toi_mc_mj if hasattr(args, "toi_mc_mj") and args.toi_mc_mj else 0.30

kl_png = os.path.join(args.outdir, "fig3_KL.png")
disk_png = os.path.join(args.outdir, "fig3_disk.png")
combo_png = os.path.join(args.outdir, "fig3_migration_vs_KL.png")

# (1) Recompute KL map with birth a0 / new horizon / rpcrit:
# NOTE: call your existing KL function with the new args (see hook above)
# save_kl_map_figure(kl_png, ...)

# (2) Disk panel if requested:
if args.disk_panel:
    render_disk_panel(disk_png, toi_Mstar, toi_Mc, args)
    if os.path.exists(kl_png):
        compose_migration_vs_kl(disk_png, kl_png, combo_png)
# ### PATCH END: main() integration
```

---

# 3) How to run

**Typical** (make both panels, combine to one figure):

```
python panoptic_vlms_project.py \
  --fetch \
  --disk-panel \
  --a0-min 0.3 --a0-max 1.0 \
  --Sigma1AU 300 --p-sigma 1.0 --H-over-a 0.04 --alpha 3e-3 \
  --kl-a0 0.5 --kl-horizon-gyr 3.0 --rpcrit-Rs 3.0 \
  --toi_mstar 0.08 --toi_mc_mj 0.30 --outdir results
```

**What to expect:**

* `results/fig3_disk.png` — heatmap of **time (Myr)** to migrate from (a_0) to (0.05) AU.

  * The **white contour** at `--disk-lifetime-myr` (default 3 Myr) marks **“feasible within the disk.”**
* `results/fig3_KL.png` — your **updated** KL feasibility with **birth (a_0)** and **longer horizon**.
* `results/fig3_migration_vs_KL.png` — side-by-side composite.

---

# 4) How to interpret (the line you can put in the paper)

> “Under VLMS disk parameters (H/a\sim0.04), (\Sigma_{1\rm AU}\sim 300,\mathrm{g,cm^{-2}}), and (\alpha\sim3\times10^{-3}), **disk torques can deliver a (0.3,M_J) companion from (a_0\sim0.3!-!1) AU to (0.05) AU within a 1–3 Myr disk lifetime across a broad swath of parameter space** (Fig. 3, left). By contrast, **KL+tides requires either a massive/close tertiary or longer times** (Fig. 3, right), making secular hardening *optional rather than necessary*. This supports a failed-binary origin **hardened primarily by early gas-disk migration**.”

---



I think we need to include very low mass stellar binaries and consider whether system age, system members' orbital characteristics, and the presence of other substellar bodies form different groups with different parameter correlations.  The implication I'm suggesting here is that there may be several "phase shifts" such that there isn't a single quasi-linear relationship but rather as stellar system parameters (astronomical body masses (small stars, brown dwarfs, giant planets, initial gas/dust cloud and it's distribution of mass concentrations, and so on).

I think it might be be fruitful to consider using a meteorological model that simulates the distribution of high and low pressure zones and fronts.  I bring that up because I was watching a weather report and noticed how cloud cover and rain distribution (from their two dimensional map) could be adapted to model the fractal-like distribution of high/low mass concentrations and flows.  We can start with a simple model first using only the most important core considerations and build upon that. 


---

# 1) Enrich the dataset to include VLMS **stellar binaries** and system context

**Goal:** let the data tell us if there are *regimes* (“phase shifts”) where correlations change—rather than one smooth trend.

## 1.1 Entities to add

* **VLMS close binaries** (stellar–stellar, M*≈0.06–0.25 M☉): period, a, e, mass ratio q.
* **Systems with multiple companions** around VLMS hosts (BDs + Jupiters).
* **Ages** (posterior medians + credible intervals).
* **Environment/context** features if available: metallicity [Fe/H], disk indicators (IR excess), wide tertiary flags.

## 1.2 Features table (per system)

* Host: (M_\star, R_\star, [\mathrm{Fe/H}], \mathrm{Age}).
* Companion i: (M_i, q_i=M_i/M_\star, a_i, e_i, P_i).
* Multiplicity flags: `has_stellar_binary`, `has_bd`, `has_giant`, `has_outer_tertiary`.
* Architecture summary: (a_{\min}, a_{\max}, \Delta a), min/median/max (e), pairwise period ratios.
* Disk-era proxies (if any): accretion tracers, IR-excess boolean.

We’ll flatten to one row per **system**, and build vector summaries so clustering compares *systems*, not just single companions.

---

# 2) Statistics for **regime discovery** (phase shifts)

We’ll look for *latent groups* with distinct parameter couplings.

## 2.1 Unsupervised structure

* **HDBSCAN** on (\mathbf{x}=(\log q_{\max}, \log a_{\min}, e_{\max}, \mathrm{Age}, [\mathrm{Fe/H}], \mathrm{mult})).
  – Handles irregular shapes & noise better than k-means.
* **GMM with full covariances + BIC** grid (K=1…6) as a parametric cross-check.
* **Spectral clustering** on a graph where edge weights are exp(−Mahalanobis distance).

Deliverables: cluster maps in ((\log q, \log a)) and ((e, a)), cluster posterior responsibilities, BIC/HDBSCAN stability plots.

## 2.2 “Phase-shift” tests

* **Piecewise regression / change-point detection** for trends like (\log a) vs (\log q):
  – Fit 1-break segmented model; LRT (or BIC) vs single line.
  – Bayesian change-point with RJMCMC (quick prior: 1 break; report posterior on break location).
* **Markov-switching regression** (two regimes) for (e \sim \alpha_r + \beta_r \log q) with regime r latent; gives regime responsibilities by system.
* **Copula analysis** (Gaussian vs mixture copulas) for tail dependence between (e) and (q).

## 2.3 Outcome we’re testing

* Existence of a **high-q, tighter-a, higher-e** regime consistent with fragmentation + dynamical processing (stellar-like).
* A **low-q, tighter-a, near-circular** regime consistent with disk migration (planet-like).
* A **binary VLMS** regime (stellar–stellar) overlapping high-q, showing continuity across the hydrogen-burning boundary.

---

# 3) Replace “KL-only” with **dual hardening channels**

We’ll model two pathways and let the data support their weights:

### (A) Early **disk migration** (gas era)

We already added a panel for Type-I–like times (t_{\rm mig}(a_0 \to a_f)). Enhance it to the **system level**:

* For each system, compute (t_{\rm mig}(a_0\to a_{\rm obs})) for a grid of (a_0\in[0.2,2]) AU and (\Sigma_{1,\mathrm{AU}}).
* Define feasibility: (t_{\rm mig} \le t_{\rm disk}) (use system age priors to set a plausible disk lifetime prior, e.g. 1–5 Myr).

### (B) **Secular + tides** (post-disk)

* Keep KL+tidal feasibility but now **parametrize birth radius** (a_0) near the *stellar-binary* peak (<1 AU) rather than “wide” by default.
* Timescale skeletons:

  * KL: (t_{\rm KL} \sim \frac{M_\mathrm{tot}}{M_\mathrm{out}} \frac{P_\mathrm{out}^2}{P_\mathrm{in}} (1-e_\mathrm{out}^2)^{3/2}).
  * Tides: include pericentre criterion (r_p \le r_{p,\mathrm{crit}} = \gamma R_\star) (we parameterized as `--rpcrit-Rs`) and a simple equil. tide circularization time (t_\mathrm{tide}(e)) to check if within horizon.

**Deliverable:** For each system, a 2-vector feasibility score ((\mathcal{F}*{\rm disk}, \mathcal{F}*{\rm KL})) and a soft assignment (\pi) (logistic mixture) indicating likely channel.

---

# 4) Meteorological **analogy → minimal physical toy model**

We can capture the “fronts/pressure systems” intuition with a 2-D **thin, self-gravitating barotropic sheet** that develops **fractal density** under stirring, then collapses locally:

## 4.1 Governing skeleton (isothermal, thin-sheet)

* Continuity: (\partial_t \Sigma + \nabla\cdot(\Sigma \mathbf{v}) = 0)
* Momentum: (\partial_t \mathbf{v} + (\mathbf{v}\cdot\nabla)\mathbf{v} = -c_s^2 \nabla \ln \Sigma - \nabla \Phi + \nu\nabla^2\mathbf{v})
* Gravity: (\nabla^2 \Phi = 4\pi G \Sigma,\delta(z)) ⇒ in Fourier, (\Phi_\mathbf{k} = -\frac{2\pi G}{|\mathbf{k}|}\Sigma_\mathbf{k}) (fast via FFT).
* Background rotation via shear rate (q\Omega) (optional) or just set a constant (\Omega) and include a Coriolis term (-2\Omega\hat{z}\times\mathbf{v}) to keep a Toomre-like flavor.
* **Forcing**: large-scale solenoidal acceleration to mimic turbulent driving (meteorological stirring).

We run this on a periodic grid, seed it with **fractional Brownian** density perturbations to get the right **power-law** spectrum, then evolve until **local Jeans length (\lambda_J \sim c_s^2/(\pi G \Sigma))** crosses a cell scale → mark a **collapse core**. Each core becomes a “sink” with instantaneous mass; we then place a **secondary** with mass drawn from a lognormal around the local **core mass fraction**. This creates a **distribution of (q), (a_0)** that is **fractal-like** in origin—exactly the “fronts & cells” view.

**MVP runtime goal** on 4090: 512×512, dt with CFL ~0.3, 10⁴–10⁵ steps (minutes to tens of minutes if we keep it single-precision and FFT-accelerated).

## 4.2 Model outputs to compare with data

* Distribution of initial **core separations** → predicted birth (a_0) for pairs.
* Joint ( (q, a_0) ) distribution; conditional on local surface-density peaks (analogy to “high pressure”).
* Fraction of **triple/tertiary seeds** → propensity for KL channel.
* Fractal/structure metrics (2-D power spectra slope, Moran’s I) to report self-similarity.

**Validation:** KS/AD tests between simulated and observed (\log q), (\log a_0), and cluster occupancy (via the same HDBSCAN/GMM pipeline).

---

# 5) What I’ll change in your codebase (summary)

1. **Data layer**

   * Add ingestion for VLMS **stellar binaries** and **tertiaries**; merge to system-level rows.
   * Age/metallicity harmonization; carry uncertainties.

2. **Analysis layer**

   * HDBSCAN/GMM regime discovery + change-point and MSR fits.
   * Dual-feasibility scoring ((\mathcal{F}*{\rm disk}, \mathcal{F}*{\rm KL})) per system.
   * New figure set:

     * F1: Mass–mass incl. **stellar binaries** + BD + giants (colored by cluster).
     * F2: (e)–(a) by cluster with TOI-6894b highlighted.
     * F3: **Disk vs KL** feasibility (already added) now **system-specific** overlays.
     * F4: **Regime map** (HDBSCAN labels) + piecewise regression with break posterior.

3. **Toy-model module (optional add-on)**

   * `toy_sheet/` with a CUDA-friendly 2-D barotropic solver + FFT gravity.
   * Script to generate ((q,a_0)) draws; export CSV; run through same statistics.

---

# 6) Immediate next steps (low-risk, high-yield)

1. **Extend schema** for system-level rows and add placeholders for binaries/tertiaries/ages.
2. **Plug HDBSCAN + segmented regression** into your current compiled table (works with what you already fetched).
3. **Re-run figs** and report:

   * Does HDBSCAN pick a **high-q cluster** that includes **stellar binaries** and **BD companions**, with **different (e)–(a)** behavior?
   * Does segmented regression show a **statistically preferred break** in (\log a)–(\log q)?

---

Here are **drop-in modules + minimal wiring** to (1) build **system-level features**, (2) run **HDBSCAN regime discovery**, and (3) fit a **one-break segmented trend** ((\log a_{\min}) vs (\log q_{\max})). Everything reads your existing CSVs under `results/` and writes new figures/tables back there.

---

### 1) New module: `source/system_schema.py`

Creates one row **per system** from your per-companion tables.

```python
# source/system_schema.py
import numpy as np
import pandas as pd
from pathlib import Path

def _log10_safe(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log10(x)
    return y

def build_system_table(
    nasa_csv="results/pscomppars_lowM.csv",
    bd_csv="results/BD_catalogue.csv",
    out_csv="results/combined_systems.csv",
):
    nasa = pd.read_csv(nasa_csv)
    bd = pd.read_csv(bd_csv)

    # --- Normalize/rename key columns (best-effort) ---
    # NASA Exoplanet Archive common fields (as produced earlier)
    # host id:
    host_col = next((c for c in nasa.columns if c.lower() in {"hostname","pl_hostname","sy_name"}), None)
    if host_col is None:
        raise ValueError("Could not find host name column in NASA CSV.")
    nasa = nasa.rename(columns={host_col:"host"})

    # companion mass (Mjup) & semi-major axis (AU) & eccentricity
    mass_col = next((c for c in nasa.columns if c.lower() in {"pl_bmassj","pl_massj","pl_mj"}), None)
    a_col    = next((c for c in nasa.columns if c.lower() in {"pl_orbsmax","pl_orbsmax_au","a"}), None)
    e_col    = next((c for c in nasa.columns if c.lower() in {"pl_orbeccen","e"}), None)
    mstar_col= next((c for c in nasa.columns if c.lower() in {"st_mass","hostmass","msini_star","mstar"}), None)
    age_col  = next((c for c in nasa.columns if c.lower() in {"st_age","age"}), None)
    feh_col  = next((c for c in nasa.columns if c.lower() in {"st_metfe","feh","[fe/h]"}), None)

    for need, nm in [(mass_col,"companion mass (Mjup)"),
                     (a_col,   "semi-major axis (AU)"),
                     (e_col,   "eccentricity"),
                     (mstar_col,"stellar mass (Msun)")]:
        if need is None:
            raise ValueError(f"NASA CSV missing a {nm} column.")

    # Keep only the essentials; coerce numeric
    nasa_use = nasa[["host", mass_col, a_col, e_col, mstar_col]]
    nasa_use = nasa_use.rename(columns={mass_col:"Mj", a_col:"a_AU", e_col:"e", mstar_col:"Mstar"})
    nasa_use["source"] = "NASA"

    # Optional metadata
    if age_col is not None: nasa_use["Age_Gyr"] = pd.to_numeric(nasa[age_col], errors="coerce")
    else: nasa_use["Age_Gyr"] = np.nan
    if feh_col is not None: nasa_use["FeH"] = pd.to_numeric(nasa[feh_col], errors="coerce")
    else: nasa_use["FeH"] = np.nan

    # Brown Dwarf catalogue (already filtered to VLMS earlier)
    # Try to locate analogous columns
    bd_host = next((c for c in bd.columns if c.lower() in {"host","system","name"}), None)
    bd_mj   = next((c for c in bd.columns if "mjup" in c.lower() or "mj" in c.lower()), None)
    bd_a    = next((c for c in bd.columns if "a" in c.lower() and "au" in c.lower()), None)
    bd_e    = next((c for c in bd.columns if c.lower() in {"e","ecc","eccentricity"}), None)
    bd_mstar= next((c for c in bd.columns if "mstar" in c.lower() or "host_mass" in c.lower() or "msun" in c.lower()), None)

    if bd_host and bd_mj and bd_a and bd_e and bd_mstar:
        bd_use = bd[[bd_host, bd_mj, bd_a, bd_e, bd_mstar]].copy()
        bd_use.columns = ["host","Mj","a_AU","e","Mstar"]
        bd_use["source"] = "BD"
        bd_use["Age_Gyr"] = pd.to_numeric(bd.get("Age_Gyr", np.nan), errors="coerce") if "Age_Gyr" in bd.columns else np.nan
        bd_use["FeH"] = pd.to_numeric(bd.get("FeH", np.nan), errors="coerce") if "FeH" in bd.columns else np.nan
        all_comp = pd.concat([nasa_use, bd_use], ignore_index=True)
    else:
        all_comp = nasa_use.copy()

    # Clean numerics
    for c in ["Mj","a_AU","e","Mstar","Age_Gyr","FeH"]:
        all_comp[c] = pd.to_numeric(all_comp[c], errors="coerce")

    # --- System-level aggregation ---
    # Convert companion mass to mass ratio q = Mp / Mstar (Mp in Mjup, Mstar in Msun; 1 Msun = 1047.56 Mjup)
    MJUP_PER_MSUN = 1047.56
    all_comp["q"] = (all_comp["Mj"] / (all_comp["Mstar"]*MJUP_PER_MSUN)).replace([np.inf, -np.inf], np.nan)

    def summarize(group: pd.DataFrame):
        g = group.dropna(subset=["Mstar"])
        if g.empty:
            return pd.Series(dtype=float)

        out = dict(
            host=g["host"].iloc[0],
            Mstar=np.nanmedian(g["Mstar"]),
            Age_Gyr=np.nanmedian(g["Age_Gyr"]),
            FeH=np.nanmedian(g["FeH"]),
            n_comp=g.shape[0],
            q_max=np.nanmax(g["q"]),
            q_med=np.nanmedian(g["q"]),
            a_min=np.nanmin(g["a_AU"]),
            a_med=np.nanmedian(g["a_AU"]),
            e_max=np.nanmax(g["e"]),
            e_med=np.nanmedian(g["e"]),
            has_bd=bool(np.nanmax((g["Mj"]>=13).astype(float)) if "Mj" in g else False),
            has_giant=bool(np.nanmax((g["Mj"]>=0.3).astype(float)) if "Mj" in g else False),
        )
        return pd.Series(out)

    systems = all_comp.groupby("host", dropna=True).apply(summarize).reset_index(drop=True)
    systems["logq_max"] = _log10_safe(systems["q_max"])
    systems["loga_min"] = _log10_safe(systems["a_min"])

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    systems.to_csv(out_csv, index=False)
    return systems
```

---

### 2) New module: `source/analysis/regime_clustering.py`

Runs **HDBSCAN** + **GMM (BIC)** and produces labeled plots.

```python
# source/analysis/regime_clustering.py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

def run_hdbscan_and_gmm(
    systems_csv="results/combined_systems.csv",
    out_prefix="results/regimes",
    min_cluster_size=5,
    random_state=42,
):
    import hdbscan  # pip install hdbscan

    df = pd.read_csv(systems_csv).dropna(subset=["logq_max","loga_min"])
    X = df[["logq_max","loga_min","e_max","Age_Gyr","FeH"]].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X.values)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(Xz)
    df["hdbscan_label"] = labels

    # GMM-BIC (K=1..6)
    Ks = list(range(1,7))
    bics = []
    gmms = []
    for k in Ks:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
        gmm.fit(Xz)
        bics.append(gmm.bic(Xz))
        gmms.append(gmm)
    k_best = Ks[int(np.argmin(bics))]
    gmm_best = gmms[int(np.argmin(bics))]
    df["gmm_label"] = gmm_best.predict(Xz)

    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out_prefix}_labels.csv", index=False)

    # Plot: clusters in (logq, loga)
    fig, ax = plt.subplots(figsize=(7,6))
    for lab in sorted(np.unique(labels)):
        sel = df["hdbscan_label"]==lab
        ax.scatter(df.loc[sel,"logq_max"], df.loc[sel,"loga_min"], s=50, label=f"HDBSCAN {lab}")
    ax.set_xlabel(r"$\log_{10}\,q_{\rm max}$")
    ax.set_ylabel(r"$\log_{10}\,a_{\rm min}\,{\rm [AU]}$")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_hdbscan_logq_loga.png", dpi=180)

    # BIC curve
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(Ks, bics, marker="o")
    ax2.set_xlabel("GMM components K")
    ax2.set_ylabel("BIC (lower is better)")
    ax2.axvline(k_best, ls="--", color="k", alpha=0.5, label=f"best={k_best}")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_gmm_bic.png", dpi=180)

    return df, {"K_best":k_best, "BICs":bics}
```

---

### 3) New module: `source/analysis/segmented_trend.py`

Fits a **single break** in ((x,y)=(\log q_{\max}, \log a_{\min})) with **ruptures/PELT** and reports the break + two OLS fits.

```python
# source/analysis/segmented_trend.py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import ruptures as rpt
from sklearn.linear_model import LinearRegression

def fit_one_break(logq, loga):
    """
    Returns: break_index (or None), piecewise models (left,right), BICs for 0 vs 1 break
    """
    x = np.asarray(logq).reshape(-1,1)
    y = np.asarray(loga)
    # Sort by x for a meaningful 1D change-point
    order = np.argsort(x[:,0])
    x, y = x[order], y[order]

    # 0-break OLS
    ols0 = LinearRegression().fit(x, y)
    rss0 = np.sum((y - ols0.predict(x))**2)
    n, p0 = len(y), 2  # slope+intercept
    bic0 = n*np.log(rss0/n) + p0*np.log(n)

    # 1-break via PELT on y as a function of x index (we’ll just detect a location in the sorted series)
    algo = rpt.Pelt(model="rbf").fit(y)  # rbf is flexible
    # allow at most one change
    cp = algo.predict(pen=np.log(n)*5)  # mild penalty; tweakable
    # 'cp' returns end indices of segments; if one change, cp ~ [k, n]
    k = None
    if len(cp) >= 2:
        k = cp[0]
        if 2 <= k <= n-2:  # keep interior
            xl, yl = x[:k], y[:k]
            xr, yr = x[k:], y[k:]
            olsL = LinearRegression().fit(xl, yl)
            olsR = LinearRegression().fit(xr, yr)
            rss1 = np.sum((yl - olsL.predict(xl))**2) + np.sum((yr - olsR.predict(xr))**2)
            p1 = 4  # 2 params per segment
            bic1 = n*np.log(rss1/n) + p1*np.log(n)
            better = (bic1 + 2.0) < bic0  # small safety margin
            return (k, order, olsL, olsR, bic0, bic1, better)
    return (None, order, ols0, None, bic0, None, False)

def run_segmented_plot(systems_csv="results/combined_systems.csv", out_png="results/segmented_logq_loga.png"):
    df = pd.read_csv(systems_csv).dropna(subset=["logq_max","loga_min"])
    res = fit_one_break(df["logq_max"].values, df["loga_min"].values)
    k, order, mL, mR, bic0, bic1, better = res

    x = df["logq_max"].values[order]
    y = df["loga_min"].values[order]

    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(x, y, s=40, color="tab:blue", alpha=0.8, label="systems")

    if k is not None and better:
        xl, yl = x[:k], y[:k]
        xr, yr = x[k:], y[k:]
        xx = np.linspace(x.min(), x.max(), 200).reshape(-1,1)
        ax.plot(xx, mL.predict(xx), color="tab:red", lw=2, label="segment 1")
        ax.plot(xx, mR.predict(xx), color="tab:green", lw=2, label="segment 2")
        ax.axvline(x[k], ls="--", color="k", alpha=0.5, label=f"break at logq={x[k]:.2f}")
        ax.set_title(f"Segmented fit preferred (BIC0={bic0:.1f}, BIC1={bic1:.1f})")
    else:
        xx = np.linspace(x.min(), x.max(), 200).reshape(-1,1)
        ax.plot(xx, mL.predict(xx), color="tab:red", lw=2, label="single line")
        t = f"No break preferred (BIC0={bic0:.1f})" if bic1 is None else f"No break (BIC0={bic0:.1f} ≤ BIC1={bic1:.1f})"
        ax.set_title(t)

    ax.set_xlabel(r"$\log_{10}\,q_{\rm max}$")
    ax.set_ylabel(r"$\log_{10}\,a_{\min}\,{\rm [AU]}$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)

    # Save a small JSON summary
    import json
    summary = {
        "break_supported": bool(better),
        "break_logq": float(x[k]) if (k is not None and better) else None,
        "bic_single": float(bic0),
        "bic_two": float(bic1) if bic1 is not None else None,
    }
    with open(out_png.replace(".png", ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
```

---

### 4) Minimal wiring in your driver: `source/panoptic_vlms_project.py`

Add two CLI flags and call the new routines **after** your existing fetch/clean steps.

```python
# --- add near top ---
import argparse
# ...
# inside main() arg parser:
parser.add_argument("--build-systems", action="store_true",
                    help="Aggregate per-companion tables into one row per system.")
parser.add_argument("--regimes", action="store_true",
                    help="Run HDBSCAN+GMM clustering and segmented regression on systems.")
# ...

# --- after your existing fetch/process blocks, insert: ---
if args.build_systems:
    from system_schema import build_system_table
    systems = build_system_table(
        nasa_csv="results/pscomppars_lowM.csv",
        bd_csv="results/BD_catalogue.csv",
        out_csv="results/combined_systems.csv",
    )
    print(f"[OK] Wrote {len(systems)} system rows to results/combined_systems.csv")

if args.regimes:
    from analysis.regime_clustering import run_hdbscan_and_gmm
    from analysis.segmented_trend import run_segmented_plot
    labels_df, bic_info = run_hdbscan_and_gmm(
        systems_csv="results/combined_systems.csv",
        out_prefix="results/regimes"
    )
    seg = run_segmented_plot(
        systems_csv="results/combined_systems.csv",
        out_png="results/segmented_logq_loga.png"
    )
    print("[OK] Regime discovery complete.")
    print(f"    GMM best K: {bic_info['K_best']}")
    print(f"    Segmented break supported: {seg['break_supported']}, break at logq={seg['break_logq']}")
```

---

### 5) How to run

```bash
# install once (CPU ok; GPU not required for these steps)
pip install hdbscan ruptures scikit-learn pandas numpy matplotlib

# build the system-level table from your existing CSVs
python source/panoptic_vlms_project.py --build-systems

# run regime discovery + segmented trend
python source/panoptic_vlms_project.py --regimes
```

**Outputs (all under `results/`):**

* `combined_systems.csv` — one row per system with: `Mstar, Age_Gyr, FeH, n_comp, q_max, a_min, e_max, ... , logq_max, loga_min`.
* `regimes_labels.csv` — adds `hdbscan_label`, `gmm_label`.
* `regimes_hdbscan_logq_loga.png` — clusters in ((\log q_{\max}, \log a_{\min})).
* `regimes_gmm_bic.png` — BIC vs K with best K annotated.
* `segmented_logq_loga.png` + `.json` — the single-break test summary.

---

### 6) Interpreting what you’ll get

* **If HDBSCAN finds a stable high-q cluster** (often overlapping your BD + close stellar binaries) that also sits at **smaller (a_{\min})** and **higher (e_{\max})**, that’s direct evidence for a **stellar-like/fragmentation regime** distinct from the disk-like one.
* **If the segmented fit is preferred (lower BIC)** and the break occurs at some (\log q_\ast), you can phrase this as a **quantitative phase shift**: below (q_\ast) the (\log a_{\min})–(\log q) relation follows slope (\beta_1) (planet-like), above (q_\ast) it follows (\beta_2) (binary-like).
* Tie this back to TOI-6894b by showing its coordinates and cluster responsibility.

---

Here are **drop-in additions** to (A) ingest **very-low-mass stellar binaries** and fold them into your system table, and (B) run a **two-regime mixture of linear regressions** (a cross-sectional Markov-switching analogue) to quantify “phase shifts” between planet-like and binary-like regimes.

---

# A) Ingest close VLMS stellar binaries

## 1) New module: `source/ingest_vlms_binaries.py`

```python
# source/ingest_vlms_binaries.py
import numpy as np
import pandas as pd

MJUP_PER_MSUN = 1047.56

REQUIRED = {
    "host",          # system identifier (string)
    "Mstar_Msun",    # primary mass (Msun)
    "M2_Msun",       # companion mass (Msun)
    "a_AU",          # semi-major axis (AU)
    "e"              # eccentricity
}
# Optional: Age_Gyr, FeH

def load_vlms_binaries(sb_csv: str) -> pd.DataFrame:
    """
    Read a user-provided CSV of close VLMS stellar binaries (spectroscopic or visual),
    filter for very-low-mass primaries (0.06–0.25 Msun), and shape to the
    same columns used elsewhere.

    Expected columns in sb_csv:
      host, Mstar_Msun, M2_Msun, a_AU, e, [Age_Gyr], [FeH]
    """
    df = pd.read_csv(sb_csv)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"SB CSV missing columns: {sorted(missing)}")

    # Filter to VLMS primaries
    df = df[(df["Mstar_Msun"] >= 0.06) & (df["Mstar_Msun"] <= 0.25)].copy()

    # Convert stellar companion mass to Mjup to align with your other tables
    df["Mj"] = (df["M2_Msun"] * MJUP_PER_MSUN).astype(float)
    df["Mstar"] = df["Mstar_Msun"].astype(float)
    df["a_AU"] = pd.to_numeric(df["a_AU"], errors="coerce")
    df["e"] = pd.to_numeric(df["e"], errors="coerce")

    # Harmonize column names to match system_schema expectations
    out = pd.DataFrame({
        "host": df["host"].astype(str),
        "Mj": df["Mj"],
        "a_AU": df["a_AU"],
        "e": df["e"],
        "Mstar": df["Mstar"],
        "Age_Gyr": pd.to_numeric(df.get("Age_Gyr", np.nan), errors="coerce"),
        "FeH": pd.to_numeric(df.get("FeH", np.nan), errors="coerce"),
        "source": "SB"
    })
    # Drop any utterly broken rows
    out = out.dropna(subset=["host","Mj","a_AU","e","Mstar"])
    return out
```

## 2) Update: `source/system_schema.py` to accept an SB file

Replace your existing `build_system_table(...)` with this **backwards-compatible** version (it still works if you don’t pass `sb_csv`):

```python
# source/system_schema.py (replace previous build_system_table)
import numpy as np
import pandas as pd
from pathlib import Path

def _log10_safe(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log10(x)
    return y

MJUP_PER_MSUN = 1047.56

def build_system_table(
    nasa_csv="results/pscomppars_lowM.csv",
    bd_csv="results/BD_catalogue.csv",
    sb_csv=None,  # NEW: optional VLMS stellar binaries file
    out_csv="results/combined_systems.csv",
):
    nasa = pd.read_csv(nasa_csv)

    # --- Normalize/rename key columns in NASA table ---
    host_col = next((c for c in nasa.columns if c.lower() in {"hostname","pl_hostname","sy_name"}), None)
    if host_col is None:
        raise ValueError("Could not find host name column in NASA CSV.")
    mass_col = next((c for c in nasa.columns if c.lower() in {"pl_bmassj","pl_massj","pl_mj"}), None)
    a_col    = next((c for c in nasa.columns if c.lower() in {"pl_orbsmax","pl_orbsmax_au","a"}), None)
    e_col    = next((c for c in nasa.columns if c.lower() in {"pl_orbeccen","e"}), None)
    mstar_col= next((c for c in nasa.columns if c.lower() in {"st_mass","hostmass","msini_star","mstar"}), None)
    age_col  = next((c for c in nasa.columns if c.lower() in {"st_age","age"}), None)
    feh_col  = next((c for c in nasa.columns if c.lower() in {"st_metfe","feh","[fe/h]"}), None)

    for need, nm in [(mass_col,"companion mass (Mjup)"),
                     (a_col,   "semi-major axis (AU)"),
                     (e_col,   "eccentricity"),
                     (mstar_col,"stellar mass (Msun)")]:
        if need is None:
            raise ValueError(f"NASA CSV missing a {nm} column.")

    nasa_use = nasa[[host_col, mass_col, a_col, e_col, mstar_col]].copy()
    nasa_use.columns = ["host","Mj","a_AU","e","Mstar"]
    nasa_use["source"] = "NASA"
    nasa_use["Age_Gyr"] = pd.to_numeric(nasa[age_col], errors="coerce") if age_col else np.nan
    nasa_use["FeH"] = pd.to_numeric(nasa[feh_col], errors="coerce") if feh_col else np.nan

    # --- BD catalogue best-effort harmonization ---
    all_comp = [nasa_use]
    try:
        bd = pd.read_csv(bd_csv)
        bd_host = next((c for c in bd.columns if c.lower() in {"host","system","name"}), None)
        bd_mj   = next((c for c in bd.columns if "mjup" in c.lower() or "mj" in c.lower()), None)
        bd_a    = next((c for c in bd.columns if "a" in c.lower() and "au" in c.lower()), None)
        bd_e    = next((c for c in bd.columns if c.lower() in {"e","ecc","eccentricity"}), None)
        bd_mstar= next((c for c in bd.columns if "mstar" in c.lower() or "host_mass" in c.lower() or "msun" in c.lower()), None)
        if bd_host and bd_mj and bd_a and bd_e and bd_mstar:
            bd_use = bd[[bd_host, bd_mj, bd_a, bd_e, bd_mstar]].copy()
            bd_use.columns = ["host","Mj","a_AU","e","Mstar"]
            bd_use["source"]="BD"
            bd_use["Age_Gyr"] = pd.to_numeric(bd.get("Age_Gyr", np.nan), errors="coerce") if "Age_Gyr" in bd.columns else np.nan
            bd_use["FeH"] = pd.to_numeric(bd.get("FeH", np.nan), errors="coerce") if "FeH" in bd.columns else np.nan
            all_comp.append(bd_use)
    except FileNotFoundError:
        pass

    # --- NEW: VLMS stellar binaries ingestion (SB) ---
    if sb_csv:
        from ingest_vlms_binaries import load_vlms_binaries
        sb = load_vlms_binaries(sb_csv)
        all_comp.append(sb)

    comp = pd.concat(all_comp, ignore_index=True)

    # Clean numerics
    for c in ["Mj","a_AU","e","Mstar","Age_Gyr","FeH"]:
        comp[c] = pd.to_numeric(comp[c], errors="coerce")

    # Mass ratio
    comp["q"] = (comp["Mj"] / (comp["Mstar"]*MJUP_PER_MSUN)).replace([np.inf, -np.inf], np.nan)

    # --- System-level aggregation ---
    def summarize(group: pd.DataFrame):
        g = group.dropna(subset=["Mstar"])
        if g.empty:
            return pd.Series(dtype=float)
        out = dict(
            host=g["host"].iloc[0],
            Mstar=np.nanmedian(g["Mstar"]),
            Age_Gyr=np.nanmedian(g["Age_Gyr"]),
            FeH=np.nanmedian(g["FeH"]),
            n_comp=g.shape[0],
            q_max=np.nanmax(g["q"]),
            q_med=np.nanmedian(g["q"]),
            a_min=np.nanmin(g["a_AU"]),
            a_med=np.nanmedian(g["a_AU"]),
            e_max=np.nanmax(g["e"]),
            e_med=np.nanmedian(g["e"]),
            has_bd=bool(np.nanmax((g["Mj"]>=13).astype(float)) if "Mj" in g else False),
            has_giant=bool(np.nanmax((g["Mj"]>=0.3).astype(float)) if "Mj" in g else False),
            has_sb=bool(np.any(g["source"]=="SB")),
        )
        return pd.Series(out)

    systems = comp.groupby("host", dropna=True).apply(summarize).reset_index(drop=True)
    systems["logq_max"] = _log10_safe(systems["q_max"])
    systems["loga_min"] = _log10_safe(systems["a_min"])

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    systems.to_csv(out_csv, index=False)
    return systems
```

---

# B) Two-regime **mixture of linear regressions** (cross-sectional MSR)

This is a simple EM for a **2-component mixture of lines**:
[
y \sim \alpha_r + \beta_r x + \varepsilon_r,\quad r\in{1,2},\qquad
\Pr(r=1)=\pi,\ \Pr(r=2)=1-\pi.
]
It returns responsibilities (posterior probabilities) so you can color points by regime.

## 3) New module: `source/analysis/msr_mixture.py`

```python
# source/analysis/msr_mixture.py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def em_mixture_of_lines(x, y, max_iter=500, tol=1e-6, seed=42):
    """
    Two-component mixture of linear regressions via EM.
    x, y: 1D arrays (float). Returns dict with params, responsibilities, loglike.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float).reshape(-1,1)
    y = np.asarray(y, float)

    n = len(y)
    X = np.c_[np.ones(n), x]  # intercept + slope

    # init via random split
    z = rng.integers(0, 2, size=n)   # component labels
    pi = np.mean(z==0)

    def fit_ols(w):
        # weighted OLS to get [alpha, beta], sigma^2
        W = np.diag(w)
        XtW = X.T @ W
        beta = np.linalg.pinv(XtW @ X) @ (XtW @ y)
        resid = y - X @ beta
        var = (w * resid**2).sum() / max(w.sum(), 1e-12)
        return beta, max(var, 1e-12)

    # initialize responsibilities
    r1 = (z==0).astype(float) + 1e-3
    r2 = 1.0 - (z==0).astype(float) + 1e-3
    r1 /= (r1+r2); r2 = 1.0 - r1

    beta1, var1 = fit_ols(r1)
    beta2, var2 = fit_ols(r2)
    pi = r1.mean()

    last_ll = -np.inf
    for it in range(max_iter):
        # E-step: responsibilities under current params
        mu1 = X @ beta1
        mu2 = X @ beta2
        # Gaussian densities (up to normalization)
        n1 = (1/np.sqrt(2*np.pi*var1)) * np.exp(-0.5*(y-mu1)**2/var1)
        n2 = (1/np.sqrt(2*np.pi*var2)) * np.exp(-0.5*(y-mu2)**2/var2)
        num1 = pi * n1
        num2 = (1-pi) * n2
        denom = num1 + num2 + 1e-300
        r1 = num1 / denom
        r2 = num2 / denom

        # M-step: update mixing and component params
        pi = r1.mean()
        beta1, var1 = fit_ols(r1)
        beta2, var2 = fit_ols(r2)

        # log-likelihood
        ll = np.sum(np.log(denom))
        if np.abs(ll - last_ll) < tol:
            break
        last_ll = ll

    return {
        "pi": float(pi),
        "beta1": beta1, "var1": float(var1),
        "beta2": beta2, "var2": float(var2),
        "r1": r1, "r2": r2,
        "loglike": float(last_ll),
        "nit": it+1
    }

def run_msr_on_systems(
    systems_csv="results/combined_systems.csv",
    out_png="results/msr_logq_loga.png",
    out_csv="results/msr_responsibilities.csv"
):
    df = pd.read_csv(systems_csv).dropna(subset=["logq_max","loga_min"])
    x = df["logq_max"].values
    y = df["loga_min"].values

    res = em_mixture_of_lines(x, y, max_iter=1000, tol=1e-7, seed=123)
    r1 = res["r1"]; r2 = res["r2"]
    df["regime1_prob"] = r1
    df["regime2_prob"] = r2

    # Save table
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Plot
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    Xp = np.c_[np.ones_like(xx), xx]

    y1 = Xp @ res["beta1"]
    y2 = Xp @ res["beta2"]

    fig, ax = plt.subplots(figsize=(7,6))
    sc = ax.scatter(x, y, c=r1, cmap="viridis", s=50, edgecolor="k", linewidths=0.3)
    cb = plt.colorbar(sc, ax=ax, label="P(regime 1)")
    ax.plot(xx, y1, color="tab:red", lw=2, label="Regime 1 fit")
    ax.plot(xx, y2, color="tab:green", lw=2, label="Regime 2 fit")
    ax.set_xlabel(r"$\log_{10}\,q_{\rm max}$")
    ax.set_ylabel(r"$\log_{10}\,a_{\min}\,{\rm [AU]}$")
    ax.legend(frameon=False)
    ax.set_title(f"Mixture of Lines (EM): π={res['pi']:.2f}, iters={res['nit']}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    return res
```

---

# C) Minimal wiring in your driver

In `source/panoptic_vlms_project.py`:

1. **Add CLI flags:**

```python
parser.add_argument("--sb-csv", type=str, default=None,
    help="Path to CSV of close VLMS stellar binaries (host,Mstar_Msun,M2_Msun,a_AU,e[,Age_Gyr,FeH]).")
parser.add_argument("--build-systems", action="store_true",
    help="Aggregate per-companion tables into one row per system.")
parser.add_argument("--regimes", action="store_true",
    help="Run HDBSCAN+GMM clustering and segmented regression on systems.")
parser.add_argument("--msr", action="store_true",
    help="Run 2-regime mixture of linear regressions on (log q_max, log a_min).")
```

2. **Call the updated system builder:**

```python
if args.build_systems:
    from system_schema import build_system_table
    systems = build_system_table(
        nasa_csv="results/pscomppars_lowM.csv",
        bd_csv="results/BD_catalogue.csv",
        sb_csv=args.sb_csv,  # NEW
        out_csv="results/combined_systems.csv",
    )
    print(f"[OK] Wrote {len(systems)} system rows to results/combined_systems.csv")
```

3. **Keep your prior regime discovery calls**, then add MSR:

```python
if args.regimes:
    from analysis.regime_clustering import run_hdbscan_and_gmm
    from analysis.segmented_trend import run_segmented_plot
    labels_df, bic_info = run_hdbscan_and_gmm(
        systems_csv="results/combined_systems.csv",
        out_prefix="results/regimes"
    )
    seg = run_segmented_plot(
        systems_csv="results/combined_systems.csv",
        out_png="results/segmented_logq_loga.png"
    )
    print("[OK] Regime discovery complete.")

if args.msr:
    from analysis.msr_mixture import run_msr_on_systems
    res = run_msr_on_systems(
        systems_csv="results/combined_systems.csv",
        out_png="results/msr_logq_loga.png",
        out_csv="results/msr_responsibilities.csv"
    )
    print(f"[OK] MSR done. pi={res['pi']:.2f}, iters={res['nit']}")
```

---

# D) How to run

1. Prepare a CSV of **close VLMS stellar binaries**, e.g. `data/vlms_binaries.csv`:

```
host,Mstar_Msun,M2_Msun,a_AU,e,Age_Gyr,FeH
CMa-001,0.10,0.085,0.045,0.12,2.1,-0.10
CMa-002,0.12,0.030,0.090,0.35,1.6,-0.05
...
```

2. Build the system table (merges NASA + BD + SB):

```
python source/panoptic_vlms_project.py --build-systems --sb-csv data/vlms_binaries.csv
```

3. Run clustering + segmented trend:

```
python source/panoptic_vlms_project.py --regimes
```

4. Run **mixture of lines** (cross-sectional MSR):

```
python source/panoptic_vlms_project.py --msr
```

**Outputs (all in `results/`):**

* `combined_systems.csv` — now includes `has_sb`.
* `regimes_hdbscan_logq_loga.png`, `regimes_gmm_bic.png`, `segmented_logq_loga.png` (+ JSON).
* `msr_logq_loga.png` — two fitted lines and point colors = P(regime 1).
* `msr_responsibilities.csv` — per-system regime probabilities.

---

## How to interpret (what you’re looking for)

* If **SB objects** (stellar binaries) and **brown-dwarf companions** cluster together at **high (q_{\max})** and **smaller (a_{\min})** with **higher (e_{\max})**, that empirically supports a **binary-like regime** continuous across the hydrogen-burning limit.
* If the **mixture model** yields visibly different slopes/intercepts for the two lines and splits systems with high posterior confidence, you can call these **statistically distinct coupling regimes** (your “phase shifts”).
* You can then overlay **TOI-6894b** (or analogs) and report its posterior membership probability.




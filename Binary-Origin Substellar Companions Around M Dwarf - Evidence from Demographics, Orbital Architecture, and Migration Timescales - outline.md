# Title

**Binary-Origin Substellar Companions Around M Dwarfs: Evidence from Demographics, Orbital Architecture, and Migration Timescales**

# Scope (what this note will *actually* deliver)

A short quantitative paper (≤10–12 pages, 4–5 figures, 2–3 tables, code+data release) that:

1. Assembles a VLMS (0.06–0.20 M⊙) companion set from **real** catalogs.
   NASA Exoplanet Archive TAP: [https://exoplanetarchive.ipac.caltech.edu/TAP](https://exoplanetarchive.ipac.caltech.edu/TAP)
   PS column docs: [https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html)
   Brown Dwarf Companion Catalogue: [https://ordo.open.ac.uk/articles/dataset/Brown_Dwarf_Companion_Catalogue/24156393](https://ordo.open.ac.uk/articles/dataset/Brown_Dwarf_Companion_Catalogue/24156393)
   GitHub mirror: [https://github.com/adam-stevenson/brown-dwarf-desert](https://github.com/adam-stevenson/brown-dwarf-desert)
2. Quantifies **population structure** in ((\log q,\log a)) and **eccentricity architecture** (e(a)).
3. Provides **numerical migration feasibility** via (i) early disk torques and (ii) KL+tides.
4. Publishes a **reusable origin classifier** with cross-validated performance and per-object probabilities.
5. Ships a **fully reproducible package** (CSV + Python scripts + figure generation) consistent with arXiv.

---

# A. Real Companion Demographics (Figure 1 + Table 1)

**Deliverables**

* **Fig. 1 (core):** (\log M_\star) vs. (\log M_c) (VLMS hosts), with lines at 13 MJ (deuterium) and 0.075 M⊙ (hydrogen). Mark TOI-6894b.
* **Fig. 1b (supplement):** Same plot, **colored by detection method** (RV, transit, AO, astrometry) to visualize selection.
* **Table 1:** The stacked catalog (VLMS subset) with: `hostname, pl_name, st_mass, pl_bmassj, q, pl_orbsmax, pl_orbeccen, discoverymethod, st_metfe, source`. (This is the paper’s data backbone.)

**Method notes**

* Compute (q=M_c/M_\star) (with (1,M_\odot=1047.56,M_J)).
* Retain only rows with ({M_\star, M_c, a, e}) measured (upper limits handled in Section B Robustness).
* Provide both **“true mass only”** and **“m sin i included”** panels (the latter flagged), to show results are not an artifact of inclination censoring.

**What this proves (visually)**
Continuity across 13 MJ at low (M_\star), and TOI-6894b sitting on the **low-(q)** tail—consistent with mass-asymmetric fragmentation + truncated accretion, not a hard planet/BD bifurcation.

---

# B. Eccentricity–Separation Structure (Figure 2 + Table 2)

**Deliverables**

* **Fig. 2 (core):** (e) vs. (a) (log (a)), points sized by (q) and colored by method.
* **Fig. 2b (supplement):** Kernel-density contours in (e)–(a) for two subsets split at (q=0.01).
* **Table 2:** Parametric fits and tests:

  * **Two-component Beta model** (e\mid z=k\sim \mathrm{Beta}(\alpha_k,\beta_k)) for low-(q) vs high-(q) subsets, MLE parameters ((\hat\alpha_k,\hat\beta_k)) with bootstrap CIs.
  * **KS test** statistics/p-values for (e) distributions (low-(q) vs high-(q)).
  * **BIC/WAIC** comparison: 1-Beta vs 2-Beta.

**Minimal equations (reported in Methods)**

* Beta-MLE via log-parametrization; KS two-sample statistic (D_{n,m}) with exact p-value.

**Robustness (do *all* of these, they’re cheap)**

* Re-run after **excluding**: (i) transits, (ii) RVs, (iii) imaging—show trend persists.
* Treat (e) **upper limits** by (a) excluding those rows, and (b) EM-like latent-(e) with truncation; report both give the same qualitative split.

**Interpretation you’ll write**
High-(q) companions show a bias to larger (e) at discovery (fragmentation/dynamical history). Compact low-(a) objects with low (e) exhibit damping signatures—consistent with post-birth shrinkage/circularization.

---

# C. Quantitative Migration Plausibility (Figure 3 + short math)

You will give **two independent routes** that can deliver (a\sim0.05) AU within realistic times.

**C1. Early disk torques (Type-I–like scaling)**
[
t_{\rm mig}\sim C,\frac{M_\star}{M_c},\frac{M_\star}{\Sigma a^2}\left(\frac{H}{a}\right)^2\Omega^{-1},\qquad
\Omega=\sqrt{\frac{GM_\star}{a^3}}.
]

* **Nominal M-dwarf values** (to quote): (M_\star=0.08,M_\odot), (H/a\simeq0.03{-}0.05), (\Sigma(1,\mathrm{AU})\sim 10^2{-}10^3,\mathrm{g,cm^{-2}}) with (\Sigma\propto a^{-p},\ p\in[0.5,1.5]); (C=\mathcal{O}(1{-}10)).
* **Deliverable:** A **1-panel band plot** of (t_{\rm mig}(a_0!\to!0.05,\mathrm{AU})) vs. (a_0) across priors (shaded region). Expect many priors to yield (\ll 1)–3 Myr—plausible within disk lifetimes.

**C2. High-(e) phases + tides after KL forcing**

* **Kozai-Lidov timescale** (quadrupole; order-of-magnitude):
  [
  t_{\rm KL} \sim \frac{M_\star + M_c}{M_{\rm out}}\frac{P_{\rm out}^2}{P_{\rm in}}(1-e_{\rm out}^2)^{3/2}.
  ]
* **Tidal shrink (circular-orbit scaling for intuition)**:
  [
  t_a \approx \frac{2Q'*\star}{9},\frac{M*\star}{M_c}\left(\frac{a}{R_\star}\right)^5 \frac{1}{n},\quad n=\sqrt{\frac{GM_\star}{a^3}}.
  ]
  At the **current** (a=0.05) AU and (Q'_\star\sim10^{6\text{–}7}), (t_a) is **far too long** (≫ Gyr)—so circularization requires **brief high-(e)** epochs with tiny periastron (r_p=a(1-e)), or else it happened earlier in a gas-rich phase.

**Deliverables**

* **Fig. 3 (core):** A **KL+tide feasibility heat-map** over ((M_{\rm out},a_{\rm out})) showing the **fraction** of random draws that meet:
  (t_{\rm KL}\le T) (e.g., 1 Gyr) **and** (r_p\le r_{\rm crit}) (e.g., (3{-}5,R_\star)) at peak cycles.
  (Your script already computes this; you’ll report one clean number: the fraction of the map with feasibility ≥ X%.)
* **One worked example** (in text): choose (M_{\rm out}=0.05,M_\odot, a_{\rm out}=50,\mathrm{AU}, e_{\rm out}=0.3), show (t_{\rm KL}\sim 10^{6\text{–}7}) yr and that (e_{\max}\gtrsim0.9) implies (r_p\lesssim5R_\star), enabling circularization to (a\approx0.05) AU within ≤ Gyr.

**Robustness**

* Repeat the map for (T={1,3,5}) Gyr and (r_{\rm crit}/R_\star={3,5,7}); include as **Fig. 3 supplement** (3×3 mini-panels or a small table of coverage fractions).

---

# D. Minimal, Testable Origin Classifier (Figure 4 + Table 3)

**Model**
Regularized logistic:
[
P(\text{binary-origin}\mid x)=\sigma!\big(w_0+w^\top x\big),\
x=\big(\log q,\ \log a,\ e,\ \log M_\star,\ [\mathrm{Fe/H}],\ \text{method dummies}\big).
]

**Labels for training (anchors only)**

* **Binary-like anchors:** wide imaged BDs/super-Jovians with clear stellar-multiplicity context.
* **Planet-like anchors:** sub-Neptune/Neptune-mass transiting planets around M dwarfs with canonical disk signatures.
  (Your current heuristic split (q>0.01) vs (q\le0.01) is the *fallback*; keep it but also report a tiny anchor-set fit.)

**Deliverables**

* **Fig. 4 (core):** Calibration plot and ROC; report 5-fold AUROC with CI (bootstrap).
* **Table 3:** Coefficients (with bootstrap CIs), and **TOI-6894b’s (P_{\rm binary})** with uncertainty (via coefficient bootstrap).
* **Release the classifier** (one JSON with coefficients + feature scaler) so others can apply it to new systems.

**Robustness**

* Re-fit excluding each detection method in turn.
* Check calibration (Brier score, reliability curve).
* Sensitivity to metallicity missingness (mean impute vs. indicator-for-missing).

---

# E. Selection-Effect Controls (short, but *essential*)

Add a half-page describing:

* **Detection-method stratification** (already done in plots and re-fits).
* **Inclination censoring** (true-mass subset vs. (m\sin i)).
* **Heterogeneous error bars** (weight-free main fits; mention that heteroscedastic EM/logistic versions give similar partitions; provide as supplement).

---

# F. Reproducibility Package (arXiv-ready)

* **Data:** `vlms_companions_stacked.csv` + the two raw inputs (or download script).
* **Code:** `panoptic_vlms_project.py` to regenerate all tables/figures from scratch.
* **Artifacts:** `fig1_massmass.png`, `fig2_ae.png`, `fig3_feasibility.png`, `objects_with_probs.csv`, `gmm_summary.json`, `beta_e_params.csv`, `ks_test_e.txt`, `logistic_cv_auc.txt`, `feasibility_map.npz`, `SUMMARY.txt`.
* **README:** exact commands, Python/env versions, and notes on randomness seeds used for the KL map and bootstraps.

---

# G. What goes in the paper (section-by-section checklist)

1. **Introduction:** The taxonomy vs. origin point; why VLMS companions are a stress-test for core accretion; statement of contributions (data, mixture, architecture, feasibility map, classifier).
2. **Data & Methods:** Sources + cuts; definitions of (q,a,e); mixture model, Beta fits, KS; KL+tide and disk-torque scalings; classifier features and CV.
3. **Results:** Fig. 1–4 + Tables 1–3; the **three key numbers**:

   * BIC(2-comp) − BIC(1-comp) in ((\log q,\log a)) (expect ΔBIC > 10: “very strong”).
   * KS (e)-distribution result (stat+p).
   * Fraction of the KL map feasible at (T=1) Gyr, and TOI-6894b’s (P_{\rm binary}).
4. **Discussion:** Binary-like origin as a natural pathway; how this reframes “formation puzzle” claims.
5. **Data/Code availability:** raw URLs above + repo/Zenodo DOI.
6. **Limitations:** Selection biases, eccentricity uncertainties, simplicity of the KL criterion; future work (Gaia DR3 NSS outer perturbers: [https://www.cosmos.esa.int/web/gaia/dr3-non-single-stars](https://www.cosmos.esa.int/web/gaia/dr3-non-single-stars)).


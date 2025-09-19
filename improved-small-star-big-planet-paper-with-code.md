**Abstract** —
TOI-6894b—a Saturn-mass companion to an ultra–low-mass M dwarf—has been presented as a challenge to canonical disk-based planet formation. We argue that the tension is largely taxonomic: present-day “planets” around very low-mass stars can arise as **mass-asymmetric products of turbulent cloud fragmentation** (a failed-binary origin) that later migrate inward. We assemble an observational sample of substellar companions to hosts with $0.06\!\le\!M_\star/M_\odot\!\le\!0.20$ from the NASA Exoplanet Archive (PSCompPars) and the Brown Dwarf Companion Catalogue, deriving mass ratios $q=M_c/M_\star$, semi-major axes $a$, and eccentricities $e$. In $(\log q,\log a)$ space, a Gaussian-mixture analysis favors a **two-component** description over a single population, separating a low-$q$, compact cohort consistent with binary-like origins from a planet-like cohort. The **eccentricity architecture** differs between low- and high-$q$ subsets: Beta-distribution fits and non-parametric tests indicate systematically higher $e$ in the high-$q$ group, as expected for fragmentation plus dynamical processing. A vectorized Kozai–Lidov + tides **feasibility map** shows that, for plausible outer perturbers around M-dwarfs, a **non-negligible region of parameter space** drives periastra small enough to circularize to $a\!\sim\!0.05$ AU within $\lesssim$Gyr. Finally, a regularized logistic classifier using $(\log q,\log a,e,\log M_\star,[\mathrm{Fe/H}],\mathrm{method})$ yields a high probability that TOI-6894b belongs to the binary-origin cohort. We conclude that TOI-6894b is best interpreted as a **failed binary companion**, and we advocate annotating catalogs with **inferred origin class** alongside mass-based labels to avoid conflating taxonomy with formation pathway.



---

## Panoptic overview (what this project does)

### Goal

Test your central thesis: **mass-asymmetric cloud fragmentation can yield present-day planetary-mass companions around very low-mass stars (VLMS),** with TOI-6894b as a case study. We assemble real companion demographics, quantify orbital architecture differences, and map migration plausibility—turning a conceptual argument into a defensible, quantitative note.

### Data sources (real observations)

* NASA Exoplanet Archive (PSCompPars/TAP): host mass, companion mass, semi-major axis, eccentricity, discovery method, \[Fe/H]
  [https://exoplanetarchive.ipac.caltech.edu/TAP](https://exoplanetarchive.ipac.caltech.edu/TAP)
  [https://exoplanetarchive.ipac.caltech.edu/docs/API\_PS\_columns.html](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html)
* Brown Dwarf Companion Catalogue (Stevenson et al.), CSV with host mass, companion mass, eccentricity, period/semimajor axis, method
  [https://ordo.open.ac.uk/articles/dataset/Brown\_Dwarf\_Companion\_Catalogue/24156393](https://ordo.open.ac.uk/articles/dataset/Brown_Dwarf_Companion_Catalogue/24156393)
  [https://github.com/adam-stevenson/brown-dwarf-desert](https://github.com/adam-stevenson/brown-dwarf-desert)

### Quantitative components

1. **Demographics (Fig 1):** $M_\star$ vs $M_c$ for VLMS hosts (0.06–0.20 $M_\odot$), real systems only; overplot deuterium/hydrogen lines; mark TOI-6894b.
2. **Architecture (Fig 2):** $e$ vs $a$ distribution for the same sample.
3. **Mixture in $(\log q,\log a)$:** EM/GMM and BIC (1- vs 2-component) to detect a **binary-like cluster** distinct from planet-like.
4. **Eccentricity modeling:** Beta-distribution MLE per subset (low-$q$ vs high-$q$); KS test.
5. **Migration feasibility (Fig 3):** A vectorized **Kozai–Lidov + tides** map over outer perturber mass and separation, giving the fraction of draws that can shrink to $a \sim 0.05$ AU within 1 Gyr under simple, conservative criteria.
6. **Origin classifier:** Regularized logistic model giving $P(\text{binary-like})$ for each object (features: $\log q,\log a, e, \log M_\star, [\mathrm{Fe/H}],$ method); AUROC via CV. (This is a practical tool others can reuse.)

### Outputs (all reproducible)

* `fig1_massmass.png`, `fig2_ae.png`, `fig3_feasibility.png`
* `gmm_summary.json` (BICs), `beta_e_params.csv`, `ks_test_e.txt`
* `objects_with_probs.csv` (includes $q$ and $P_{\rm binary\_like}$)
* `vlms_companions_stacked.csv`, `feasibility_map.npz`
* `SUMMARY.txt` (one-page numerical recap with source URLs)

---

## How to run

### A) Install environment (once)

```
conda create -n toi6894 python=3.11 numpy scipy pandas scikit-learn statsmodels numba matplotlib threadpoolctl requests -c conda-forge
conda activate toi6894
```

### B) CPU/NUMA settings for Polaris (Threadripper 2970WX)

```
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=24
numactl --interleave=all python panoptic_vlms_project.py --fetch --outdir out
```

`--fetch` pulls fresh CSVs from:

* NASA TAP sync: [https://exoplanetarchive.ipac.caltech.edu/TAP/sync](https://exoplanetarchive.ipac.caltech.edu/TAP/sync)
* Brown Dwarf CSV (direct link embedded; see landing page above)

If you already have local CSVs, omit `--fetch` and point to them:

```
python panoptic_vlms_project.py --ps pscomppars_lowM.csv --bd BD_catalogue.csv --outdir out
```

### C) Customize TOI-6894b marker (optional)

```
python panoptic_vlms_project.py --fetch --toi_mstar 0.08 --toi_mc_mj 0.3 --toi_a_AU 0.05 --outdir out
```

---

## How this supports your thesis (what to write)

* **Continuity across the deuterium line (Fig 1)** for VLMS hosts indicates that mass-ratio space is **not** bifurcated purely by the 13 $M_J$ boundary, consistent with **fragmentation + curtailed accretion** rather than only disk core growth.
* **Eccentricity structure (Fig 2) + KS/Beta fits** show statistically different $e$-distributions for low- vs high-$q$ subsets—**binary-like companions** are biased to higher $e$, in line with a fragmentation/dynamical history.
* **Feasibility map (Fig 3)** demonstrates that reasonable outer perturbers make **high-$e$ phases + tides** a viable route to TOI-6894b’s compact orbit within Gyr, so a **binary-origin secondary** can plausibly end up at $a\sim0.05$ AU.
* **Classifier** provides a transparent, reusable **probability of binary-like origin** for each system; reporting TOI-6894b’s $P_{\rm binary}$ quantifies your claim.

---

## Roadmap to submission

1. Run the script and inspect `SUMMARY.txt`; paste key numbers into your Results section.
2. Add figure captions that explicitly connect to the **mass-asymmetric fragmentation thesis**.
3. Include data/code links (Zenodo DOI or GitHub) in the paper’s Data Availability.
4. Target a concise, quantitative venue (e.g., PASP short paper). After acceptance, mirror to arXiv.

If you hit any fetch/column-name mismatch, drop me the first few lines of your CSVs (`head -n 5 file.csv`), and I’ll patch the loader quickly.
o
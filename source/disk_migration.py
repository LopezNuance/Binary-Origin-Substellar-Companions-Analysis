"""
Disk migration timescale calculations for Type-I migration
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Physical constants
AU = 1.495978707e13  # cm
M_sun = 1.98847e33   # g
M_jup = 1.89813e30   # g
G = 6.67430e-8       # cgs
YEAR = 3.15576e7     # s


def _Omega(Mstar_Msun, a_AU):
    """Calculate orbital frequency"""
    M = Mstar_Msun * M_sun
    a = a_AU * AU
    return np.sqrt(G * M / a**3)


def typeI_timescale_sec(Mstar_Msun, Mp_Mj, a_AU, Sigma1_gcm2, p_sigma, H_over_a, C=3.0):
    """
    Calculate Type-I migration timescale using Tanaka et al. formalism

    Parameters:
    -----------
    Mstar_Msun : float
        Stellar mass in solar masses
    Mp_Mj : float
        Planet mass in Jupiter masses
    a_AU : float
        Semimajor axis in AU
    Sigma1_gcm2 : float
        Surface density at 1 AU in g/cm^2
    p_sigma : float
        Surface density power law index
    H_over_a : float
        Disk aspect ratio
    C : float
        Calibration constant (default 3.0)

    Returns:
    --------
    float : Migration timescale in seconds
    """
    Mstar = Mstar_Msun * M_sun
    Mp = Mp_Mj * M_jup
    a = a_AU * AU
    Sigma = Sigma1_gcm2 * (a_AU ** (-p_sigma))
    Omega = _Omega(Mstar_Msun, a_AU)

    return C * (Mstar/Mp) * (Mstar/(Sigma * a*a)) * (H_over_a**2) / Omega


def migrate_time_numeric(Mstar_Msun, Mp_Mj, a0_AU, af_AU,
                        Sigma1_gcm2, p_sigma, H_over_a, C=3.0, nstep=2000):
    """
    Numerically integrate migration time from a0 to af

    Returns:
    --------
    float : Total migration time in seconds
    """
    a_hi, a_lo = max(a0_AU, af_AU), min(a0_AU, af_AU)
    a_grid = np.geomspace(a_hi, a_lo, nstep)

    vals = []
    for a in a_grid:
        t_local = typeI_timescale_sec(Mstar_Msun, Mp_Mj, a,
                                     Sigma1_gcm2, p_sigma, H_over_a, C)
        vals.append(t_local / (a * AU))

    dt_sec = np.trapz(vals, a_grid)
    return dt_sec


def render_disk_panel(outpath_png, Mstar_Msun, Mp_Mj, args):
    """
    Create heatmap of migration timescales

    Parameters:
    -----------
    outpath_png : str
        Output path for figure
    Mstar_Msun : float
        Host star mass
    Mp_Mj : float
        Companion mass in Jupiter masses
    args : namespace
        Command-line arguments containing disk parameters
    """
    a0s = np.geomspace(args.a0_min, args.a0_max, 60)
    Sigma1_list = np.geomspace(args.Sigma1AU/5.0, args.Sigma1AU*5.0, 60)
    Z = np.zeros((len(Sigma1_list), len(a0s)))

    af = 0.05  # Final position in AU

    for i, S1 in enumerate(Sigma1_list):
        for j, a0 in enumerate(a0s):
            t_sec = migrate_time_numeric(Mstar_Msun, Mp_Mj, a0, af,
                                       S1, args.p_sigma, args.H_over_a, C=3.0)
            Z[i, j] = t_sec / (1e6 * YEAR)  # Convert to Myr

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4), constrained_layout=True)

    im = ax.pcolormesh(a0s, Sigma1_list, Z, shading='auto',
                      norm=mpl.colors.LogNorm(vmin=0.1, vmax=100))
    cb = fig.colorbar(im, ax=ax, label="Migration time (Myr) to 0.05 AU")

    # Add disk lifetime contour
    cs = ax.contour(a0s, Sigma1_list, Z, levels=[args.disk_lifetime_myr],
                   colors='white', linewidths=1.5)
    ax.clabel(cs, fmt={args.disk_lifetime_myr: f"{args.disk_lifetime_myr:.0f} Myr"},
             inline=True)

    ax.axhline(args.Sigma1AU, ls='--', lw=1, color='k', alpha=0.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Birth $a_0$ (AU)")
    ax.set_ylabel(r"$\Sigma_{1\,\mathrm{AU}}$ (g cm$^{-2}$)")
    ax.set_title("Disk Migration: Is $a_0\\rightarrow0.05$ AU feasible?")

    # Add parameter box
    ax.text(0.03, 0.02,
           f"H/a={args.H_over_a:.2f}, p={args.p_sigma:.1f}, α={args.alpha:.0e}\n"
           f"M*={Mstar_Msun:.2f} M$_\\odot$, M$_c$={Mp_Mj:.2f} M$_J$",
           transform=ax.transAxes, fontsize=9, va='bottom')

    fig.savefig(outpath_png, dpi=200)
    plt.close(fig)

    return Z
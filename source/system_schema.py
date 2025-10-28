"""
System-level data aggregation for multi-companion analysis
"""
import numpy as np
import pandas as pd
from pathlib import Path

MJUP_PER_MSUN = 1047.56


def _log10_safe(x):
    """Safe log10 that handles zeros and negatives"""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log10(x)
    return y


def build_system_table(nasa_csv="results/pscomppars_lowM.csv",
                      bd_csv="results/BD_catalogue.csv",
                      sb_csv=None,
                      out_csv="results/combined_systems.csv"):
    """
    Aggregate companion data into system-level rows

    Parameters:
    -----------
    nasa_csv : str
        Path to NASA Exoplanet Archive data
    bd_csv : str
        Path to Brown Dwarf catalog
    sb_csv : str, optional
        Path to stellar binary catalog
    out_csv : str
        Output path for combined system table

    Returns:
    --------
    pd.DataFrame : System-level aggregated data
    """
    # Load NASA data
    nasa = pd.read_csv(nasa_csv)

    # Normalize column names (handle variations)
    host_col = next((c for c in nasa.columns
                    if c.lower() in {"hostname", "pl_hostname", "sy_name"}), None)
    if host_col is None:
        raise ValueError("Could not find host name column in NASA CSV")

    # Map columns to standard names
    nasa = nasa.rename(columns={host_col: "host"})

    # Process companion data
    all_comp = [nasa]

    # Add brown dwarf data if available
    try:
        bd = pd.read_csv(bd_csv)
        all_comp.append(bd)
    except FileNotFoundError:
        pass

    # Add stellar binaries if provided
    if sb_csv:
        from source.ingest_vlms_binaries import load_vlms_binaries
        sb = load_vlms_binaries(sb_csv)
        all_comp.append(sb)

    # Combine all sources
    comp = pd.concat(all_comp, ignore_index=True)

    # System-level aggregation function
    def summarize(group):
        """Aggregate companion properties per system"""
        if group.empty:
            return pd.Series(dtype=float)

        # Calculate mass ratios
        if 'Mstar' in group.columns and 'Mj' in group.columns:
            q = group['Mj'] / (group['Mstar'] * MJUP_PER_MSUN)
        else:
            q = pd.Series([np.nan])

        out = {
            'host': group['host'].iloc[0],
            'Mstar': np.nanmedian(group.get('Mstar', np.nan)),
            'n_comp': len(group),
            'q_max': np.nanmax(q),
            'q_med': np.nanmedian(q),
            'a_min': np.nanmin(group.get('a_AU', np.nan)),
            'a_med': np.nanmedian(group.get('a_AU', np.nan)),
            'e_max': np.nanmax(group.get('e', np.nan)),
            'e_med': np.nanmedian(group.get('e', np.nan)),
            'has_bd': bool(np.any(group.get('Mj', 0) >= 13)),
            'has_giant': bool(np.any(group.get('Mj', 0) >= 0.3)),
            'has_sb': bool('SB' in group.get('source', []))
        }
        return pd.Series(out)

    # Group by host and aggregate
    systems = comp.groupby('host', dropna=True).apply(summarize).reset_index(drop=True)

    # Add logarithmic features
    systems['logq_max'] = _log10_safe(systems['q_max'])
    systems['loga_min'] = _log10_safe(systems['a_min'])

    # Save results
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    systems.to_csv(out_csv, index=False)

    return systems
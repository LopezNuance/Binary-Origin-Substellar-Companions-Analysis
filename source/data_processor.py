import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import warnings


JUPITER_TO_EARTH = 317.828

class VLMSDataProcessor:
    """Process and combine data from NASA and Brown Dwarf catalogues for VLMS analysis"""

    def __init__(self, min_stellar_mass: float = 0.06, max_stellar_mass: float = 0.20):
        self.min_stellar_mass = min_stellar_mass
        self.max_stellar_mass = max_stellar_mass

    def process_nasa_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process NASA Exoplanet Archive data

        Parameters:
        -----------
        df : pd.DataFrame
            Raw NASA data

        Returns:
        --------
        pd.DataFrame with standardized columns
        """

        processed = df.copy()

        # Convert Jupiter masses to Earth masses if needed
        if 'pl_massj' in processed.columns and processed['pl_massj'].notna().any():
            # Fill missing Earth masses with Jupiter mass conversion when available
            mask = processed['pl_masse'].isna() & processed['pl_massj'].notna()
            processed.loc[mask, 'pl_masse'] = processed.loc[mask, 'pl_massj'] * JUPITER_TO_EARTH

        # Create standardized column names
        column_mapping = {
            'pl_name': 'companion_name',
            'hostname': 'host_name',
            'st_mass': 'host_mass_msun',
            'st_age': 'host_age_gyr',
            'pl_masse': 'companion_mass_mearth',
            'pl_massj': 'companion_mass_mjup',
            'pl_orbsmax': 'semimajor_axis_au',
            'pl_orbeccen': 'eccentricity',
            'discoverymethod': 'discovery_method',
            'st_met': 'metallicity',
            'st_rad': 'host_radius_rsun',
            'st_teff': 'host_teff_k'
        }

        # Rename columns
        for old_col, new_col in column_mapping.items():
            if old_col in processed.columns:
                processed[new_col] = processed[old_col]

        # Derive missing mass columns
        if 'companion_mass_mearth' in processed.columns and 'companion_mass_mjup' not in processed.columns:
            processed['companion_mass_mjup'] = processed['companion_mass_mearth'] / JUPITER_TO_EARTH
        if 'companion_mass_mjup' in processed.columns and 'companion_mass_mearth' not in processed.columns:
            processed['companion_mass_mearth'] = processed['companion_mass_mjup'] * JUPITER_TO_EARTH

        if 'host_age_gyr' in processed.columns:
            processed['host_age_gyr'] = pd.to_numeric(processed['host_age_gyr'], errors='coerce')

        # Filter by stellar mass
        if 'host_mass_msun' in processed.columns:
            mask = ((processed['host_mass_msun'] >= self.min_stellar_mass) &
                   (processed['host_mass_msun'] <= self.max_stellar_mass))
            processed = processed[mask].copy()

        # Add source label
        processed['data_source'] = 'NASA'

        print(f"Processed {len(processed)} NASA entries for VLMS hosts")
        return processed

    def process_bd_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Brown Dwarf Catalogue data

        Parameters:
        -----------
        df : pd.DataFrame
            Raw Brown Dwarf catalogue data

        Returns:
        --------
        pd.DataFrame with standardized columns
        """

        processed = df.copy()

        # Try to identify column names (catalogue format may vary)
        possible_mappings = {
            'host_mass_msun': ['M_star', 'stellar_mass', 'host_mass', 'M_host', 'Mstar'],
            'host_age_gyr': ['age', 'Age', 'stellar_age', 'host_age', 'Age_Gyr', 'Age_Gyrs'],
            'companion_mass_mearth': ['M_comp', 'companion_mass', 'M_companion', 'mass_comp'],
            'companion_mass_mjup': ['M_comp_mjup', 'M_comp_mj', 'companion_mass_mjup', 'mass_comp_mj'],
            'semimajor_axis_au': ['a_au', 'semimajor_axis', 'sma', 'a'],
            'eccentricity': ['e', 'ecc', 'eccentricity'],
            'period_days': ['P_days', 'period', 'period_days'],
            'discovery_method': ['method', 'discovery_method', 'detection_method']
        }

        # Map columns
        for std_col, possible_cols in possible_mappings.items():
            for col in possible_cols:
                if col in processed.columns:
                    processed[std_col] = processed[col]
                    break

        # Convert period to semimajor axis if needed (Kepler's 3rd law)
        if ('period_days' in processed.columns and
            'semimajor_axis_au' not in processed.columns and
            'host_mass_msun' in processed.columns):

            # a^3 = (G*M*P^2)/(4*pi^2), simplified for solar units
            P_years = processed['period_days'] / 365.25
            M_total = processed['host_mass_msun']  # Assume M_comp << M_star
            processed['semimajor_axis_au'] = (M_total * P_years**2)**(1/3)

        # Convert companion mass to Earth masses if in Jupiter masses
        if 'companion_mass_mearth' in processed.columns:
            # Check if values suggest Jupiter masses (< 20 would be unusual for Earth masses)
            if processed['companion_mass_mearth'].median() < 20:
                processed['companion_mass_mearth'] *= JUPITER_TO_EARTH

        # Derive missing mass columns
        if 'companion_mass_mearth' in processed.columns and 'companion_mass_mjup' not in processed.columns:
            processed['companion_mass_mjup'] = processed['companion_mass_mearth'] / JUPITER_TO_EARTH
        if 'companion_mass_mjup' in processed.columns and 'companion_mass_mearth' not in processed.columns:
            processed['companion_mass_mearth'] = processed['companion_mass_mjup'] * JUPITER_TO_EARTH

        if 'host_age_gyr' in processed.columns:
            processed['host_age_gyr'] = pd.to_numeric(processed['host_age_gyr'], errors='coerce')
            processed.loc[processed['host_age_gyr'] <= 0, 'host_age_gyr'] = np.nan

        # Filter by stellar mass
        if 'host_mass_msun' in processed.columns:
            mask = ((processed['host_mass_msun'] >= self.min_stellar_mass) &
                   (processed['host_mass_msun'] <= self.max_stellar_mass))
            processed = processed[mask].copy()

        # Add source label
        processed['data_source'] = 'BD_Catalogue'

        print(f"Processed {len(processed)} Brown Dwarf entries for VLMS hosts")
        return processed

    def combine_datasets(self, nasa_df: pd.DataFrame, bd_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine NASA and Brown Dwarf datasets

        Parameters:
        -----------
        nasa_df : pd.DataFrame
            Processed NASA data
        bd_df : pd.DataFrame
            Processed Brown Dwarf data

        Returns:
        --------
        pd.DataFrame combined dataset
        """

        # Define required columns for analysis (require Jupiter-mass column)
        required_cols = [
            'host_mass_msun',
            'companion_mass_mjup',
            'semimajor_axis_au',
            'eccentricity',
        ]

        # Filter datasets to have required columns
        nasa_clean = nasa_df.dropna(subset=required_cols).copy()
        bd_clean = bd_df.dropna(subset=required_cols).copy()

        # Derive Earth masses where needed for downstream tables/exports
        for _df in (nasa_clean, bd_clean):
            if 'companion_mass_mearth' not in _df.columns:
                _df['companion_mass_mearth'] = _df['companion_mass_mjup'] * JUPITER_TO_EARTH
            else:
                missing_mask = _df['companion_mass_mearth'].isna()
                if missing_mask.any():
                    _df.loc[missing_mask, 'companion_mass_mearth'] = (
                        _df.loc[missing_mask, 'companion_mass_mjup'] * JUPITER_TO_EARTH
                    )

        # Combine while preserving the full union of feature columns so downstream
        # analyses (e.g. metallicity, discovery method) remain available even when
        # supplied by only one catalogue.
        combined = pd.concat([nasa_clean, bd_clean], ignore_index=True, sort=False)

        # Move the core analysis columns to the front for readability
        primary_cols = required_cols + [col for col in combined.columns if col not in required_cols]
        combined = combined[primary_cols]

        print(f"Combined dataset: {len(combined)} total entries")
        print(f"  NASA: {len(nasa_clean)} entries")
        print(f"  Brown Dwarf: {len(bd_clean)} entries")

        return combined

    def compute_derived_quantities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived quantities for analysis

        Parameters:
        -----------
        df : pd.DataFrame
            Combined dataset

        Returns:
        --------
        pd.DataFrame with additional columns
        """

        result = df.copy()

        # Ensure both mass representations are available
        if 'companion_mass_mjup' not in result.columns:
            if 'companion_mass_mearth' in result.columns:
                result['companion_mass_mjup'] = result['companion_mass_mearth'] / JUPITER_TO_EARTH
            else:
                raise KeyError("companion_mass_mjup column is required for derived quantities")

        if 'companion_mass_mearth' not in result.columns:
            result['companion_mass_mearth'] = result['companion_mass_mjup'] * JUPITER_TO_EARTH

        # Convert companion mass to solar masses
        result['companion_mass_msun'] = result['companion_mass_mjup'] / 1047.6

        # Compute mass ratio q = M_companion / M_star
        result['mass_ratio'] = result['companion_mass_msun'] / result['host_mass_msun']

        # Log quantities for analysis
        result['log_mass_ratio'] = np.log10(result['mass_ratio'])
        result['log_semimajor_axis'] = np.log10(result['semimajor_axis_au'])
        result['log_host_mass'] = np.log10(result['host_mass_msun'])

        # Determine if above/below deuterium burning limit (13 MJ ≈ 0.0124 Msun)
        deuterium_limit_mjup = 13.0
        result['above_deuterium_limit'] = result['companion_mass_mjup'] >= deuterium_limit_mjup

        # Set default eccentricity to 0 if missing (many assume circular orbits)
        if 'eccentricity' not in result.columns:
            result['eccentricity'] = 0.0
        else:
            result['eccentricity'] = result['eccentricity'].fillna(0.0)

        # Classify by mass ratio (arbitrary threshold for "high-q" vs "low-q")
        q_threshold = 0.01  # 1% mass ratio
        result['high_mass_ratio'] = result['mass_ratio'] >= q_threshold

        print(f"Computed derived quantities for {len(result)} objects")
        print(f"  Above deuterium limit: {result['above_deuterium_limit'].sum()}")
        print(f"  High mass ratio (q >= {q_threshold}): {result['high_mass_ratio'].sum()}")

        return result

    def annotate_age_relative_to_toi(self, df: pd.DataFrame, toi_age_gyr: float | None) -> pd.DataFrame:
        """Annotate dataset with age comparisons to TOI-6894b host."""

        if toi_age_gyr is None or np.isnan(toi_age_gyr):
            return df

        result = df.copy()
        if 'host_age_gyr' not in result.columns:
            result['host_age_gyr'] = np.nan

        result['age_delta_vs_toi_gyr'] = result['host_age_gyr'] - toi_age_gyr
        result['is_younger_than_toi'] = result['age_delta_vs_toi_gyr'] < 0

        return result

    def add_toi6894b(self, df: pd.DataFrame, toi_mstar: float = 0.08,
                    toi_mc_mj: float = 0.3, toi_a_au: float = 0.05,
                    toi_ecc: float = 0.0, toi_age_gyr: float | None = None) -> pd.DataFrame:
        """
        Add TOI-6894b to the dataset

        Parameters:
        -----------
        df : pd.DataFrame
            Existing dataset
        toi_mstar : float
            TOI-6894 stellar mass (solar masses)
        toi_mc_mj : float
            TOI-6894b companion mass (Jupiter masses)
        toi_a_au : float
            TOI-6894b semi-major axis (AU)
        toi_ecc : float
            TOI-6894b eccentricity

        Returns:
        --------
        pd.DataFrame with TOI-6894b added
        """

        result = df.copy()

        # Create TOI-6894b entry
        toi_entry = {
            'companion_name': 'TOI-6894b',
            'host_name': 'TOI-6894',
            'host_mass_msun': toi_mstar,
            'companion_mass_mearth': toi_mc_mj * 317.8,
            'companion_mass_mjup': toi_mc_mj,
            'companion_mass_msun': toi_mc_mj / 1047.6,
            'semimajor_axis_au': toi_a_au,
            'eccentricity': toi_ecc,
            'host_age_gyr': toi_age_gyr,
            'discovery_method': 'Transit',
            'data_source': 'TOI',
            'mass_ratio': (toi_mc_mj / 1047.6) / toi_mstar,
            'log_mass_ratio': np.log10((toi_mc_mj / 1047.6) / toi_mstar),
            'log_semimajor_axis': np.log10(toi_a_au),
            'log_host_mass': np.log10(toi_mstar),
            'above_deuterium_limit': toi_mc_mj >= 13.0,
            'high_mass_ratio': ((toi_mc_mj / 1047.6) / toi_mstar) >= 0.01
        }

        # Add any missing columns with NaN
        for col in result.columns:
            if col not in toi_entry:
                toi_entry[col] = np.nan

        # Append to dataframe
        result = pd.concat([result, pd.DataFrame([toi_entry])], ignore_index=True)

        print(f"Added TOI-6894b to dataset (M_star={toi_mstar}, M_c={toi_mc_mj} MJ, a={toi_a_au} AU)")
        return result

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to CSV"""
        df.to_csv(filename, index=False)
        print(f"Saved processed data to {filename}")


if __name__ == "__main__":
    # Test the processor
    processor = VLMSDataProcessor()

    # This would normally load real data
    # For testing, create dummy data
    dummy_nasa = pd.DataFrame({
        'st_mass': [0.1, 0.15, 0.08],
        'pl_masse': [100, 500, 95],
        'pl_orbsmax': [0.1, 0.05, 0.03],
        'pl_orbeccen': [0.1, 0.0, 0.2],
        'pl_name': ['Test1b', 'Test2b', 'Test3b'],
        'hostname': ['Test1', 'Test2', 'Test3'],
        'discoverymethod': ['Transit', 'RV', 'Transit']
    })

    dummy_bd = pd.DataFrame({
        'M_star': [0.12, 0.18],
        'M_comp': [800, 1200],  # Earth masses
        'a_au': [0.08, 0.12],
        'e': [0.3, 0.1]
    })

    # Process data
    nasa_processed = processor.process_nasa_data(dummy_nasa)
    bd_processed = processor.process_bd_data(dummy_bd)
    combined = processor.combine_datasets(nasa_processed, bd_processed)
    final = processor.compute_derived_quantities(combined)
    final_with_toi = processor.add_toi6894b(final)

    print("\nFinal dataset columns:")
    print(final_with_toi.columns.tolist())
    print(f"\nFinal dataset shape: {final_with_toi.shape}")

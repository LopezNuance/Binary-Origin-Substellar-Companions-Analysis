import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import time
import io

class NASAExoplanetArchiveFetcher:
    """Fetch data from NASA Exoplanet Archive TAP service"""

    def __init__(self, base_url: str = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"):
        self.base_url = base_url

    def fetch_vlms_companions(self, min_stellar_mass: float = 0.06,
                             max_stellar_mass: float = 0.20) -> pd.DataFrame:
        """
        Fetch companions to very low mass stars from PSCompPars table

        Parameters:
        -----------
        min_stellar_mass : float
            Minimum stellar mass in solar masses
        max_stellar_mass : float
            Maximum stellar mass in solar masses

        Returns:
        --------
        pd.DataFrame with columns: pl_name, hostname, st_mass, st_age, pl_masse,
                                   pl_orbsmax, pl_orbeccen, discoverymethod, st_met,
                                   pl_massj, st_rad, st_teff
        """
        query = f"""
        SELECT pl_name, hostname, st_mass, st_age, pl_masse, pl_orbsmax, pl_orbeccen,
               discoverymethod, st_met, pl_massj, st_rad, st_teff
        FROM pscomppars
        WHERE st_mass >= {min_stellar_mass}
        AND st_mass <= {max_stellar_mass}
        AND pl_masse IS NOT NULL
        AND pl_orbsmax IS NOT NULL
        AND st_mass IS NOT NULL
        """  # nosec - safe controlled float formatting

        params = {
            'REQUEST': 'doQuery',
            'LANG': 'ADQL',
            'FORMAT': 'csv',
            'QUERY': query
        }

        print(f"Fetching NASA Exoplanet Archive data for VLMS hosts ({min_stellar_mass}-{max_stellar_mass} M_sun)...")

        try:
            start = time.time()
            response = requests.get(self.base_url, params=params, timeout=60)
            response.raise_for_status()
            duration = time.time() - start
            print(f"Fetch completed in {duration:.2f} seconds.")

            # Parse CSV data
            df = pd.read_csv(io.StringIO(response.text))
            print(f"Retrieved {len(df)} entries from NASA Exoplanet Archive")

            df = self.postprocess_data(df)
            return df

        except requests.RequestException as e:
            print(f"Error fetching NASA data: {e}")
            raise

    def postprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and filter the raw dataframe

        Removes:
            - Non-physical or extreme values
            - Entries with NaNs in critical columns

        Returns:
            Cleaned pd.DataFrame
        """
        # Remove rows with clearly invalid or missing temperature
        if 'st_teff' in df.columns:
            df = df[df['st_teff'].between(2000, 4000, inclusive='both')]

        # Replace extreme metallicity outliers with NaN
        if 'st_met' in df.columns:
            df.loc[(df['st_met'] < -2.5) | (df['st_met'] > 0.7), 'st_met'] = np.nan

        if 'st_age' in df.columns:
            df.loc[df['st_age'] <= 0, 'st_age'] = np.nan

        # Drop rows with any remaining nulls in important fields
        df = df.dropna(subset=['pl_name', 'hostname', 'pl_orbsmax', 'st_mass'])

        print(f"Postprocessed to {len(df)} rows after cleaning")
        return df

    def save_data(self, df: pd.DataFrame, filename: str):
        df.to_csv(filename, index=False)
        print(f"Saved NASA data to {filename}")


class BrownDwarfCatalogueFetcher:
    """Fetch Brown Dwarf Companion Catalogue data"""

    def __init__(self, csv_url: str = "https://ordo.open.ac.uk/ndownloader/articles/24156393/versions/1"):
        self.csv_url = csv_url
        self.github_url = "https://raw.githubusercontent.com/adam-stevenson/brown-dwarf-desert/main/BD_catalogue.csv"

    def fetch_catalogue(self) -> pd.DataFrame:
        """
        Fetch Brown Dwarf Companion Catalogue

        Returns:
        --------
        pd.DataFrame with brown dwarf companion data
        """

        print("Fetching Brown Dwarf Companion Catalogue...")

        # Try GitHub first (more reliable direct CSV access)
        urls_to_try = [self.github_url, self.csv_url]

        for url in urls_to_try:
            try:
                print(f"Trying URL: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()

                # Try to parse as CSV
                df = pd.read_csv(io.StringIO(response.text))
                print(f"Retrieved {len(df)} entries from Brown Dwarf Catalogue")

                df = self.postprocess_data(df)
                return df

            except Exception as e:
                print(f"Failed to fetch from {url}: {e}")
                continue

        raise Exception("Could not fetch Brown Dwarf Catalogue from any URL")

    def postprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and harmonize data for downstream use
        """
        if 'orbital_period' in df.columns:
            df['orbital_period'] = df['orbital_period'].replace(0, np.nan)

        return df

    def filter_vlms_hosts(self, df: pd.DataFrame,
                         min_stellar_mass: float = 0.06,
                         max_stellar_mass: float = 0.20) -> pd.DataFrame:
        """Filter catalogue for VLMS hosts"""

        # Assume stellar mass column might be named differently
        mass_cols = ['M_star', 'stellar_mass', 'host_mass', 'M_host', 'Mstar']
        mass_col = next((col for col in mass_cols if col in df.columns), None)

        if mass_col is None:
            print("Warning: Could not identify stellar mass column.")
            print(f"Available columns: {list(df.columns)}")
            return df

        # Filter by stellar mass
        mask = (df[mass_col] >= min_stellar_mass) & (df[mass_col] <= max_stellar_mass)
        filtered_df = df[mask].copy()

        print(f"Filtered to {len(filtered_df)} VLMS hosts from Brown Dwarf Catalogue")
        return filtered_df

    def save_data(self, df: pd.DataFrame, filename: str):
        """Save dataframe to CSV"""
        df.to_csv(filename, index=False)
        print(f"Saved Brown Dwarf Catalogue to {filename}")


def load_local_data(nasa_file: Optional[str] = None,
                   bd_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load data from local CSV files

    Parameters:
    -----------
    nasa_file : str, optional
        Path to NASA Exoplanet Archive CSV file
    bd_file : str, optional
        Path to Brown Dwarf Catalogue CSV file

    Returns:
        dict with keys 'nasa_df', 'bd_df'
    """
    nasa_df, bd_df = pd.DataFrame(), pd.DataFrame()

    if nasa_file:
        try:
            nasa_df = pd.read_csv(nasa_file)
            print(f"Loaded {len(nasa_df)} entries from local NASA file: {nasa_file}")
        except Exception as e:
            print(f"Error loading NASA file {nasa_file}: {e}")

    if bd_file:
        try:
            bd_df = pd.read_csv(bd_file)
            print(f"Loaded {len(bd_df)} entries from local BD file: {bd_file}")
        except Exception as e:
            print(f"Error loading BD file {bd_file}: {e}")

    return {"nasa_df": nasa_df, "bd_df": bd_df}


if __name__ == "__main__":
    # Example standalone usage
    nasa_fetcher = NASAExoplanetArchiveFetcher()
    bd_fetcher = BrownDwarfCatalogueFetcher()

    # Fetch NASA data
    nasa_data = nasa_fetcher.fetch_vlms_companions()
    nasa_fetcher.save_data(nasa_data, "pscomppars_lowM.csv")

    # Fetch Brown Dwarf data
    bd_data = bd_fetcher.fetch_catalogue()
    bd_filtered = bd_fetcher.filter_vlms_hosts(bd_data)
    bd_fetcher.save_data(bd_filtered, "BD_catalogue.csv")

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
        pd.DataFrame with columns: pl_name, hostname, st_mass, pl_masse,
                                  pl_orbsmax, pl_orbeccen, discoverymethod, st_met
        """

        query = f"""
        SELECT pl_name, hostname, st_mass, pl_masse, pl_orbsmax, pl_orbeccen,
               discoverymethod, st_met, pl_massj, st_rad, st_teff
        FROM pscomppars
        WHERE st_mass >= {min_stellar_mass}
        AND st_mass <= {max_stellar_mass}
        AND pl_masse IS NOT NULL
        AND pl_orbsmax IS NOT NULL
        AND st_mass IS NOT NULL
        """

        params = {
            'REQUEST': 'doQuery',
            'LANG': 'ADQL',
            'FORMAT': 'csv',
            'QUERY': query
        }

        print(f"Fetching NASA Exoplanet Archive data for VLMS hosts ({min_stellar_mass}-{max_stellar_mass} M_sun)...")

        try:
            response = requests.get(self.base_url, params=params, timeout=60)
            response.raise_for_status()

            # Parse CSV data
            df = pd.read_csv(io.StringIO(response.text))

            print(f"Retrieved {len(df)} entries from NASA Exoplanet Archive")
            return df

        except requests.RequestException as e:
            print(f"Error fetching NASA data: {e}")
            raise

    def save_data(self, df: pd.DataFrame, filename: str):
        """Save dataframe to CSV"""
        df.to_csv(filename, index=False)
        print(f"Saved NASA data to {filename}")


class BrownDwarfCatalogueFetcher:
    """Fetch Brown Dwarf Companion Catalogue data"""

    def __init__(self, csv_url: str = "https://ordo.open.ac.uk/ndownloader/articles/24156393/versions/1"):
        self.csv_url = csv_url
        # Alternative direct GitHub URL if available
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
                return df

            except Exception as e:
                print(f"Failed to fetch from {url}: {e}")
                continue

        raise Exception("Could not fetch Brown Dwarf Catalogue from any URL")

    def filter_vlms_hosts(self, df: pd.DataFrame, min_stellar_mass: float = 0.06,
                         max_stellar_mass: float = 0.20) -> pd.DataFrame:
        """Filter catalogue for VLMS hosts"""

        # Assume stellar mass column might be named differently
        mass_cols = ['M_star', 'stellar_mass', 'host_mass', 'M_host', 'Mstar']
        mass_col = None

        for col in mass_cols:
            if col in df.columns:
                mass_col = col
                break

        if mass_col is None:
            print("Warning: Could not identify stellar mass column in Brown Dwarf Catalogue")
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
                   bd_file: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from local CSV files

    Parameters:
    -----------
    nasa_file : str, optional
        Path to NASA Exoplanet Archive CSV file
    bd_file : str, optional
        Path to Brown Dwarf Catalogue CSV file

    Returns:
    --------
    tuple of (nasa_df, bd_df)
    """

    nasa_df = pd.DataFrame()
    bd_df = pd.DataFrame()

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

    return nasa_df, bd_df


if __name__ == "__main__":
    # Test the fetchers
    nasa_fetcher = NASAExoplanetArchiveFetcher()
    bd_fetcher = BrownDwarfCatalogueFetcher()

    # Fetch NASA data
    nasa_data = nasa_fetcher.fetch_vlms_companions()
    nasa_fetcher.save_data(nasa_data, "pscomppars_lowM.csv")

    # Fetch Brown Dwarf data
    bd_data = bd_fetcher.fetch_catalogue()
    bd_filtered = bd_fetcher.filter_vlms_hosts(bd_data)
    bd_fetcher.save_data(bd_filtered, "BD_catalogue.csv")
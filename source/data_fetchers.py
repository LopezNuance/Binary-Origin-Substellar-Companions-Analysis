import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import time
import io
from astropy.coordinates import SkyCoord
from astropy import units as u
from astrometric_orbits import GaiaNSSOrbitFitter

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


class GaiaDR3NSSFetcher:
    """Fetch Gaia DR3 Non-Single Stars (NSS) data for outer perturber analysis"""

    def __init__(self, base_url: str = "https://gea.esac.esa.int/tap-server/tap/sync"):
        self.base_url = base_url
        self.orbit_fitter = GaiaNSSOrbitFitter(base_url)

    def fetch_nss_companions(self, ra_min: float = None, ra_max: float = None,
                            dec_min: float = None, dec_max: float = None,
                            parallax_min: float = 0.1) -> pd.DataFrame:
        """
        Fetch NSS (Non-Single Star) data from Gaia DR3

        Parameters:
        -----------
        ra_min, ra_max : float, optional
            RA bounds in degrees
        dec_min, dec_max : float, optional
            Dec bounds in degrees
        parallax_min : float
            Minimum parallax in mas (default 0.1 for nearby stars)

        Returns:
        --------
        pd.DataFrame with Gaia NSS data including astrometric companions
        """

        # Build spatial constraints if provided
        spatial_constraints = []
        if ra_min is not None and ra_max is not None:
            spatial_constraints.append(f"ra BETWEEN {ra_min} AND {ra_max}")
        if dec_min is not None and dec_max is not None:
            spatial_constraints.append(f"dec BETWEEN {dec_min} AND {dec_max}")

        spatial_clause = " AND " + " AND ".join(spatial_constraints) if spatial_constraints else ""

        query = f"""
        SELECT
            source_id, ra, dec, parallax, parallax_error,
            pmra, pmra_error, pmdec, pmdec_error,
            phot_g_mean_mag, bp_rp,
            nss_solution_type, nss_two_body_orbits_flag,
            mass_flame, mass_flame_lower, mass_flame_upper,
            age_flame, age_flame_lower, age_flame_upper,
            teff_gspphot, teff_gspphot_lower, teff_gspphot_upper,
            logg_gspphot, logg_gspphot_lower, logg_gspphot_upper,
            mh_gspphot, mh_gspphot_lower, mh_gspphot_upper
        FROM gaiadr3.gaia_source_lite
        WHERE parallax > {parallax_min}
        AND nss_solution_type IS NOT NULL
        AND nss_two_body_orbits_flag = 1
        {spatial_clause}
        """

        params = {
            'REQUEST': 'doQuery',
            'LANG': 'ADQL',
            'FORMAT': 'csv',
            'QUERY': query.strip()
        }

        print(f"Fetching Gaia DR3 NSS data (parallax > {parallax_min} mas)...")

        try:
            start = time.time()
            response = requests.get(self.base_url, params=params, timeout=180)
            response.raise_for_status()
            duration = time.time() - start
            print(f"Gaia fetch completed in {duration:.2f} seconds.")

            df = pd.read_csv(io.StringIO(response.text))
            print(f"Retrieved {len(df)} Gaia NSS entries")

            df = self.postprocess_nss_data(df)
            return df

        except requests.RequestException as e:
            print(f"Error fetching Gaia NSS data: {e}")
            raise

    def postprocess_nss_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and derive additional parameters for NSS data
        """
        processed = df.copy()

        # Convert parallax to distance
        processed['distance_pc'] = 1000.0 / processed['parallax']  # pc

        # Estimate stellar mass from Gaia photometry (rough main sequence relation)
        # M/M_sun ≈ (L/L_sun)^0.25 for main sequence, using G mag as proxy
        if 'phot_g_mean_mag' in processed.columns:
            # Rough absolute G magnitude (ignoring extinction)
            processed['abs_g_mag'] = processed['phot_g_mean_mag'] - 5*np.log10(processed['distance_pc']) + 5
            # Very rough mass estimate (better to use mass_flame when available)
            processed['mass_phot_estimate'] = np.power(10, (4.83 - processed['abs_g_mag'])/10) ** 0.25

        # Use FLAME mass estimates when available, fallback to photometric
        if 'mass_flame' in processed.columns:
            processed['stellar_mass_msun'] = processed['mass_flame'].fillna(processed.get('mass_phot_estimate', np.nan))
        else:
            processed['stellar_mass_msun'] = processed.get('mass_phot_estimate', np.nan)

        # Clean up obvious outliers
        processed = processed[processed['parallax'] > 0]
        processed = processed[processed['distance_pc'] < 1000]  # Within 1 kpc

        # Filter for potential VLMS hosts if we have mass estimates
        if 'stellar_mass_msun' in processed.columns:
            vlms_mask = (processed['stellar_mass_msun'] >= 0.06) & (processed['stellar_mass_msun'] <= 0.20)
            processed = processed[vlms_mask | processed['stellar_mass_msun'].isna()]

        print(f"Postprocessed to {len(processed)} Gaia NSS entries after cleaning")
        return processed

    def fetch_enhanced_nss_orbits(self, source_ids: list = None,
                                 ra_min: float = None, ra_max: float = None,
                                 dec_min: float = None, dec_max: float = None,
                                 primary_masses: pd.Series = None) -> pd.DataFrame:
        """
        Fetch enhanced NSS orbital solutions with proper astrometric fitting

        Parameters:
        -----------
        source_ids : list, optional
            List of Gaia source_ids to query
        ra_min, ra_max : float, optional
            RA bounds in degrees
        dec_min, dec_max : float, optional
            Dec bounds in degrees
        primary_masses : pd.Series, optional
            Primary star masses indexed by source_id

        Returns:
        --------
        pd.DataFrame with detailed orbital parameters and uncertainties
        """

        print("Fetching enhanced NSS orbital solutions with astrometric fitting...")

        try:
            # Fetch raw orbital data from nss_two_body_orbit table
            nss_orbital_data = self.orbit_fitter.fetch_nss_orbital_solutions(
                source_ids=source_ids,
                ra_min=ra_min, ra_max=ra_max,
                dec_min=dec_min, dec_max=dec_max
            )

            if nss_orbital_data.empty:
                print("No NSS orbital solutions found")
                return pd.DataFrame()

            # Process with proper orbital parameter conversion
            enhanced_orbits = self.orbit_fitter.process_nss_orbital_catalog(
                nss_orbital_data, primary_masses
            )

            print(f"Enhanced orbital fitting completed for {len(enhanced_orbits)} systems")

            return enhanced_orbits

        except Exception as e:
            print(f"Error in enhanced NSS orbit fetching: {e}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception details: {str(e)}")
            if hasattr(e, '__traceback__'):
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
            print("Falling back to basic NSS data")
            return pd.DataFrame()

    def save_data(self, df: pd.DataFrame, filename: str):
        """Save Gaia NSS data to CSV"""
        df.to_csv(filename, index=False)
        print(f"Saved Gaia NSS data to {filename}")

    def save_enhanced_orbits(self, df: pd.DataFrame, filename: str):
        """Save enhanced orbital solutions to CSV"""
        self.orbit_fitter.save_enhanced_orbits(df, filename)


def cross_match_coordinates(df1: pd.DataFrame, df2: pd.DataFrame,
                          ra1_col: str = 'ra', dec1_col: str = 'dec',
                          ra2_col: str = 'ra', dec2_col: str = 'dec',
                          max_sep_arcsec: float = 5.0) -> pd.DataFrame:
    """
    Cross-match two catalogs based on sky coordinates

    Parameters:
    -----------
    df1, df2 : pd.DataFrame
        Input catalogs to cross-match
    ra1_col, dec1_col : str
        RA/Dec column names in df1
    ra2_col, dec2_col : str
        RA/Dec column names in df2
    max_sep_arcsec : float
        Maximum separation for a match in arcseconds

    Returns:
    --------
    pd.DataFrame with matches from both catalogs
    """

    # Create SkyCoord objects
    coords1 = SkyCoord(ra=df1[ra1_col].values * u.degree,
                      dec=df1[dec1_col].values * u.degree,
                      frame='icrs')

    coords2 = SkyCoord(ra=df2[ra2_col].values * u.degree,
                      dec=df2[dec2_col].values * u.degree,
                      frame='icrs')

    # Perform cross-match
    idx2, sep, _ = coords1.match_to_catalog_sky(coords2)

    # Filter by separation threshold
    good_matches = sep < max_sep_arcsec * u.arcsec

    if not good_matches.any():
        print(f"No cross-matches found within {max_sep_arcsec} arcsec")
        return pd.DataFrame()

    # Combine matched rows
    df1_matched = df1[good_matches].copy()
    df2_matched = df2.iloc[idx2[good_matches]].copy()

    # Add suffix to avoid column name conflicts
    df1_matched = df1_matched.add_suffix('_primary')
    df2_matched = df2_matched.add_suffix('_gaia')

    # Concatenate horizontally
    matched = pd.concat([df1_matched.reset_index(drop=True),
                        df2_matched.reset_index(drop=True)], axis=1)

    # Add separation column
    matched['separation_arcsec'] = sep[good_matches].to(u.arcsec).value

    print(f"Found {len(matched)} cross-matches within {max_sep_arcsec} arcsec")

    return matched


if __name__ == "__main__":
    # Example standalone usage
    nasa_fetcher = NASAExoplanetArchiveFetcher()
    bd_fetcher = BrownDwarfCatalogueFetcher()
    gaia_fetcher = GaiaDR3NSSFetcher()

    # Fetch NASA data
    nasa_data = nasa_fetcher.fetch_vlms_companions()
    nasa_fetcher.save_data(nasa_data, "pscomppars_lowM.csv")

    # Fetch Brown Dwarf data
    bd_data = bd_fetcher.fetch_catalogue()
    bd_filtered = bd_fetcher.filter_vlms_hosts(bd_data)
    bd_fetcher.save_data(bd_filtered, "BD_catalogue.csv")

    # Fetch Gaia NSS data
    gaia_data = gaia_fetcher.fetch_nss_companions()
    gaia_fetcher.save_data(gaia_data, "gaia_nss_vlms.csv")

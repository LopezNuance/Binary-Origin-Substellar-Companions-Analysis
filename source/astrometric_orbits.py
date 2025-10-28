"""
Astrometric orbit fitting and parameter conversion for Gaia DR3 NSS data

This module provides functions to:
1. Fetch detailed orbital parameters from Gaia DR3 nss_two_body_orbit table
2. Convert Thiele-Innes elements to standard orbital elements
3. Derive accurate outer perturber orbital solutions for KL analysis
4. Handle uncertainty propagation in orbital parameter calculations

Author: Enhanced VLMS analysis pipeline
"""

import numpy as np
import pandas as pd
import requests
import io
import warnings
from typing import Tuple, Optional, Dict, Any
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy import constants as const


class GaiaNSSOrbitFitter:
    """Enhanced Gaia NSS orbit fitting with proper astrometric solutions"""

    def __init__(self, base_url: str = "https://gea.esac.esa.int/tap-server/tap/sync"):
        self.base_url = base_url

    def fetch_nss_orbital_solutions(self, source_ids: list = None,
                                   ra_min: float = None, ra_max: float = None,
                                   dec_min: float = None, dec_max: float = None) -> pd.DataFrame:
        """
        Fetch detailed orbital solutions from Gaia DR3 nss_two_body_orbit table

        Parameters:
        -----------
        source_ids : list, optional
            List of Gaia source_ids to query
        ra_min, ra_max : float, optional
            RA bounds in degrees
        dec_min, dec_max : float, optional
            Dec bounds in degrees

        Returns:
        --------
        pd.DataFrame with orbital parameters including Thiele-Innes elements
        """

        # Build constraints
        constraints = ["nss_solution_type LIKE '%Orbital%'"]  # Focus on astrometric orbits

        if source_ids:
            source_list = "','".join(map(str, source_ids))
            constraints.append(f"source_id IN ('{source_list}')")

        if ra_min is not None and ra_max is not None:
            constraints.append(f"ra BETWEEN {ra_min} AND {ra_max}")
        if dec_min is not None and dec_max is not None:
            constraints.append(f"dec BETWEEN {dec_min} AND {dec_max}")

        where_clause = " AND ".join(constraints)

        query = f"""
        SELECT
            source_id, ra, dec, parallax, parallax_error,
            period, period_error,
            eccentricity, eccentricity_error,
            t_periastron, t_periastron_error,
            omega_periastron, omega_periastron_error,
            a_thiele_innes, a_thiele_innes_error,
            b_thiele_innes, b_thiele_innes_error,
            f_thiele_innes, f_thiele_innes_error,
            g_thiele_innes, g_thiele_innes_error,
            center_of_mass_velocity, center_of_mass_velocity_error,
            nss_solution_type,
            goodness_of_fit,
            n_transits,
            n_observations
        FROM gaiadr3.nss_two_body_orbit
        WHERE {where_clause}
        """

        params = {
            'REQUEST': 'doQuery',
            'LANG': 'ADQL',
            'FORMAT': 'csv',
            'QUERY': query.strip()
        }

        print(f"Fetching NSS orbital solutions...")

        try:
            response = requests.get(self.base_url, params=params, timeout=300)
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text))
            print(f"Retrieved {len(df)} NSS orbital solutions")

            return df

        except Exception as e:
            print(f"Error fetching NSS orbital data: {e}")
            raise

    def thiele_innes_to_orbital_elements(self, a_ti: float, b_ti: float,
                                       f_ti: float, g_ti: float,
                                       period: float, parallax: float,
                                       a_ti_err: float = None, b_ti_err: float = None,
                                       f_ti_err: float = None, g_ti_err: float = None,
                                       period_err: float = None, parallax_err: float = None) -> Dict[str, float]:
        """
        Convert Thiele-Innes elements to standard orbital elements

        Thiele-Innes elements relate observed astrometric motion to orbital parameters:
        A = a cos(Omega) cos(omega) - a cos(i) sin(Omega) sin(omega)
        B = a sin(Omega) cos(omega) + a cos(i) cos(Omega) sin(omega)
        F = -a cos(Omega) sin(omega) - a cos(i) sin(Omega) cos(omega)
        G = -a sin(Omega) sin(omega) + a cos(i) cos(Omega) cos(omega)

        Parameters:
        -----------
        a_ti, b_ti, f_ti, g_ti : float
            Thiele-Innes elements (arcsec)
        period : float
            Orbital period (days)
        parallax : float
            Parallax (mas)
        *_err : float, optional
            Uncertainties for error propagation

        Returns:
        --------
        Dict with orbital elements: sma_au, inclination_deg, omega_deg, Omega_deg, etc.
        """

        # Convert Thiele-Innes elements to orbital parameters
        # Angular semimajor axis of photocentre orbit (arcsec)
        alpha = np.sqrt(a_ti**2 + b_ti**2 + f_ti**2 + g_ti**2)

        # Physical semimajor axis in AU using parallax
        if parallax > 0:
            sma_au = alpha / (parallax / 1000.0)  # Convert mas to arcsec
        else:
            sma_au = np.nan

        # Inclination (radians, then convert to degrees)
        cos_i = (a_ti * g_ti - b_ti * f_ti) / (alpha**2)
        cos_i = np.clip(cos_i, -1, 1)  # Ensure valid range
        inclination_rad = np.arccos(abs(cos_i))
        inclination_deg = np.degrees(inclination_rad)

        # Longitude of ascending node (Omega)
        Omega_rad = np.arctan2((a_ti * f_ti + b_ti * g_ti), (a_ti * g_ti - b_ti * f_ti))
        Omega_deg = np.degrees(Omega_rad) % 360

        # Argument of periapsis (omega)
        omega_rad = np.arctan2(-(f_ti * np.cos(Omega_rad) + g_ti * np.sin(Omega_rad)),
                               (a_ti * np.cos(Omega_rad) + b_ti * np.sin(Omega_rad)))
        omega_deg = np.degrees(omega_rad) % 360

        # Uncertainty propagation (simplified linear approximation)
        sma_au_err = np.nan
        inclination_deg_err = np.nan
        omega_deg_err = np.nan
        Omega_deg_err = np.nan

        if all(x is not None for x in [a_ti_err, b_ti_err, f_ti_err, g_ti_err, parallax_err]):
            # Linear error propagation for semimajor axis
            dalpha_da = a_ti / alpha
            dalpha_db = b_ti / alpha
            dalpha_df = f_ti / alpha
            dalpha_dg = g_ti / alpha

            alpha_err = np.sqrt((dalpha_da * a_ti_err)**2 + (dalpha_db * b_ti_err)**2 +
                               (dalpha_df * f_ti_err)**2 + (dalpha_dg * g_ti_err)**2)

            if parallax_err and parallax > 0:
                rel_alpha_err = alpha_err / alpha
                rel_parallax_err = parallax_err / parallax
                sma_au_err = sma_au * np.sqrt(rel_alpha_err**2 + rel_parallax_err**2)

        return {
            'sma_au': sma_au,
            'sma_au_err': sma_au_err,
            'inclination_deg': inclination_deg,
            'inclination_deg_err': inclination_deg_err,
            'omega_deg': omega_deg,
            'omega_deg_err': omega_deg_err,
            'Omega_deg': Omega_deg,
            'Omega_deg_err': Omega_deg_err,
            'angular_sma_arcsec': alpha,
            'cos_inclination': cos_i
        }

    def compute_companion_mass(self, period_days: float, sma_au: float,
                             primary_mass_msun: float,
                             period_err: float = None, sma_err: float = None,
                             primary_mass_err: float = None) -> Tuple[float, float]:
        """
        Compute companion mass using Kepler's third law

        Parameters:
        -----------
        period_days : float
            Orbital period in days
        sma_au : float
            Semimajor axis in AU
        primary_mass_msun : float
            Primary star mass in solar masses
        *_err : float, optional
            Uncertainties

        Returns:
        --------
        Tuple of (companion_mass_msun, companion_mass_err)
        """

        # Kepler's third law: P^2 = 4π^2 a^3 / G(M1 + M2)
        # Rearrange: M2 = 4π^2 a^3 / (G P^2) - M1

        period_years = period_days / 365.25

        # Total system mass from Kepler's law
        G_solar = 39.478  # G in units of AU^3 / (Msun * year^2)
        total_mass = 4 * np.pi**2 * sma_au**3 / (G_solar * period_years**2)

        # Companion mass (assuming primary mass dominates for wide binaries)
        companion_mass = total_mass - primary_mass_msun

        # Error propagation
        companion_mass_err = np.nan
        if all(x is not None for x in [period_err, sma_err, primary_mass_err]):
            # Partial derivatives for error propagation
            dtotal_dP = -2 * total_mass / period_years * (period_err / 365.25)
            dtotal_da = 3 * total_mass / sma_au * sma_err

            total_mass_err = np.sqrt(dtotal_dP**2 + dtotal_da**2)
            companion_mass_err = np.sqrt(total_mass_err**2 + primary_mass_err**2)

        return companion_mass, companion_mass_err

    def process_nss_orbital_catalog(self, nss_df: pd.DataFrame,
                                  primary_masses: pd.Series = None) -> pd.DataFrame:
        """
        Process NSS orbital catalog to derive accurate orbital elements

        Parameters:
        -----------
        nss_df : pd.DataFrame
            NSS two-body orbit solutions from Gaia
        primary_masses : pd.Series, optional
            Primary star masses (indexed by source_id)

        Returns:
        --------
        pd.DataFrame with enhanced orbital parameters
        """

        print(f"Processing {len(nss_df)} NSS orbital solutions...")

        results = []

        for idx, row in nss_df.iterrows():
            try:
                source_id = row['source_id']

                # Extract Thiele-Innes elements
                a_ti = row.get('a_thiele_innes', np.nan)
                b_ti = row.get('b_thiele_innes', np.nan)
                f_ti = row.get('f_thiele_innes', np.nan)
                g_ti = row.get('g_thiele_innes', np.nan)

                # Extract uncertainties
                a_ti_err = row.get('a_thiele_innes_error', None)
                b_ti_err = row.get('b_thiele_innes_error', None)
                f_ti_err = row.get('f_thiele_innes_error', None)
                g_ti_err = row.get('g_thiele_innes_error', None)

                period = row.get('period', np.nan)
                period_err = row.get('period_error', None)
                parallax = row.get('parallax', np.nan)
                parallax_err = row.get('parallax_error', None)

                # Skip if missing critical Thiele-Innes elements
                if any(np.isnan(x) for x in [a_ti, b_ti, f_ti, g_ti, period, parallax]):
                    continue

                # Convert to orbital elements
                orbital_elements = self.thiele_innes_to_orbital_elements(
                    a_ti, b_ti, f_ti, g_ti, period, parallax,
                    a_ti_err, b_ti_err, f_ti_err, g_ti_err, period_err, parallax_err
                )

                # Estimate companion mass if primary mass available
                companion_mass = np.nan
                companion_mass_err = np.nan

                if primary_masses is not None and source_id in primary_masses.index:
                    primary_mass = primary_masses[source_id]
                    if not np.isnan(primary_mass) and not np.isnan(orbital_elements['sma_au']):
                        companion_mass, companion_mass_err = self.compute_companion_mass(
                            period, orbital_elements['sma_au'], primary_mass,
                            period_err, orbital_elements['sma_au_err'], None
                        )

                # Compile results
                result = {
                    'source_id': source_id,
                    'ra': row.get('ra', np.nan),
                    'dec': row.get('dec', np.nan),
                    'parallax': parallax,
                    'period_days': period,
                    'eccentricity': row.get('eccentricity', np.nan),
                    'omega_periastron_deg': row.get('omega_periastron', np.nan),
                    'sma_au_fitted': orbital_elements['sma_au'],
                    'inclination_deg_fitted': orbital_elements['inclination_deg'],
                    'omega_deg_fitted': orbital_elements['omega_deg'],
                    'Omega_deg_fitted': orbital_elements['Omega_deg'],
                    'companion_mass_msun_fitted': companion_mass,
                    'angular_sma_arcsec': orbital_elements['angular_sma_arcsec'],
                    'cos_inclination': orbital_elements['cos_inclination'],
                    'goodness_of_fit': row.get('goodness_of_fit', np.nan),
                    'n_observations': row.get('n_observations', np.nan),
                    'nss_solution_type': row.get('nss_solution_type', ''),
                    'fit_quality': 'high' if row.get('goodness_of_fit', 0) < 1.2 else 'medium'
                }

                # Add uncertainties if available
                if orbital_elements['sma_au_err'] is not None:
                    result.update({
                        'sma_au_err': orbital_elements['sma_au_err'],
                        'inclination_deg_err': orbital_elements['inclination_deg_err'],
                        'omega_deg_err': orbital_elements['omega_deg_err'],
                        'Omega_deg_err': orbital_elements['Omega_deg_err'],
                        'companion_mass_msun_err': companion_mass_err
                    })

                results.append(result)

            except Exception as e:
                print(f"Warning: Error processing source_id {source_id}: {e}")
                continue

        processed_df = pd.DataFrame(results)

        print(f"Successfully processed {len(processed_df)} orbital solutions")
        print(f"High quality fits: {(processed_df['fit_quality'] == 'high').sum()}")

        return processed_df

    def save_enhanced_orbits(self, orbital_df: pd.DataFrame, filename: str):
        """Save enhanced orbital solutions to file"""
        orbital_df.to_csv(filename, index=False)
        print(f"Saved enhanced orbital solutions to {filename}")


if __name__ == "__main__":
    # Test the orbit fitter
    print("Testing Gaia NSS orbit fitting...")

    orbit_fitter = GaiaNSSOrbitFitter()

    # Test Thiele-Innes conversion with example values
    # These are realistic values for a wide binary system
    test_elements = orbit_fitter.thiele_innes_to_orbital_elements(
        a_ti=0.1, b_ti=0.05, f_ti=-0.08, g_ti=0.03,  # arcsec
        period=3650,  # days (10 years)
        parallax=10   # mas (100 pc distance)
    )

    print("Test Thiele-Innes conversion:")
    for key, value in test_elements.items():
        if not np.isnan(value):
            print(f"  {key}: {value:.4f}")

    print("\nOrbit fitting module ready for integration.")
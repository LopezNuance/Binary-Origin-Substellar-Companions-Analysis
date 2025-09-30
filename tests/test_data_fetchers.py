from pathlib import Path

import pandas as pd
import requests

from data_fetchers import (
    NASAExoplanetArchiveFetcher,
    BrownDwarfCatalogueFetcher,
    load_local_data,
)


class DummyResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(f"status {self.status_code}")


def test_nasa_fetcher_fetch_vlms_companions_filters_invalid_rows(monkeypatch):
    csv_text = (
        "pl_name,hostname,st_mass,pl_masse,pl_orbsmax,pl_orbeccen,discoverymethod,st_met,pl_massj,st_rad,st_teff\n"
        "Companion A,Host A,0.1,100,0.05,0.1,Transit,0.0,0.5,0.3,3000\n"
        "Companion B,Host B,0.12,,0.04,0.2,RV,0.1,1.0,0.4,5000\n"  # filtered by st_teff
        "Companion C,Host C,0.09,80,0.06,0.3,Transit,5.0,0.3,0.2,2800\n"  # metallicity outlier -> NaN -> drop
    )

    def fake_get(url, params, timeout):
        assert "TAP" in url
        return DummyResponse(csv_text)

    monkeypatch.setattr(requests, "get", fake_get)

    fetcher = NASAExoplanetArchiveFetcher()
    df = fetcher.fetch_vlms_companions()

    assert len(df) == 2
    assert df.iloc[0]["pl_name"] == "Companion A"
    # Row with metallicity outlier should have NaN metallicity after cleaning
    assert pd.isna(df.iloc[1]["st_met"])


def test_brown_dwarf_fetcher_fetch_catalogue_fallback(monkeypatch):
    csv_text = "M_star,M_comp,orbital_period\n0.1,1.0,10\n0.08,2.0,0\n"
    responses = [
        requests.RequestException("primary source down"),
        DummyResponse(csv_text),
    ]

    def fake_get(url, timeout):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(requests, "get", fake_get)

    fetcher = BrownDwarfCatalogueFetcher()
    df = fetcher.fetch_catalogue()

    # orbital_period zero should become NaN after postprocess
    assert df["orbital_period"].isna().sum() == 1


def test_filter_vlms_hosts_detects_mass_column():
    df = pd.DataFrame({
        "M_star": [0.05, 0.10, 0.25],
        "value": [1, 2, 3],
    })
    fetcher = BrownDwarfCatalogueFetcher()
    filtered = fetcher.filter_vlms_hosts(df, min_stellar_mass=0.06, max_stellar_mass=0.20)

    assert filtered["M_star"].tolist() == [0.10]


def test_load_local_data_reads_available_files(tmp_path: Path):
    nasa_file = tmp_path / "nasa.csv"
    bd_file = tmp_path / "bd.csv"

    pd.DataFrame({"a": [1]}).to_csv(nasa_file, index=False)
    pd.DataFrame({"b": [2]}).to_csv(bd_file, index=False)

    loaded = load_local_data(str(nasa_file), str(bd_file))

    assert not loaded["nasa_df"].empty
    assert not loaded["bd_df"].empty
    assert loaded["nasa_df"].iloc[0]["a"] == 1
    assert loaded["bd_df"].iloc[0]["b"] == 2

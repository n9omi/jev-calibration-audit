"""Fetch weekly petroleum series from the EIA Open Data API (v2).

Uses the v2 `/seriesid/{id}` compatibility route, which accepts legacy v1 series IDs such as
`PET.WCESTUS1.W`. The API key must travel as a URL parameter (EIA does not read it from
headers), so it is never logged here; keep it out of anything you commit.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd
import requests

from jev_audit.config import EIAConfig


class EIAError(RuntimeError):
    pass


_META_KEYS = {
    "period",
    "series",
    "series-description",
    "seriesDescription",
    "duoarea",
    "area-name",
    "product",
    "product-name",
    "process",
    "process-name",
    "units",
    "unit",
    "frequency",
}


def rows_to_series(rows: Iterable[dict[str, Any]]) -> pd.Series:
    """Convert EIA `response.data` rows to a float Series indexed by period.

    Petroleum routes name the data column `value`. If a route uses another name, the first
    numeric non-metadata field is used instead; check one live response when adding series.
    """
    values: dict[pd.Timestamp, float] = {}
    for row in rows:
        period = row.get("period")
        raw = row.get("value")
        if raw is None:
            for key, val in row.items():
                if key in _META_KEYS or key.endswith("-units") or key.endswith("-name"):
                    continue
                try:
                    float(val)
                except (TypeError, ValueError):
                    continue
                raw = val
                break
        if period is None or raw in (None, ""):
            continue
        values[pd.Timestamp(period)] = float(raw)
    series = pd.Series(values, dtype=float).sort_index()
    series.index.name = "week"
    return series


def _get_json(
    session: requests.Session, url: str, params: dict[str, Any], max_retries: int
) -> dict[str, Any]:
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.ConnectionError as exc:
            if attempt == max_retries:
                raise EIAError(f"Could not reach {url}") from exc
        else:
            if resp.status_code == 200:
                payload = resp.json()
                if "error" in payload:
                    raise EIAError(f"EIA returned an error for {url}: {payload['error']}")
                return payload
            if resp.status_code not in (429, 500, 502, 503, 504) or attempt == max_retries:
                raise EIAError(f"EIA request to {url} failed with HTTP {resp.status_code}")
        time.sleep(delay)
        delay *= 2
    raise EIAError(f"EIA request to {url} failed")  # pragma: no cover


def fetch_series(
    series_id: str,
    api_key: str,
    base_url: str = "https://api.eia.gov/v2",
    start: str | None = None,
    session: requests.Session | None = None,
    page_size: int = 5000,
    max_retries: int = 4,
    pause_s: float = 0.3,
) -> pd.Series:
    url = f"{base_url.rstrip('/')}/seriesid/{series_id}"
    sess = session or requests.Session()
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        params: dict[str, Any] = {
            "api_key": api_key,
            "length": page_size,
            "offset": offset,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        if start:
            params["start"] = start
        payload = _get_json(sess, url, params, max_retries)
        response = payload.get("response") or {}
        page = response.get("data") or []
        rows.extend(page)
        offset += len(page)
        total = int(response.get("total") or 0)
        if not page or offset >= total:
            break
        time.sleep(pause_s)
    if not rows:
        raise EIAError(f"No rows returned for {series_id}")
    return rows_to_series(rows)


def fetch_all(eia: EIAConfig, start: str, api_key: str | None = None) -> pd.DataFrame:
    """Fetch every configured series into one wide frame (index: week-ending date)."""
    key = api_key or os.environ.get(eia.api_key_env)
    if not key:
        raise EIAError(f"No EIA API key. Set {eia.api_key_env} (free key: https://www.eia.gov/opendata/).")
    session = requests.Session()
    cols = {}
    for name, spec in eia.series.items():
        cols[name] = fetch_series(spec.id, key, eia.base_url, start=start, session=session)
        time.sleep(0.3)  # stay well under EIA's per-second throttle
    frame = pd.DataFrame(cols).sort_index()
    frame.index.name = "week"
    return frame

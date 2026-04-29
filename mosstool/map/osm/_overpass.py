"""
HTTP client for public Overpass API instances.

Apache on overpass-api.de often returns **406 Not Acceptable** unless the client
looks like a normal browser/OSM toolchain: descriptive User-Agent, Referer, and
broad Accept — not the default ``python-requests/...`` UA nor ``Accept: application/json``
alone. See: https://community.openstreetmap.org/t/overpass-api-error-406/
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import requests
from requests.exceptions import RequestException

INTERPRETER_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
)

_HEADERS = {
    "User-Agent": "mosstool/1 (+https://pypi.org/project/mosstool/) Overpass downloader",
    "Referer": "https://overpass-turbo.eu/",
    "Accept": "*/*",
}


def fetch_overpass(
    query: str,
    proxies: Optional[dict[str, str]] = None,
    timeout: float = 300,
) -> dict[str, Any]:
    """
    POST ``query`` to a public interpreter and return JSON (decoded dict).

    Tries HTTPS mirrors in order; raises RuntimeError only if every attempt fails.
    """
    last_err: Optional[BaseException] = None
    for endpoint in INTERPRETER_ENDPOINTS:
        try:
            resp = requests.post(
                endpoint,
                data={"data": query},
                proxies=proxies,
                headers=dict(_HEADERS),
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except RequestException as e:
            last_err = e
            preview = ""
            if getattr(e, "response", None) is not None and e.response is not None:
                preview = (e.response.text or "")[:512]
            logging.warning(
                "Overpass request failed (%s): %s — body preview: %r",
                endpoint,
                e,
                preview,
            )
    assert last_err is not None  # pragma: no cover — loop exited without success
    raise RuntimeError(
        "All Overpass API endpoints failed. Check network/proxy or try again later."
    ) from last_err

"""
NASA DONKI (Database Of Notifications, Knowledge, Information) — HTTP client.

Base URL: https://api.nasa.gov/DONKI/

Endpoints use UPPERCASE paths (e.g. FLR, not flr). Requires api_key (DEMO_KEY works with low rate limits).

Env: NASA_API_KEY — get a key at https://api.nasa.gov/

Docs (overview): https://api.nasa.gov/ — browse DONKI in the catalog.

Typical FLR fields: flrID, beginTime, peakTime, endTime, classType, activeRegionNum,
sourceLocation, linkedEvents, instruments, ...
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from typing import Any

DONKI_BASE = "https://api.nasa.gov/DONKI"

# Known DONKI paths (must match NASA; case-sensitive)
DONKI_FLR = "FLR"
DONKI_CME = "CME"
DONKI_GST = "GST"
DONKI_SEP = "SEP"
DONKI_IPS = "IPS"
DONKI_MPC = "MPC"
DONKI_RBE = "RBE"
DONKI_HSS = "HSS"
DONKI_NOTIFICATIONS = "notifications"


def get_api_key(explicit: str | None = None) -> str:
    return explicit or os.environ.get("NASA_API_KEY", "DEMO_KEY")


def donki_get(
    endpoint: str,
    start_date: str | None = None,
    end_date: str | None = None,
    api_key: str | None = None,
    extra_params: dict[str, str] | None = None,
    timeout_sec: float = 60.0,
) -> list[dict[str, Any]] | dict[str, Any]:
    """
    GET https://api.nasa.gov/DONKI/<ENDPOINT>?startDate=...&endDate=...&api_key=...

    Parameters
    ----------
    endpoint
        e.g. ``\"FLR\"``, ``\"CME\"``, ``\"notifications\"`` (notifications is lowercase).
    start_date, end_date
        ``YYYY-MM-DD``. If both omitted, NASA defaults apply (often last ~30 days for events).
    extra_params
        e.g. notifications: ``{\"type\": \"all\"}``.

    Returns
    -------
    Parsed JSON. Most event endpoints return a list of dicts; errors return a dict with ``error``.
    """
    api_key = get_api_key(api_key)
    path = endpoint if endpoint.startswith("http") else f"{DONKI_BASE}/{endpoint.lstrip('/')}"
    params: dict[str, str] = {"api_key": api_key}
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date
    if extra_params:
        params.update(extra_params)
    url = f"{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "HeliosGuard-DONKI/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            err = json.loads(body)
        except json.JSONDecodeError:
            err = {"error": True, "code": e.code, "reason": str(e.reason), "body": body[:500]}
        return err

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": True, "message": "invalid JSON", "raw": raw[:500]}

    return data


def _ensure_event_list(out: Any) -> list[dict[str, Any]]:
    """Normalize donki_get result to a list of event dicts (or single error dict)."""
    if isinstance(out, dict) and out.get("error"):
        return [out]
    if isinstance(out, list):
        return out
    if isinstance(out, dict):
        return [out]
    return []


def fetch_flr(
    start_date: str,
    end_date: str,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Solar flares (FLR)."""
    return _ensure_event_list(donki_get(DONKI_FLR, start_date=start_date, end_date=end_date, api_key=api_key))


def fetch_cme(
    start_date: str,
    end_date: str,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Coronal mass ejections."""
    return _ensure_event_list(donki_get(DONKI_CME, start_date=start_date, end_date=end_date, api_key=api_key))


def fetch_gst(
    start_date: str,
    end_date: str,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Geomagnetic storms."""
    return _ensure_event_list(donki_get(DONKI_GST, start_date=start_date, end_date=end_date, api_key=api_key))


def fetch_sep(
    start_date: str,
    end_date: str,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Solar energetic particles."""
    return _ensure_event_list(donki_get(DONKI_SEP, start_date=start_date, end_date=end_date, api_key=api_key))


def fetch_notifications(
    start_date: str | None = None,
    end_date: str | None = None,
    type_: str = "all",
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """DONKI notifications stream."""
    extra = {"type": type_}
    return _ensure_event_list(
        donki_get(
            DONKI_NOTIFICATIONS,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
            extra_params=extra,
        )
    )


def flare_class_type(entry: dict[str, Any]) -> str:
    """classType from FLR record, e.g. M5.1, X2.3."""
    return str(entry.get("classType") or "")


def filter_flr_by_class_prefix(
    flr_list: list[dict[str, Any]],
    prefixes: tuple[str, ...] = ("M", "X"),
) -> list[dict[str, Any]]:
    """Keep flares whose classType starts with one of the prefixes (M/X for strong flares)."""
    out: list[dict[str, Any]] = []
    for e in flr_list:
        if e.get("error"):
            continue
        ct = flare_class_type(e).upper()
        if any(ct.startswith(p.upper()) for p in prefixes):
            out.append(e)
    return out


def default_date_range_last_days(days: int = 30) -> tuple[str, str]:
    """UTC calendar dates for (end - days) .. end."""
    end = date.today()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def summarize_flr_json(flr_list: list[dict[str, Any]], max_rows: int = 20) -> str:
    """Human-readable table for console (ASCII)."""
    lines: list[str] = []
    if not flr_list:
        return "No FLR events in range (or API error / empty)."
    if flr_list[0].get("error"):
        return "DONKI error: " + json.dumps(flr_list[0], ensure_ascii=False)[:800]

    lines.append(f"FLR count: {len(flr_list)}")
    lines.append("peakTime (UTC)          class    AR    sourceLocation")
    for e in flr_list[:max_rows]:
        pk = e.get("peakTime") or e.get("beginTime") or "?"
        ct = flare_class_type(e) or "?"
        ar = e.get("activeRegionNum")
        ars = str(ar) if ar is not None else "?"
        loc = (e.get("sourceLocation") or "")[:24]
        lines.append(f"{str(pk)[:22]:<22} {str(ct):<8} {ars:<5} {loc}")
    if len(flr_list) > max_rows:
        lines.append(f"... ({len(flr_list) - max_rows} more)")
    return "\n".join(lines)


if __name__ == "__main__":
    s, e = default_date_range_last_days(14)
    print("DONKI FLR", s, "..", e, "api_key=", get_api_key()[:8] + "...")
    rows = fetch_flr(s, e)
    print(summarize_flr_json(rows))
    mx = filter_flr_by_class_prefix(rows, ("M", "X"))
    print("M/X class count:", len(mx))

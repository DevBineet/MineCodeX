"""
satellite.py — INSAT-3DS visible-imagery ingestion for VAYUNET's "Cloud Map"
mode.

IMD republishes the latest INSAT-3DS visible-band frame at a fixed URL
(SATELLITE_URL below) — the image *content* changes underneath that same
URL as new satellite passes come in (roughly every 15-30 minutes), rather
than IMD publishing a new URL per frame. There's no history endpoint, so
building a timelapse means polling that URL ourselves, noticing when the
bytes actually changed, and keeping every distinct frame.

Polling faster than IMD's real refresh cadence is harmless: `poll_once()`
hashes the downloaded bytes and skips saving if it's identical to the last
frame we already have, so re-checking every few minutes just costs a
request, not a duplicate frame.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

import db

SATELLITE_URL = "https://mausam.imd.gov.in/Satellite/3Dasiasec_vis.jpg"
STORAGE_DIR = Path(__file__).parent / "satellite_frames"
STORAGE_DIR.mkdir(exist_ok=True)

IST = timezone(timedelta(hours=5, minutes=30))

# IMD's site serves this without a browser looking like a browser tends to
# get blocked by some government WAFs — a normal desktop UA + referer
# matching the actual satellite page avoids that without doing anything
# deceptive about what this is (a straightforward periodic image fetch).
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://mausam.imd.gov.in/imd_latest/contents/satellite.php",
}


async def poll_once() -> dict | None:
    """Fetch the current frame; store it only if it's genuinely new.
    Returns the saved row (as db.insert_satellite_frame returns it) if a
    new frame was stored, or None if IMD is still serving the same one we
    already have."""
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as client:
        resp = await client.get(SATELLITE_URL)
        resp.raise_for_status()
        content = resp.content

    if not content or len(content) < 1024:
        raise ValueError(f"satellite response too small to be a real image ({len(content)} bytes)")

    digest = hashlib.sha256(content).hexdigest()
    if digest == db.latest_satellite_hash():
        return None

    now_utc = datetime.now(timezone.utc)
    ist_time = now_utc.astimezone(IST)
    date_str = ist_time.date().isoformat()

    day_dir = STORAGE_DIR / date_str
    day_dir.mkdir(exist_ok=True)
    fname = f"{ist_time.strftime('%H%M%S')}.jpg"
    (day_dir / fname).write_bytes(content)

    row = {
        "date": date_str,
        "captured_at": now_utc.isoformat(),
        "filename": f"{date_str}/{fname}",
        "sha256": digest,
    }
    return db.insert_satellite_frame(row)
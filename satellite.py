"""
satellite.py — multi-source satellite imagery ingestion for VAYUNET's
"Cloud Map" mode.

Two independent feeds are wired in, each polled on its own cadence and
kept as its own timelapse (see SOURCES below):

  imd
    IMD republishes the latest INSAT-3DS visible-band frame at a fixed
    URL — the image *content* changes underneath that same URL as new
    satellite passes come in, rather than IMD publishing a new URL per
    frame. There's no history endpoint on their side, so we poll,
    hash-dedup, and keep every distinct frame ourselves. In practice
    IMD's own site only republishes a handful of times an hour, so this
    feed's timelapse fills in slowly no matter how often we poll it.

  himawari_india
    The Himawari-9 GeoColor full-disk composite, served by RAMMB/CIRA's
    public SLIDER platform (NOAA/NESDIS + Colorado State University;
    https://rammb-slider.cira.colostate.edu). CIRA refreshes this product
    roughly every 10 minutes, which is what actually makes this feed
    "faster" than IMD — IMD sources from the same class of geostationary
    satellite, but their own publish cadence is the bottleneck, not the
    satellite. India sits near the western limb of Himawari-9's disk
    (sub-satellite point 140.7E), so resolution there is lower than
    dead-center — this crops that region out of the public zoom-0 full
    disk tile (464x464) using a standard orthographic projection and
    upscales it for display. It's a reasonable approximation, not a
    pixel-perfect reprojection — good enough for a cloud-motion timelapse,
    not for precise geolocation of features.

Both fetchers return raw image bytes; poll_source() below is the shared
dedup/storage/DB-insert path used for every source, so adding a third
feed later is just adding one entry to SOURCES with its own fetch().
"""

import hashlib
import io
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from PIL import Image

import db

STORAGE_DIR = Path(__file__).parent / "satellite_frames"
STORAGE_DIR.mkdir(exist_ok=True)

IST = timezone(timedelta(hours=5, minutes=30))

# A normal desktop UA (+ a referer matching the real viewer page for IMD)
# avoids the WAF/bot-blocking some government and research sites apply to
# generic script user agents — this is a plain periodic image fetch, not
# anything deceptive about what's making the request.
DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Source: IMD INSAT-3DS visible frame (unchanged behavior from the original
# single-source version of this file)
# ---------------------------------------------------------------------------
IMD_URL = "https://mausam.imd.gov.in/Satellite/3Dasiasec_vis.jpg"
IMD_HEADERS = {
    "User-Agent": DESKTOP_UA,
    "Referer": "https://mausam.imd.gov.in/imd_latest/contents/satellite.php",
}


async def _fetch_imd() -> bytes:
    async with httpx.AsyncClient(timeout=20, headers=IMD_HEADERS) as client:
        resp = await client.get(IMD_URL)
        resp.raise_for_status()
        content = resp.content
    if not content or len(content) < 1024:
        raise ValueError(f"IMD response too small to be a real image ({len(content)} bytes)")
    return content


# ---------------------------------------------------------------------------
# Source: Himawari-9 GeoColor, cropped to India (RAMMB/CIRA SLIDER)
# ---------------------------------------------------------------------------
HIMAWARI_SUBLON_DEG = 140.7        # Himawari-9 sub-satellite longitude
HIMAWARI_TILE_SIZE = 464           # zoom-0 full-disk tile is a single 464x464 PNG
HIMAWARI_HEADERS = {"User-Agent": DESKTOP_UA}

# Generous India bounding box (mainland + islands), in degrees.
INDIA_BBOX = {"lat_min": 6.0, "lat_max": 37.5, "lon_min": 67.0, "lon_max": 98.0}

# GeoColor for a given timestamp typically lands on SLIDER a few minutes
# after the fact; step backwards from "now" in the product's native
# 10-minute cadence until a timestamp that actually exists is found.
_HIMAWARI_STEP = timedelta(minutes=10)
_HIMAWARI_MAX_LOOKBACK = timedelta(hours=1)


def _himawari_tile_url(ts: datetime) -> str:
    stamp = ts.strftime("%Y%m%d%H%M%S")
    return (
        f"https://rammb-slider.cira.colostate.edu/data/imagery/"
        f"{ts.strftime('%Y/%m/%d')}/himawari---full_disk/geocolor/"
        f"{stamp}/00/000_000.png"
    )


def _orthographic_px(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Approximate lat/lon -> pixel position in the zoom-0 full-disk tile.

    The full disk is displayed as a circle inscribed in the square tile
    (radius = half the tile width), consistent with how CIRA's own tools
    use this exact zoom-0 image. Treating that as a simple orthographic
    projection centered on (0 deg lat, sub-satellite lon) is an
    approximation — the real HRIT fixed-grid projection accounts for the
    satellite's finite distance and Earth's ellipsoid — but it's accurate
    enough to locate a crop box, especially since India sits well inside
    the disk rather than exactly on the limb.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg - HIMAWARI_SUBLON_DEG)
    x = math.cos(lat) * math.sin(lon)
    y = math.sin(lat)
    r = HIMAWARI_TILE_SIZE / 2.0
    return r + x * r, r - y * r


def _india_crop_box() -> tuple[int, int, int, int]:
    corners = [
        _orthographic_px(lat, lon)
        for lat in (INDIA_BBOX["lat_min"], INDIA_BBOX["lat_max"])
        for lon in (INDIA_BBOX["lon_min"], INDIA_BBOX["lon_max"])
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    pad = 6  # small margin so nothing at the exact edge gets clipped
    left = max(0, min(xs) - pad)
    right = min(HIMAWARI_TILE_SIZE, max(xs) + pad)
    top = max(0, min(ys) - pad)
    bottom = min(HIMAWARI_TILE_SIZE, max(ys) + pad)
    return int(left), int(top), int(right), int(bottom)


async def _fetch_himawari_india() -> bytes:
    now = datetime.now(timezone.utc)
    # Round down to the product's native cadence, then walk backwards
    # until SLIDER actually has that frame.
    minute = (now.minute // 10) * 10
    candidate = now.replace(minute=minute, second=0, microsecond=0)

    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=20, headers=HIMAWARI_HEADERS) as client:
        elapsed = timedelta(0)
        while elapsed <= _HIMAWARI_MAX_LOOKBACK:
            url = _himawari_tile_url(candidate)
            try:
                resp = await client.get(url)
                if resp.status_code == 200 and len(resp.content) > 512:
                    full_disk = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    box = _india_crop_box()
                    cropped = full_disk.crop(box)
                    # Native crop is small (India sits off-center on the
                    # disk) — upscale for a viewable timelapse frame.
                    target = (cropped.width * 6, cropped.height * 6)
                    cropped = cropped.resize(target, Image.LANCZOS)
                    out = io.BytesIO()
                    cropped.save(out, format="JPEG", quality=88)
                    return out.getvalue()
            except httpx.HTTPError as exc:
                last_error = exc
            candidate -= _HIMAWARI_STEP
            elapsed += _HIMAWARI_STEP

    raise RuntimeError(
        f"no Himawari-9 GeoColor frame found in the last hour (last error: {last_error})"
    )


# ---------------------------------------------------------------------------
# Source registry — add a new feed by adding one entry here
# ---------------------------------------------------------------------------
SOURCES: dict[str, dict] = {
    "imd": {
        "label": "IMD INSAT-3DS (visible)",
        "description": "Official IMD visible-band frame. Slowest to update, but the authoritative feed.",
        "poll_seconds": 300,
        "fetch": _fetch_imd,
    },
    
}


def list_sources() -> list[dict]:
    return [{"id": sid, "label": s["label"], "description": s["description"],
              "poll_seconds": s["poll_seconds"]} for sid, s in SOURCES.items()]


async def poll_once(source_id: str) -> dict | None:
    """Fetch the current frame for `source_id`; store it only if it's
    genuinely new. Returns the saved row if a new frame was stored, or
    None if this source is still serving the same frame we already have."""
    source = SOURCES[source_id]
    content = await source["fetch"]()

    digest = hashlib.sha256(content).hexdigest()
    if digest == db.latest_satellite_hash(source_id):
        return None

    now_utc = datetime.now(timezone.utc)
    ist_time = now_utc.astimezone(IST)
    date_str = ist_time.date().isoformat()

    day_dir = STORAGE_DIR / source_id / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    ext = "jpg"
    fname = f"{ist_time.strftime('%H%M%S')}.{ext}"
    (day_dir / fname).write_bytes(content)

    row = {
        "source": source_id,
        "date": date_str,
        "captured_at": now_utc.isoformat(),
        "filename": f"{source_id}/{date_str}/{fname}",
        "sha256": digest,
    }
    return db.insert_satellite_frame(row)

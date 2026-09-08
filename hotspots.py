"""
hotspots.py — turns clustering of live citizen/social reports (Verified
AND Unverified both count here — the point is an early signal, not a
confirmed one) into two things for the dashboard's left-panel "Emerging
Pattern Watch":

  1. Hotspots: cities where reports of one event category have piled up
     in a recent time window (see compute_hotspots).
  2. Forecast cone: for one hotspot, whether OTHER nearby cities are
     ALSO seeing a pile-up of the same event category, and if so, in
     what rough direction that suggests the pattern is spreading — a
     center bearing plus a spread ("cone"), rendered as three lines
     (low bound / center / high bound), plus a ranked list of the real
     cities that cone points toward next (see compute_forecast).

This is explicitly a heuristic pattern-detection signal derived from
report density and geometry, NOT a physical model of wind, pressure, or
storm motion, and not an official IMD forecast — compute_forecast says
so in its own "available"/"reason" fields when there isn't enough
surrounding signal to responsibly claim a direction, rather than
inventing one from noise. The frontend must surface that distinction
rather than presenting this as a confirmed forecast.
"""

import math

EARTH_RADIUS_KM = 6371.0

# Mirrors the map legend colors in main.html so a hotspot's color always
# matches how that same event category is drawn elsewhere on the map.
EVENT_COLORS = {
    "Rain": "#3b82f6",
    "Heatwave": "#f43f5e",
    "Thunderstorm": "#a78bfa",
    "Flooding": "#0ea5e9",
    "Fog": "#94a3b8",
    "Dust Storm": "#d97706",
    "Strong Winds": "#22d3ee",
    "Normal": "#7c88ac",
}

# Tuning knobs for both functions below.
MAX_NEIGHBOR_KM = 700          # how far away another city's reports can still count as "the same system"
MIN_NEIGHBOR_SIGNAL = 2        # need at least this many other matching-event cities to claim a direction
CONE_HALF_ANGLE_MIN_DEG = 18.0
CONE_HALF_ANGLE_MAX_DEG = 55.0
FORECAST_LEGS_KM = [150, 350, 600]   # cone line vertices / outer search radius for "next cities"
NEXT_CITY_MIN_KM = 40                # don't list the hotspot's own immediate vicinity as "next"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, 0-360, 0=N."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def destination_point(lat: float, lon: float, bearing: float, distance_km: float) -> tuple[float, float]:
    """Point `distance_km` out from (lat,lon) along `bearing` degrees."""
    br = math.radians(bearing)
    d_r = distance_km / EARTH_RADIUS_KM
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(d_r) + math.cos(lat1) * math.sin(d_r) * math.cos(br))
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(d_r) * math.cos(lat1),
        math.cos(d_r) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180


def circular_weighted_mean_deg(bearings_weights: list[tuple[float, float]]) -> float | None:
    sx = sum(w * math.cos(math.radians(b)) for b, w in bearings_weights)
    sy = sum(w * math.sin(math.radians(b)) for b, w in bearings_weights)
    if abs(sx) < 1e-9 and abs(sy) < 1e-9:
        return None  # signals cancel out exactly (or none) -- no coherent direction
    return (math.degrees(math.atan2(sy, sx)) + 360) % 360


def angular_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def aggregate_city_reports(reports: list[dict], cities: list[dict]) -> dict:
    """Buckets raw report rows by (city, state) into per-city stats,
    anchored to that city's canonical grid coordinates (not the jittered
    per-report lat/lon) so distances/bearings between cities are stable."""
    cities_by_key = {(c["name"], c["state"]): c for c in cities}
    agg: dict[tuple, dict] = {}
    for r in reports:
        key = (r["city"], r["state"])
        c = cities_by_key.get(key)
        if not c:
            continue  # report's city isn't one of our known grid points -- skip for geo purposes
        bucket = agg.setdefault(key, {
            "city": r["city"], "state": r["state"], "lat": c["lat"], "lon": c["lon"],
            "total": 0, "verified": 0, "unverified": 0, "flagged": 0,
            "event_counts": {}, "latest_created_at": r["created_at"],
        })
        bucket["total"] += 1
        status = r["verification_status"]
        if status == "Verified":
            bucket["verified"] += 1
        elif status == "Flagged":
            bucket["flagged"] += 1
        elif status == "Duplicate":
            pass  # not an independent signal
        else:
            bucket["unverified"] += 1
        ev = r["event_category"]
        bucket["event_counts"][ev] = bucket["event_counts"].get(ev, 0) + 1
        if r["created_at"] > bucket["latest_created_at"]:
            bucket["latest_created_at"] = r["created_at"]
    for b in agg.values():
        b["dominant_event"] = max(b["event_counts"], key=b["event_counts"].get)
    return agg


def compute_hotspots(agg: dict, min_reports: int = 4, top_n: int = 6) -> list[dict]:
    """Cities whose report volume (in whatever window `agg` was built
    over) clears `min_reports` and isn't dominated by 'Normal' chatter,
    ranked by volume."""
    hotspots = [b for b in agg.values() if b["total"] >= min_reports and b["dominant_event"] != "Normal"]
    hotspots.sort(key=lambda b: b["total"], reverse=True)
    return hotspots[:top_n]


def compute_forecast(hotspot: dict, agg: dict, cities: list[dict]) -> dict:
    """For one hotspot, look at every OTHER city's aggregated reports:
    if enough of them share the hotspot's dominant event category
    within MAX_NEIGHBOR_KM, treat their bearings from the hotspot
    (weighted by report volume) as a directional signal, and project a
    cone. If not enough signal exists, `available` is False and `reason`
    explains why -- callers must not fabricate a direction in that case."""
    event = hotspot["dominant_event"]
    origin_lat, origin_lon = hotspot["lat"], hotspot["lon"]
    origin_key = (hotspot["city"], hotspot["state"])

    signals = []  # (bearing, weight, distance_km)
    for key, b in agg.items():
        if key == origin_key or b["dominant_event"] != event:
            continue
        dist = haversine_km(origin_lat, origin_lon, b["lat"], b["lon"])
        if dist < 1 or dist > MAX_NEIGHBOR_KM:
            continue
        brg = bearing_deg(origin_lat, origin_lon, b["lat"], b["lon"])
        signals.append((brg, float(b["total"]), dist))

    if len(signals) < MIN_NEIGHBOR_SIGNAL:
        return {
            "available": False,
            "reason": (
                f"Only {len(signals)} other area(s) are showing matching {event.lower()} reports "
                f"right now — need at least {MIN_NEIGHBOR_SIGNAL} to responsibly project a direction."
            ),
            "event": event,
            "origin": {"lat": origin_lat, "lon": origin_lon, "city": hotspot["city"], "state": hotspot["state"]},
            "color": EVENT_COLORS.get(event, "#7c88ac"),
        }

    center_bearing = circular_weighted_mean_deg([(brg, w) for brg, w, _ in signals])
    if center_bearing is None:
        return {
            "available": False,
            "reason": "Surrounding reports are pointing in cancelling directions — no coherent spread direction.",
            "event": event,
            "origin": {"lat": origin_lat, "lon": origin_lon, "city": hotspot["city"], "state": hotspot["state"]},
            "color": EVENT_COLORS.get(event, "#7c88ac"),
        }

    total_w = sum(w for _, w, _ in signals)
    avg_angular_spread = sum(w * angular_diff(brg, center_bearing) for brg, w, _ in signals) / total_w
    # Tighter agreement among neighbors -> narrower, more confident cone.
    half_angle = max(CONE_HALF_ANGLE_MIN_DEG, min(CONE_HALF_ANGLE_MAX_DEG, avg_angular_spread * 1.4 + 10))

    def line(bearing: float) -> list[dict]:
        pts = [{"lat": origin_lat, "lon": origin_lon}]
        for d in FORECAST_LEGS_KM:
            lat, lon = destination_point(origin_lat, origin_lon, bearing, d)
            pts.append({"lat": lat, "lon": lon})
        return pts

    max_leg = FORECAST_LEGS_KM[-1]
    next_cities = []
    for c in cities:
        if (c["name"], c["state"]) == origin_key:
            continue
        dist = haversine_km(origin_lat, origin_lon, c["lat"], c["lon"])
        if dist < NEXT_CITY_MIN_KM or dist > max_leg:
            continue
        brg = bearing_deg(origin_lat, origin_lon, c["lat"], c["lon"])
        off = angular_diff(brg, center_bearing)
        if off > half_angle:
            continue
        # Heuristic score, not a physical probability: closer to the
        # center line and closer in distance both push it up.
        angular_score = 1 - (off / half_angle)
        distance_score = 1 - (dist / max_leg)
        probability = round(max(0.05, min(0.92, 0.5 * angular_score + 0.5 * distance_score)), 2)
        # Placeholder ETA from a generic regional system speed assumption
        # (~25 km/h) -- NOT derived from real wind/motion data.
        eta_hours = round(dist / 25, 1)
        already = agg.get((c["name"], c["state"]))
        next_cities.append({
            "name": c["name"], "state": c["state"], "lat": c["lat"], "lon": c["lon"],
            "distance_km": round(dist), "eta_hours": eta_hours, "probability": probability,
            "already_reporting": bool(already and already["dominant_event"] == event),
        })
    next_cities.sort(key=lambda x: -x["probability"])

    return {
        "available": True,
        "event": event,
        "color": EVENT_COLORS.get(event, "#7c88ac"),
        "origin": {"lat": origin_lat, "lon": origin_lon, "city": hotspot["city"], "state": hotspot["state"]},
        "center_bearing": round(center_bearing, 1),
        "half_angle_deg": round(half_angle, 1),
        "confidence_signals": len(signals),
        "lines": {
            "low": line((center_bearing - half_angle) % 360),
            "center": line(center_bearing),
            "high": line((center_bearing + half_angle) % 360),
        },
        "next_cities": next_cities[:8],
    }

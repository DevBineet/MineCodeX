"""
pipeline.py — the "AI/ML-based" processing stage of the ingestion pipeline.

The problem statement asks participants to use ML/AI techniques to (a)
auto-categorize weather events, (b) score source trust / flag fake or
misleading reports, and (c) remove duplicates. This module implements all
three as transparent, auditable rule sets rather than a trained model,
because there's no labelled Indian weather-social-media dataset available
here to train or validate one honestly. Each function below has the exact
interface a trained model would have (text/metadata in, label or score
out), so it's a direct drop-in point: swap `classify_event_from_text` for
a fine-tuned text classifier (e.g. IndicBERT) and `score_trust` for a
learned credibility model, and nothing else in the app needs to change.
"""

import difflib
import math

EVENT_KEYWORDS = {
    "Flooding":       ["flood", "waterlog", "inundat", "overflow", "submerg", "swept away"],
    "Thunderstorm":   ["thunder", "lightning", "storm cell", "squall", "hailstorm", "hail"],
    "Heatwave":       ["heatwave", "heat wave", "scorching", "extreme heat", "sunstroke", "heatstroke"],
    "Fog":            ["fog", "smog", "low visibility", "mist"],
    "Dust Storm":     ["dust storm", "duster", "dust haze", "sandstorm", "aandhi"],
    "Strong Winds":   ["strong wind", "gale", "gusty", "high wind", "tree fell", "uprooted"],
    "Rain":           ["rain", "downpour", "drizzle", "showers", "monsoon", "cloudburst"],
}

# Prior credibility by channel. Open social media starts lower and has to
# earn trust via corroborating signal (GPS, media, hashtag, text quality);
# official/instrumented sources start high.
TRUSTED_SOURCES = {"IMD Station", "Public API", "Satellite Feed"}
SEMI_TRUSTED_SOURCES = {"Citizen App"}

SPAM_MARKERS = ["click here", "bit.ly", "free followers", "subscribe now", "!!!!", "www.win"]


def classify_event_from_text(text: str, fallback: str = "Normal") -> str:
    """Keyword-frequency classifier. Picks the category with the most
    keyword hits in the (lower-cased) text; falls back to `fallback`
    (usually "Normal") when nothing matches."""
    t = (text or "").lower()
    best, best_hits = fallback, 0
    for event, keywords in EVENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in t)
        if hits > best_hits:
            best, best_hits = event, hits
    return best


def score_trust(source: str, text: str, has_gps: bool, has_media: bool, hashtags: list[str]) -> float:
    """Heuristic 0..1 credibility score used to drive the verification
    pipeline. Higher = more likely to be a genuine, well-corroborated
    report; lower = more likely fake, spammy, or unverifiable."""
    score = 0.5
    if source in TRUSTED_SOURCES:
        score += 0.30
    elif source in SEMI_TRUSTED_SOURCES:
        score += 0.15
    else:
        score -= 0.05  # unauthenticated open social post

    if has_gps:
        score += 0.12
    if has_media:
        score += 0.10
    if any(h.lower() in ("imd", "weatherindia", "MineCodeX", "indianweather") for h in hashtags):
        score += 0.05

    text = text or ""
    if 15 <= len(text) <= 280:
        score += 0.05

    lowered = text.lower()
    if any(marker in lowered for marker in SPAM_MARKERS):
        score -= 0.45
    if len(set(text.split())) < 3:
        score -= 0.20  # near-empty / low-information text

    return max(0.0, min(1.0, round(score, 2)))


def verification_from_score(score: float) -> str:
    if score >= 0.72:
        return "Verified"
    if score >= 0.40:
        return "Unverified"
    return "Flagged"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def find_duplicate(
    new_text: str,
    new_lat: float,
    new_lon: float,
    candidates: list[dict],
    radius_km: float = 5.0,
    text_similarity_threshold: float = 0.6,
) -> int | None:
    """Near-duplicate detection: same event category (caller pre-filters
    candidates to that), within `radius_km` of each other, and
    `difflib`-similar text. Returns the id of the earliest matching report
    to point the duplicate at, or None."""
    for c in candidates:
        if haversine_km(new_lat, new_lon, c["lat"], c["lon"]) > radius_km:
            continue
        similarity = difflib.SequenceMatcher(None, (new_text or "").lower(), (c["raw_text"] or "").lower()).ratio()
        if similarity >= text_similarity_threshold:
            return c["id"]
    return None
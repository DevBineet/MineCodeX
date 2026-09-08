"""
db.py — centralized storage for MineCodeX.

The problem statement asks for collected reports (social posts, citizen
reports, station data, public datasets) with metadata — date/time, city,
state, GPS, media, event category, verification status — to land in "a
centralized database". SQLite is used here so the whole platform runs
with zero external services to stand up (no Postgres/Mongo cluster
needed to try this out) — but every caller in the app goes through the
functions in this module, so swapping in Postgres/PostGIS (or a
Kafka -> Spark -> HBase style big-data stack) for a real deployment is a
change to this file only, not to backend.py, pipeline.py, or the
frontend.

Table `reports` is the single source of truth: one row per ingested item
(social post / citizen submission / IMD station reading), each carrying
the metadata block the problem statement calls out — timestamp, city,
state, lat/lon, media, event category, source, and a verification/trust
trail.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "MineCodeXeX.db"

_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source              TEXT NOT NULL,          -- Twitter/X, Citizen App, Public API, Satellite Feed, IMD Station...
    raw_text            TEXT NOT NULL,
    city                TEXT NOT NULL,
    state               TEXT NOT NULL,
    lat                 REAL NOT NULL,
    lon                 REAL NOT NULL,
    event_category      TEXT NOT NULL,          -- Rain / Thunderstorm / Flooding / Heatwave / Fog / Dust Storm / Strong Winds / Normal
    hashtags            TEXT NOT NULL DEFAULT '[]',
    media_type          TEXT NOT NULL DEFAULT 'none',   -- none | photo | video
    media_url           TEXT,
    trust_score         REAL NOT NULL,
    verification_status TEXT NOT NULL,          -- Verified | Unverified | Flagged | Duplicate
    duplicate_of        INTEGER,
    reviewer_note       TEXT,
    created_at          TEXT NOT NULL           -- ISO-8601 UTC
);
CREATE INDEX IF NOT EXISTS idx_reports_created  ON reports(created_at);
CREATE INDEX IF NOT EXISTS idx_reports_event    ON reports(event_category);
CREATE INDEX IF NOT EXISTS idx_reports_state    ON reports(state);
CREATE INDEX IF NOT EXISTS idx_reports_verify   ON reports(verification_status);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id   INTEGER NOT NULL,
    action      TEXT NOT NULL,
    actor       TEXT NOT NULL,
    note        TEXT,
    at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS satellite_frames (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL DEFAULT 'imd',  -- which feed this frame came from, e.g. "imd", "himawari_india"
    date         TEXT NOT NULL,     -- IST calendar date (YYYY-MM-DD) the frame belongs to, for "today's timelapse"
    captured_at  TEXT NOT NULL,     -- UTC ISO-8601 timestamp of when we fetched it
    filename     TEXT NOT NULL,     -- relative path under satellite_frames/, e.g. "imd/2026-08-27/100512.jpg"
    sha256       TEXT NOT NULL      -- content hash, used to skip re-saving a frame the source hasn't updated yet
);
CREATE INDEX IF NOT EXISTS idx_satellite_date ON satellite_frames(date);
CREATE INDEX IF NOT EXISTS idx_satellite_source_date ON satellite_frames(source, date);
"""

REPORT_COLUMNS = [
    "source", "raw_text", "city", "state", "lat", "lon", "event_category",
    "hashtags", "media_type", "media_url", "trust_score",
    "verification_status", "duplicate_of", "reviewer_note", "created_at",
]


def init_db() -> None:
    with _lock:
        _conn.executescript(SCHEMA)
        # Migration: DBs created before multi-source Cloud Map won't have
        # the `source` column yet (CREATE TABLE IF NOT EXISTS doesn't
        # retrofit existing tables). Add it and backfill old rows as
        # "imd", since that was the only feed that ever existed before.
        cols = {r["name"] for r in _conn.execute("PRAGMA table_info(satellite_frames)")}
        if "source" not in cols:
            _conn.execute("ALTER TABLE satellite_frames ADD COLUMN source TEXT NOT NULL DEFAULT 'imd'")
            _conn.execute("CREATE INDEX IF NOT EXISTS idx_satellite_source_date ON satellite_frames(source, date)")
        _conn.commit()


def is_empty() -> bool:
    with _lock:
        row = _conn.execute("SELECT COUNT(*) c FROM reports").fetchone()
    return row["c"] == 0


def insert_report(row: dict) -> dict:
    placeholders = ",".join("?" for _ in REPORT_COLUMNS)
    cols = ",".join(REPORT_COLUMNS)
    vals = [row.get(c) for c in REPORT_COLUMNS]
    with _lock:
        cur = _conn.execute(f"INSERT INTO reports ({cols}) VALUES ({placeholders})", vals)
        _conn.commit()
        new_id = cur.lastrowid
        saved = _conn.execute("SELECT * FROM reports WHERE id = ?", (new_id,)).fetchone()
    return dict(saved)


def recent_candidates(event_category: str, since_iso: str, limit: int = 300) -> list[dict]:
    """Reports in the same event category since `since_iso`, newest first —
    the candidate pool the duplicate detector checks a new report against."""
    with _lock:
        rows = _conn.execute(
            "SELECT * FROM reports WHERE event_category = ? AND created_at >= ? "
            "ORDER BY created_at DESC LIMIT ?",
            (event_category, since_iso, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def query_reports(
    date: Optional[str] = None,
    event: Optional[str] = None,
    state: Optional[str] = None,
    verification: Optional[str] = None,
    limit: int = 300,
    offset: int = 0,
) -> list[dict]:
    q = "SELECT * FROM reports WHERE 1=1"
    params: list = []
    if date:
        q += " AND substr(created_at, 1, 10) = ?"
        params.append(date)
    if event and event != "All":
        q += " AND event_category = ?"
        params.append(event)
    if state and state != "All":
        q += " AND state = ?"
        params.append(state)
    if verification and verification != "All":
        q += " AND verification_status = ?"
        params.append(verification)
    q += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params += [limit, offset]
    with _lock:
        rows = _conn.execute(q, params).fetchall()
    return [dict(r) for r in rows]


def get_report(report_id: int) -> Optional[dict]:
    with _lock:
        row = _conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    return dict(row) if row else None


def update_verification(report_id: int, status: str, note: Optional[str], actor: str = "admin") -> bool:
    with _lock:
        cur = _conn.execute(
            "UPDATE reports SET verification_status = ?, reviewer_note = COALESCE(?, reviewer_note) WHERE id = ?",
            (status, note, report_id),
        )
        _conn.execute(
            "INSERT INTO audit_log (report_id, action, actor, note, at) VALUES (?,?,?,?,?)",
            (report_id, f"set_{status.lower()}", actor, note, datetime.now(timezone.utc).isoformat()),
        )
        _conn.commit()
    return cur.rowcount > 0


def get_states() -> list[str]:
    with _lock:
        rows = _conn.execute("SELECT DISTINCT state FROM reports ORDER BY state").fetchall()
    return [r["state"] for r in rows]


def get_stats() -> dict:
    with _lock:
        total = _conn.execute("SELECT COUNT(*) c FROM reports").fetchone()["c"]
        by_event = {
            r["event_category"]: r["c"]
            for r in _conn.execute(
                "SELECT event_category, COUNT(*) c FROM reports GROUP BY event_category"
            ).fetchall()
        }
        by_verification = {
            r["verification_status"]: r["c"]
            for r in _conn.execute(
                "SELECT verification_status, COUNT(*) c FROM reports GROUP BY verification_status"
            ).fetchall()
        }
        by_state = {
            r["state"]: r["c"]
            for r in _conn.execute(
                "SELECT state, COUNT(*) c FROM reports GROUP BY state ORDER BY c DESC LIMIT 15"
            ).fetchall()
        }
        last_24h = _conn.execute(
            "SELECT COUNT(*) c FROM reports WHERE created_at >= datetime('now', '-1 day')"
        ).fetchone()["c"]
    return {
        "total": total,
        "last_24h": last_24h,
        "by_event": by_event,
        "by_verification": by_verification,
        "by_state": by_state,
    }


# ---------------------------------------------------------------------------
# Satellite frames — multi-source "Cloud Map" mode (IMD, Himawari-9, ...).
# Every function is keyed by `source` so each feed gets its own dedup
# chain, its own date index, and its own independent timelapse; nothing
# here assumes there's only one feed.
# ---------------------------------------------------------------------------
def insert_satellite_frame(row: dict) -> dict:
    with _lock:
        cur = _conn.execute(
            "INSERT INTO satellite_frames (source, date, captured_at, filename, sha256) VALUES (?,?,?,?,?)",
            (row["source"], row["date"], row["captured_at"], row["filename"], row["sha256"]),
        )
        _conn.commit()
        new_id = cur.lastrowid
        saved = _conn.execute("SELECT * FROM satellite_frames WHERE id = ?", (new_id,)).fetchone()
    return dict(saved)


def latest_satellite_hash(source: str) -> Optional[str]:
    with _lock:
        row = _conn.execute(
            "SELECT sha256 FROM satellite_frames WHERE source = ? ORDER BY id DESC LIMIT 1", (source,)
        ).fetchone()
    return row["sha256"] if row else None


def latest_satellite_frame(source: str) -> Optional[dict]:
    with _lock:
        row = _conn.execute(
            "SELECT * FROM satellite_frames WHERE source = ? ORDER BY id DESC LIMIT 1", (source,)
        ).fetchone()
    return dict(row) if row else None


def list_satellite_frames(source: str, date: str) -> list[dict]:
    with _lock:
        rows = _conn.execute(
            "SELECT * FROM satellite_frames WHERE source = ? AND date = ? ORDER BY captured_at ASC",
            (source, date),
        ).fetchall()
    return [dict(r) for r in rows]


def list_satellite_dates(source: str) -> list[str]:
    with _lock:
        rows = _conn.execute(
            "SELECT DISTINCT date FROM satellite_frames WHERE source = ? ORDER BY date DESC", (source,)
        ).fetchall()
    return [r["date"] for r in rows]


def count_satellite_frames_for_date(date: str) -> dict[str, int]:
    """Frame counts per source for a given date — used to pick a sensible
    default source (whichever currently has the longest timelapse)."""
    with _lock:
        rows = _conn.execute(
            "SELECT source, COUNT(*) c FROM satellite_frames WHERE date = ? GROUP BY source", (date,)
        ).fetchall()
    return {r["source"]: r["c"] for r in rows}


# ---------------------------------------------------------------------------
# Rolling-window report pull for hotspots.py — pattern/forecast detection
# ---------------------------------------------------------------------------
def recent_reports(since_iso: str, limit: int = 3000) -> list[dict]:
    """All reports (any event/verification status) since `since_iso`,
    newest first — the rolling window hotspots.py's compute_hotspots() /
    compute_forecast() operate over. Unlike recent_candidates() this is
    not filtered to one event category, since a hotspot's dominant event
    is only known after aggregating."""
    with _lock:
        rows = _conn.execute(
            "SELECT * FROM reports WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
            (since_iso, limit),
        ).fetchall()
    return [dict(r) for r in rows]

"""
NBA Rest & Back-to-Back Engine — Schedule density, rest days, and travel fatigue.

ESPN Scoreboard API: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
Free, public, no API key required.

Used by: server.py analysis prompt (rest/fatigue context injection)
Cache: 2 hours for schedule data
"""

import httpx
import logging
import math
from datetime import datetime, timedelta, timezone

from engines.mlb_pitcher_engine import _get_cached, _set_cache

logger = logging.getLogger("edge-crew")

ESPN_NBA_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

# NBA team abbreviation → (city, latitude, longitude)
# Used for travel distance estimation between game locations.
NBA_TEAM_LOCATIONS = {
    "ATL": ("Atlanta", 33.749, -84.388),
    "BOS": ("Boston", 42.361, -71.057),
    "BKN": ("Brooklyn", 40.683, -73.975),
    "CHA": ("Charlotte", 35.225, -80.839),
    "CHI": ("Chicago", 41.881, -87.674),
    "CLE": ("Cleveland", 41.497, -81.688),
    "DAL": ("Dallas", 32.790, -96.810),
    "DEN": ("Denver", 39.749, -104.999),
    "DET": ("Detroit", 42.341, -83.055),
    "GS":  ("San Francisco", 37.768, -122.388),
    "HOU": ("Houston", 29.751, -95.362),
    "IND": ("Indianapolis", 39.764, -86.156),
    "LAC": ("Los Angeles", 34.043, -118.267),
    "LAL": ("Los Angeles", 34.043, -118.267),
    "MEM": ("Memphis", 35.138, -90.051),
    "MIA": ("Miami", 25.781, -80.187),
    "MIL": ("Milwaukee", 43.045, -87.917),
    "MIN": ("Minneapolis", 44.980, -93.276),
    "NO":  ("New Orleans", 29.949, -90.082),
    "NY":  ("New York", 40.751, -73.994),
    "OKC": ("Oklahoma City", 35.463, -97.515),
    "ORL": ("Orlando", 28.539, -81.384),
    "PHI": ("Philadelphia", 39.901, -75.172),
    "PHX": ("Phoenix", 33.446, -112.071),
    "POR": ("Portland", 45.532, -122.667),
    "SAC": ("Sacramento", 38.580, -121.500),
    "SA":  ("San Antonio", 29.427, -98.438),
    "TOR": ("Toronto", 43.643, -79.379),
    "UTA": ("Salt Lake City", 40.768, -111.901),
    "WAS": ("Washington", 38.898, -77.021),
}

# ESPN uses its own abbreviations — map common ESPN variants to our keys
ESPN_ABBR_MAP = {
    "GS": "GS", "GSW": "GS",
    "NY": "NY", "NYK": "NY",
    "NO": "NO", "NOP": "NO",
    "SA": "SA", "SAS": "SA",
    "UTA": "UTA", "UTAH": "UTA",
    "WSH": "WAS", "WAS": "WAS",
    "PHO": "PHX", "PHX": "PHX",
    "BKN": "BKN", "BRK": "BKN",
    "CHA": "CHA", "CHO": "CHA",
}


def _normalize_abbr(abbr: str) -> str:
    """Normalize ESPN team abbreviation to our standard key."""
    upper = (abbr or "").upper().strip()
    return ESPN_ABBR_MAP.get(upper, upper)


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate approximate distance in miles between two coordinates."""
    R = 3958.8  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def _travel_distance(team_a: str, team_b: str) -> float:
    """Estimate travel distance in miles between two team cities."""
    loc_a = NBA_TEAM_LOCATIONS.get(team_a)
    loc_b = NBA_TEAM_LOCATIONS.get(team_b)
    if not loc_a or not loc_b:
        return 0.0
    return _haversine_miles(loc_a[1], loc_a[2], loc_b[1], loc_b[2])


async def fetch_nba_scoreboard(date_str: str) -> list:
    """Fetch NBA scoreboard for a given date from ESPN API.

    Args:
        date_str: Date in YYYYMMDD format.

    Returns list of dicts:
        [{"home": "LAL", "away": "BOS", "status": "Final"}, ...]
    Cache: 2 hours
    """
    cache_key = f"nba_scoreboard:{date_str}"
    cached = _get_cached(cache_key, ttl=7200)  # 2 hours
    if cached is not None:
        return cached

    url = ESPN_NBA_SCOREBOARD
    params = {"dates": date_str}

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NBA REST] Scoreboard fetch failed for {date_str}: {e}")
        return []

    games = []
    for event in data.get("events", []):
        competitors = event.get("competitions", [{}])[0].get("competitors", [])
        home_abbr = ""
        away_abbr = ""
        for comp in competitors:
            abbr = _normalize_abbr(comp.get("team", {}).get("abbreviation", ""))
            if comp.get("homeAway") == "home":
                home_abbr = abbr
            else:
                away_abbr = abbr

        status = event.get("status", {}).get("type", {}).get("name", "")
        games.append({
            "home": home_abbr,
            "away": away_abbr,
            "status": status,
        })

    _set_cache(cache_key, games)
    return games


async def _fetch_week_schedule(reference_date: datetime) -> dict:
    """Fetch the past 7 days of NBA scoreboards and build per-team game logs.

    Returns dict keyed by team abbreviation:
        {"LAL": [{"date": "20260401", "home_away": "home", "opponent": "BOS"}, ...], ...}
    Cache: 2 hours (via individual scoreboard caching)
    """
    cache_key = f"nba_week_schedule:{reference_date.strftime('%Y%m%d')}"
    cached = _get_cached(cache_key, ttl=7200)
    if cached is not None:
        return cached

    team_logs = {}

    for days_ago in range(1, 8):
        dt = reference_date - timedelta(days=days_ago)
        date_str = dt.strftime("%Y%m%d")
        games = await fetch_nba_scoreboard(date_str)

        for game in games:
            # Only count completed games
            if "Final" not in (game.get("status") or ""):
                continue

            home = game["home"]
            away = game["away"]

            if home:
                team_logs.setdefault(home, []).append({
                    "date": date_str,
                    "home_away": "home",
                    "opponent": away,
                })
            if away:
                team_logs.setdefault(away, []).append({
                    "date": date_str,
                    "home_away": "away",
                    "opponent": home,
                })

    # Sort each team's log by date descending (most recent first)
    for team in team_logs:
        team_logs[team].sort(key=lambda g: g["date"], reverse=True)

    _set_cache(cache_key, team_logs)
    return team_logs


def _compute_rest_profile(team_abbr: str, team_logs: dict, today_str: str) -> dict:
    """Compute rest/schedule profile for a team.

    Returns:
        {"rest_days": int, "is_b2b": bool, "games_in_7d": int,
         "road_trip_len": int, "last_game_date": str, "last_opponent": str}
    """
    logs = team_logs.get(team_abbr, [])
    if not logs:
        return {
            "rest_days": 99,
            "is_b2b": False,
            "games_in_7d": 0,
            "road_trip_len": 0,
            "last_game_date": "",
            "last_opponent": "",
        }

    # Rest days since last game
    last_game = logs[0]
    try:
        last_dt = datetime.strptime(last_game["date"], "%Y%m%d")
        today_dt = datetime.strptime(today_str, "%Y%m%d")
        rest_days = (today_dt - last_dt).days - 1  # days between, minus game day
    except (ValueError, TypeError):
        rest_days = 99

    is_b2b = rest_days == 0

    # Games in last 7 days
    games_in_7d = len(logs)

    # Consecutive away games (road trip length)
    road_trip_len = 0
    for game in logs:
        if game["home_away"] == "away":
            road_trip_len += 1
        else:
            break

    return {
        "rest_days": max(rest_days, 0),
        "is_b2b": is_b2b,
        "games_in_7d": games_in_7d,
        "road_trip_len": road_trip_len,
        "last_game_date": last_game["date"],
        "last_opponent": last_game.get("opponent", ""),
    }


def _fatigue_tag(profile: dict, is_home: bool) -> str:
    """Assign qualitative fatigue tag based on rest profile.

    Tags:
        RESTED — 2+ days rest, home
        FATIGUED — B2B road or 4+ games in 5 nights (games_in_7d >= 4)
        NEUTRAL — everything else
    """
    if profile["is_b2b"] and not is_home:
        return "FATIGUED"
    if profile["games_in_7d"] >= 4:
        return "FATIGUED"
    if profile["rest_days"] >= 2 and is_home:
        return "RESTED"
    return "NEUTRAL"


def _schedule_description(profile: dict, is_home: bool) -> str:
    """Build human-readable schedule context string."""
    parts = []

    if is_home:
        if profile["road_trip_len"] == 0:
            parts.append("Home stand")
        else:
            parts.append("Home")
    else:
        if profile["road_trip_len"] > 0:
            parts.append(f"Road trip ({profile['road_trip_len'] + 1} games)")
        else:
            parts.append("Road")

    return " | ".join(parts) if parts else ""


async def build_rest_context(games: list) -> str:
    """Build the rest/schedule context block for the AI analysis prompt.

    Args:
        games: list of dicts with at minimum {"home": "LAL", "away": "BOS"}

    Returns formatted text:
    === REST & SCHEDULE ===
    RULE: B2B road teams vs rested home teams historically cover 8-10% less. ...

      LAL: 1 day rest | NOT B2B | 2 games in 7d | Home stand — RESTED
      BOS: 0 days rest | B2B (road) | 3 games in 7d | Road trip (4th game) — FATIGUED
      Rest edge: LAL (+1 day, home vs road B2B)
    """
    if not games:
        return "=== REST & SCHEDULE ===\nNo games to evaluate."

    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y%m%d")

    team_logs = await _fetch_week_schedule(now)

    lines = ["=== REST & SCHEDULE ==="]
    lines.append(
        "RULE: B2B road teams vs rested home teams historically cover 8-10% less. "
        "Rest differential > 2 days = significant edge. "
        "4-in-5 nights = fatigue risk."
    )
    lines.append("")

    for game in games:
        away = game.get("away", "?")
        home = game.get("home", "?")

        away_profile = _compute_rest_profile(away, team_logs, today_str)
        home_profile = _compute_rest_profile(home, team_logs, today_str)

        away_tag = _fatigue_tag(away_profile, is_home=False)
        home_tag = _fatigue_tag(home_profile, is_home=True)

        # Away line
        away_b2b = "B2B (road)" if away_profile["is_b2b"] else "NOT B2B"
        away_sched = _schedule_description(away_profile, is_home=False)
        lines.append(
            f"  {away}: {away_profile['rest_days']} day rest | {away_b2b} | "
            f"{away_profile['games_in_7d']} games in 7d | {away_sched} — {away_tag}"
        )

        # Home line
        home_b2b = "B2B (home)" if home_profile["is_b2b"] else "NOT B2B"
        home_sched = _schedule_description(home_profile, is_home=True)
        lines.append(
            f"  {home}: {home_profile['rest_days']} day rest | {home_b2b} | "
            f"{home_profile['games_in_7d']} games in 7d | {home_sched} — {home_tag}"
        )

        # Rest edge summary
        rest_diff = home_profile["rest_days"] - away_profile["rest_days"]
        travel_miles = _travel_distance(away, home)
        cross_country = travel_miles > 1500

        edge_parts = []
        if rest_diff > 0:
            edge_parts.append(f"{home} (+{rest_diff} day{'s' if rest_diff > 1 else ''} rest)")
        elif rest_diff < 0:
            edge_parts.append(f"{away} (+{abs(rest_diff)} day{'s' if abs(rest_diff) > 1 else ''} rest)")
        else:
            edge_parts.append("Even rest")

        if home_profile["is_b2b"] != away_profile["is_b2b"]:
            b2b_team = away if away_profile["is_b2b"] else home
            rested_team = home if away_profile["is_b2b"] else away
            edge_parts.append(f"{b2b_team} on B2B vs rested {rested_team}")

        if cross_country:
            edge_parts.append(f"cross-country travel ({int(travel_miles)} mi)")

        lines.append(f"  Rest edge: {', '.join(edge_parts)}")
        lines.append("")

    return "\n".join(lines)

"""
NHL Travel & Schedule Engine — Rest, B2B, travel distance, timezone fatigue.

ESPN Scoreboard API: https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard
Free, public, no API key required.

Used by: server.py analysis prompt (travel + schedule context injection)
Cache: 2 hours for schedule data
"""

import math
import httpx
import logging
from datetime import datetime, timedelta, timezone

from engines.mlb_pitcher_engine import _get_cached, _set_cache

logger = logging.getLogger("edge-crew")

ESPN_NHL_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"

# ---------------------------------------------------------------------------
# NHL ARENAS — All 32 teams (2024-25 season)
#
# arena_name: current arena name
# city: city name
# timezone: ET / CT / MT / PT
# lat, lon: venue coordinates for haversine distance
# ---------------------------------------------------------------------------

NHL_ARENAS = {
    "ANA": {
        "arena_name": "Honda Center",
        "city": "Anaheim",
        "timezone": "PT",
        "lat": 33.8078,
        "lon": -117.8765,
    },
    "ARI": {
        "arena_name": "Mullett Arena",
        "city": "Tempe",
        "timezone": "MT",
        "lat": 33.4255,
        "lon": -111.9325,
    },
    "BOS": {
        "arena_name": "TD Garden",
        "city": "Boston",
        "timezone": "ET",
        "lat": 42.3662,
        "lon": -71.0621,
    },
    "BUF": {
        "arena_name": "KeyBank Center",
        "city": "Buffalo",
        "timezone": "ET",
        "lat": 42.8750,
        "lon": -78.8764,
    },
    "CGY": {
        "arena_name": "Scotiabank Saddledome",
        "city": "Calgary",
        "timezone": "MT",
        "lat": 51.0375,
        "lon": -114.0519,
    },
    "CAR": {
        "arena_name": "PNC Arena",
        "city": "Raleigh",
        "timezone": "ET",
        "lat": 35.8032,
        "lon": -78.7220,
    },
    "CHI": {
        "arena_name": "United Center",
        "city": "Chicago",
        "timezone": "CT",
        "lat": 41.8807,
        "lon": -87.6742,
    },
    "COL": {
        "arena_name": "Ball Arena",
        "city": "Denver",
        "timezone": "MT",
        "lat": 39.7487,
        "lon": -105.0077,
    },
    "CBJ": {
        "arena_name": "Nationwide Arena",
        "city": "Columbus",
        "timezone": "ET",
        "lat": 39.9691,
        "lon": -83.0062,
    },
    "DAL": {
        "arena_name": "American Airlines Center",
        "city": "Dallas",
        "timezone": "CT",
        "lat": 32.7905,
        "lon": -96.8103,
    },
    "DET": {
        "arena_name": "Little Caesars Arena",
        "city": "Detroit",
        "timezone": "ET",
        "lat": 42.3411,
        "lon": -83.0553,
    },
    "EDM": {
        "arena_name": "Rogers Place",
        "city": "Edmonton",
        "timezone": "MT",
        "lat": 53.5469,
        "lon": -113.4979,
    },
    "FLA": {
        "arena_name": "Amerant Bank Arena",
        "city": "Sunrise",
        "timezone": "ET",
        "lat": 26.1584,
        "lon": -80.3256,
    },
    "LA": {
        "arena_name": "Crypto.com Arena",
        "city": "Los Angeles",
        "timezone": "PT",
        "lat": 34.0430,
        "lon": -118.2673,
    },
    "MIN": {
        "arena_name": "Xcel Energy Center",
        "city": "Saint Paul",
        "timezone": "CT",
        "lat": 44.9448,
        "lon": -93.1011,
    },
    "MTL": {
        "arena_name": "Bell Centre",
        "city": "Montreal",
        "timezone": "ET",
        "lat": 45.4961,
        "lon": -73.5693,
    },
    "NSH": {
        "arena_name": "Bridgestone Arena",
        "city": "Nashville",
        "timezone": "CT",
        "lat": 36.1592,
        "lon": -86.7785,
    },
    "NJ": {
        "arena_name": "Prudential Center",
        "city": "Newark",
        "timezone": "ET",
        "lat": 40.7334,
        "lon": -74.1712,
    },
    "NYI": {
        "arena_name": "UBS Arena",
        "city": "Elmont",
        "timezone": "ET",
        "lat": 40.7172,
        "lon": -73.7256,
    },
    "NYR": {
        "arena_name": "Madison Square Garden",
        "city": "New York",
        "timezone": "ET",
        "lat": 40.7505,
        "lon": -73.9934,
    },
    "OTT": {
        "arena_name": "Canadian Tire Centre",
        "city": "Ottawa",
        "timezone": "ET",
        "lat": 45.2969,
        "lon": -75.9272,
    },
    "PHI": {
        "arena_name": "Wells Fargo Center",
        "city": "Philadelphia",
        "timezone": "ET",
        "lat": 39.9012,
        "lon": -75.1720,
    },
    "PIT": {
        "arena_name": "PPG Paints Arena",
        "city": "Pittsburgh",
        "timezone": "ET",
        "lat": 40.4395,
        "lon": -79.9891,
    },
    "SJ": {
        "arena_name": "SAP Center",
        "city": "San Jose",
        "timezone": "PT",
        "lat": 37.3327,
        "lon": -121.9010,
    },
    "SEA": {
        "arena_name": "Climate Pledge Arena",
        "city": "Seattle",
        "timezone": "PT",
        "lat": 47.6221,
        "lon": -122.3540,
    },
    "STL": {
        "arena_name": "Enterprise Center",
        "city": "Saint Louis",
        "timezone": "CT",
        "lat": 38.6268,
        "lon": -90.2025,
    },
    "TB": {
        "arena_name": "Amalie Arena",
        "city": "Tampa",
        "timezone": "ET",
        "lat": 27.9427,
        "lon": -82.4519,
    },
    "TOR": {
        "arena_name": "Scotiabank Arena",
        "city": "Toronto",
        "timezone": "ET",
        "lat": 43.6435,
        "lon": -79.3791,
    },
    "UTA": {
        "arena_name": "Delta Center",
        "city": "Salt Lake City",
        "timezone": "MT",
        "lat": 40.7683,
        "lon": -111.9011,
    },
    "VAN": {
        "arena_name": "Rogers Arena",
        "city": "Vancouver",
        "timezone": "PT",
        "lat": 49.2778,
        "lon": -123.1089,
    },
    "VGK": {
        "arena_name": "T-Mobile Arena",
        "city": "Las Vegas",
        "timezone": "PT",
        "lat": 36.1029,
        "lon": -115.1785,
    },
    "WPG": {
        "arena_name": "Canada Life Centre",
        "city": "Winnipeg",
        "timezone": "CT",
        "lat": 49.8928,
        "lon": -97.1438,
    },
    "WSH": {
        "arena_name": "Capital One Arena",
        "city": "Washington",
        "timezone": "ET",
        "lat": 38.8981,
        "lon": -77.0209,
    },
}

# Default for unknown teams
_UNKNOWN_ARENA = {
    "arena_name": "Unknown",
    "city": "Unknown",
    "timezone": "ET",
    "lat": 0.0,
    "lon": 0.0,
}

# Timezone offset map (hours west of ET)
_TZ_OFFSETS = {"ET": 0, "CT": 1, "MT": 2, "PT": 3}


# ---------------------------------------------------------------------------
# ESPN team abbreviation mapping
# ESPN uses its own abbreviations; map them to ours
# ---------------------------------------------------------------------------

_ESPN_ABBR_MAP = {
    "ANA": "ANA", "ARI": "ARI", "BOS": "BOS", "BUF": "BUF",
    "CGY": "CGY", "CAR": "CAR", "CHI": "CHI", "COL": "COL",
    "CBJ": "CBJ", "DAL": "DAL", "DET": "DET", "EDM": "EDM",
    "FLA": "FLA", "LA": "LA", "LAK": "LA", "MIN": "MIN",
    "MTL": "MTL", "MON": "MTL", "NSH": "NSH", "NJ": "NJ",
    "NJD": "NJ", "NYI": "NYI", "NYR": "NYR", "OTT": "OTT",
    "PHI": "PHI", "PIT": "PIT", "SJ": "SJ", "SJS": "SJ",
    "SEA": "SEA", "STL": "STL", "TB": "TB", "TBL": "TB",
    "TOR": "TOR", "UTA": "UTA", "UTAH": "UTA", "VAN": "VAN",
    "VGK": "VGK", "WPG": "WPG", "WSH": "WSH",
}


# ---------------------------------------------------------------------------
# Haversine distance (miles)
# ---------------------------------------------------------------------------

def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two lat/lon points in miles."""
    R = 3958.8  # Earth radius in miles
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _timezone_diff(tz1: str, tz2: str) -> int:
    """Return absolute timezone zone difference between two timezones."""
    return abs(_TZ_OFFSETS.get(tz1, 0) - _TZ_OFFSETS.get(tz2, 0))


def get_arena(team_abbr: str) -> dict:
    """Sync lookup for a team's arena data. Returns copy with defaults for unknown teams."""
    return NHL_ARENAS.get(team_abbr, _UNKNOWN_ARENA).copy()


# ---------------------------------------------------------------------------
# ESPN schedule fetching (past 7 days)
# ---------------------------------------------------------------------------

def _normalize_abbr(espn_abbr: str) -> str:
    """Convert ESPN team abbreviation to our standard abbreviation."""
    return _ESPN_ABBR_MAP.get(espn_abbr, espn_abbr)


async def _fetch_scoreboard(date_str: str) -> list:
    """Fetch NHL scoreboard from ESPN for a given date (YYYYMMDD).

    Returns list of game dicts:
        [{"date": "20260403", "away": "PIT", "home": "NYR", "status": "Final"}, ...]

    Cache: 2 hours per date.
    """
    cache_key = f"nhl_scoreboard:{date_str}"
    cached = _get_cached(cache_key, ttl=7200)  # 2 hours
    if cached is not None:
        return cached

    params = {"dates": date_str}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(ESPN_NHL_SCOREBOARD, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NHL TRAVEL] Scoreboard fetch failed for {date_str}: {e}")
        return []

    games = []
    for event in data.get("events", []):
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        if len(competitors) < 2:
            continue

        game = {"date": date_str, "status": event.get("status", {}).get("type", {}).get("name", "")}
        for comp in competitors:
            side = comp.get("homeAway", "")
            abbr = _normalize_abbr(comp.get("team", {}).get("abbreviation", ""))
            if side == "home":
                game["home"] = abbr
            elif side == "away":
                game["away"] = abbr

        if "home" in game and "away" in game:
            games.append(game)

    _set_cache(cache_key, games)
    return games


async def _fetch_recent_schedule(days: int = 7) -> list:
    """Fetch NHL scoreboards for the past N days. Returns flat list of all games."""
    today = datetime.now(timezone.utc)
    all_games = []

    for offset in range(days, 0, -1):
        d = today - timedelta(days=offset)
        date_str = d.strftime("%Y%m%d")
        day_games = await _fetch_scoreboard(date_str)
        all_games.extend(day_games)

    return all_games


# ---------------------------------------------------------------------------
# Team schedule analysis
# ---------------------------------------------------------------------------

def _analyze_team_schedule(team_abbr: str, recent_games: list, today_str: str) -> dict:
    """Analyze a team's recent schedule for rest, B2B, density, travel.

    Args:
        team_abbr: Team abbreviation (e.g. "PIT")
        recent_games: All games from past 7 days (from _fetch_recent_schedule)
        today_str: Today's date as YYYYMMDD

    Returns dict with:
        rest_days, is_b2b, games_in_7d, road_trip_length, total_travel_miles,
        last_venue, tz_changes, tag
    """
    # Filter to games involving this team
    team_games = []
    for g in recent_games:
        if g.get("away") == team_abbr or g.get("home") == team_abbr:
            team_games.append(g)

    # Sort by date
    team_games.sort(key=lambda x: x.get("date", ""))

    games_in_7d = len(team_games)

    # Rest days: days since last game
    if team_games:
        try:
            last_game_date = datetime.strptime(team_games[-1]["date"], "%Y%m%d")
            today_date = datetime.strptime(today_str, "%Y%m%d")
            rest_days = (today_date - last_game_date).days
        except (ValueError, KeyError):
            rest_days = -1
    else:
        rest_days = -1  # No recent games found

    # Back-to-back detection (0 days rest = B2B, played yesterday)
    is_b2b = rest_days == 1

    # Road trip length: count consecutive away games ending with the most recent
    road_trip_length = 0
    for g in reversed(team_games):
        if g.get("away") == team_abbr:
            road_trip_length += 1
        else:
            break

    # Travel distance and timezone changes during road trip
    total_travel_miles = 0.0
    tz_changes = 0
    last_venue_abbr = None

    # Walk through the road trip games to calculate cumulative travel
    road_trip_games = []
    for g in reversed(team_games):
        if g.get("away") == team_abbr:
            road_trip_games.insert(0, g)
        else:
            break

    prev_venue = None
    for g in road_trip_games:
        venue_abbr = g.get("home", "")
        venue = get_arena(venue_abbr)
        if prev_venue and venue["lat"] != 0.0:
            prev = get_arena(prev_venue)
            if prev["lat"] != 0.0:
                dist = _haversine_miles(prev["lat"], prev["lon"], venue["lat"], venue["lon"])
                total_travel_miles += dist
                tz_changes += _timezone_diff(prev["timezone"], venue["timezone"])
        prev_venue = venue_abbr
        last_venue_abbr = venue_abbr

    total_travel_miles = round(total_travel_miles)

    # Tag assignment
    if is_b2b and road_trip_length > 0:
        tag = "FATIGUED"
    elif road_trip_length >= 4:
        tag = "ROAD WARRIOR"
    elif rest_days >= 3:
        tag = "FRESH"
    elif rest_days >= 2:
        tag = "RESTED"
    else:
        tag = "NEUTRAL"

    return {
        "rest_days": rest_days,
        "is_b2b": is_b2b,
        "games_in_7d": games_in_7d,
        "road_trip_length": road_trip_length,
        "total_travel_miles": total_travel_miles,
        "last_venue": last_venue_abbr,
        "tz_changes": tz_changes,
        "tag": tag,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_team_travel(team_abbr: str) -> dict:
    """Fetch travel/schedule analysis for a single team.

    Checks past 7 days of ESPN scoreboards, computes rest, B2B, road trip,
    travel miles, timezone changes.

    Cache: 2 hours (inherited from scoreboard cache)
    """
    today = datetime.now(timezone.utc)
    today_str = today.strftime("%Y%m%d")

    cache_key = f"nhl_travel:{team_abbr}:{today_str}"
    cached = _get_cached(cache_key, ttl=7200)  # 2 hours
    if cached:
        return cached

    recent_games = await _fetch_recent_schedule(days=7)
    result = _analyze_team_schedule(team_abbr, recent_games, today_str)

    _set_cache(cache_key, result)
    return result


async def build_travel_context(games: list) -> str:
    """Build the travel & schedule context block for the AI analysis prompt.

    Args:
        games: List of game dicts with "away" and "home" team abbreviations.
               e.g. [{"away": "PIT", "home": "NYR"}, ...]

    Returns formatted text block:
        === TRAVEL & SCHEDULE ===
        RULE: NHL B2B on the road is the strongest fade signal in hockey ...

          NYR: 2 days rest | NOT B2B | 1 game in 7d | Home — RESTED
          PIT: 0 days rest | B2B (road) | 3 games in 7d | Road trip (3rd game, 1200mi) — FATIGUED
          Travel: PIT traveled 1200mi in 3 days, crossed 1 timezone
          Rest edge: NYR (+2 days, home vs road B2B)
    """
    if not games:
        return "=== TRAVEL & SCHEDULE ===\nNo games provided for travel context."

    lines = ["=== TRAVEL & SCHEDULE ==="]
    lines.append(
        "RULE: NHL B2B on the road is the strongest fade signal in hockey "
        "(covers ~12% less). Cross-country travel (3+ timezone changes) adds fatigue. "
        "Home ice advantage is the largest in pro sports (~55.5% home win rate). "
        "Rest differential > 2 days = significant."
    )
    lines.append("")

    for game in games:
        away_abbr = game.get("away", "?")
        home_abbr = game.get("home", "?")

        away_data = await fetch_team_travel(away_abbr)
        home_data = await fetch_team_travel(home_abbr)

        # Format each team line
        for abbr, data, side in [(home_abbr, home_data, "Home"), (away_abbr, away_data, "Away")]:
            rest = data["rest_days"]
            rest_str = f"{rest} days rest" if rest >= 0 else "rest unknown"

            b2b_str = "B2B (road)" if data["is_b2b"] and data["road_trip_length"] > 0 else (
                "B2B" if data["is_b2b"] else "NOT B2B"
            )

            density_str = f"{data['games_in_7d']} games in 7d"

            if side == "Home":
                location_str = "Home"
            elif data["road_trip_length"] > 0:
                location_str = (
                    f"Road trip ({_ordinal(data['road_trip_length'])} game, "
                    f"{data['total_travel_miles']}mi traveled)"
                )
            else:
                location_str = "Away"

            tag = data["tag"]

            lines.append(
                f"  {abbr}: {rest_str} | {b2b_str} | {density_str} | "
                f"{location_str} — {tag}"
            )

        # Travel summary for away team
        away_d = away_data
        if away_d["road_trip_length"] > 0 and away_d["total_travel_miles"] > 0:
            lines.append(
                f"  Travel: {away_abbr} traveled {away_d['total_travel_miles']}mi "
                f"in {away_d['road_trip_length']} games, "
                f"crossed {away_d['tz_changes']} timezone(s)"
            )

        # Rest edge
        home_rest = home_data["rest_days"]
        away_rest = away_data["rest_days"]
        if home_rest >= 0 and away_rest >= 0:
            diff = home_rest - away_rest
            if diff != 0:
                advantage_team = home_abbr if diff > 0 else away_abbr
                advantage_side = "home" if diff > 0 else "away"

                # Build context for the rest edge
                edge_parts = [f"+{abs(diff)} days"]
                if home_data["is_b2b"] or away_data["is_b2b"]:
                    b2b_team = home_abbr if home_data["is_b2b"] else away_abbr
                    if advantage_team != b2b_team:
                        edge_parts.append(f"{advantage_side} vs {'road ' if away_data['road_trip_length'] > 0 else ''}B2B")

                lines.append(
                    f"  Rest edge: {advantage_team} ({', '.join(edge_parts)})"
                )

        lines.append("")

    return "\n".join(lines)


def _ordinal(n: int) -> str:
    """Return ordinal string for an integer (1st, 2nd, 3rd, etc.)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

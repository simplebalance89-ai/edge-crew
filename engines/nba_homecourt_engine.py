"""
NBA Home Court Advantage Engine — Static arena data + home/road record splits.

Data source: Scores archive (historical completed games already loaded in platform).
Arena data is static. No external API key required.

Used by: server.py analysis prompt (home court context injection)
Cache: 6 hours for home/road records
"""

import logging
from datetime import datetime, timezone

from engines.mlb_pitcher_engine import _get_cached, _set_cache

logger = logging.getLogger("edge-crew")

# ---------------------------------------------------------------------------
# NBA ARENAS — All 30 teams (2024-25 season)
#
# altitude_ft: Elevation above sea level. DEN (5280) and UTA (4226) are the
#              only high-altitude venues; all others are effectively sea level.
# ---------------------------------------------------------------------------

NBA_ARENAS = {
    "ATL": {
        "arena_name": "State Farm Arena",
        "city": "Atlanta",
        "altitude_ft": 1050,
        "lat": 33.7573,
        "lon": -84.3963,
    },
    "BOS": {
        "arena_name": "TD Garden",
        "city": "Boston",
        "altitude_ft": 20,
        "lat": 42.3662,
        "lon": -71.0621,
    },
    "BKN": {
        "arena_name": "Barclays Center",
        "city": "Brooklyn",
        "altitude_ft": 30,
        "lat": 40.6826,
        "lon": -73.9754,
    },
    "CHA": {
        "arena_name": "Spectrum Center",
        "city": "Charlotte",
        "altitude_ft": 751,
        "lat": 35.2251,
        "lon": -80.8392,
    },
    "CHI": {
        "arena_name": "United Center",
        "city": "Chicago",
        "altitude_ft": 594,
        "lat": 41.8807,
        "lon": -87.6742,
    },
    "CLE": {
        "arena_name": "Rocket Mortgage FieldHouse",
        "city": "Cleveland",
        "altitude_ft": 653,
        "lat": 41.4965,
        "lon": -81.6882,
    },
    "DAL": {
        "arena_name": "American Airlines Center",
        "city": "Dallas",
        "altitude_ft": 430,
        "lat": 32.7905,
        "lon": -96.8103,
    },
    "DEN": {
        "arena_name": "Ball Arena",
        "city": "Denver",
        "altitude_ft": 5280,
        "lat": 39.7487,
        "lon": -105.0077,
    },
    "DET": {
        "arena_name": "Little Caesars Arena",
        "city": "Detroit",
        "altitude_ft": 600,
        "lat": 42.3411,
        "lon": -83.0553,
    },
    "GSW": {
        "arena_name": "Chase Center",
        "city": "San Francisco",
        "altitude_ft": 5,
        "lat": 37.7680,
        "lon": -122.3877,
    },
    "HOU": {
        "arena_name": "Toyota Center",
        "city": "Houston",
        "altitude_ft": 42,
        "lat": 29.7508,
        "lon": -95.3621,
    },
    "IND": {
        "arena_name": "Gainbridge Fieldhouse",
        "city": "Indianapolis",
        "altitude_ft": 715,
        "lat": 39.7640,
        "lon": -86.1555,
    },
    "LAC": {
        "arena_name": "Intuit Dome",
        "city": "Inglewood",
        "altitude_ft": 110,
        "lat": 33.9442,
        "lon": -118.3417,
    },
    "LAL": {
        "arena_name": "Crypto.com Arena",
        "city": "Los Angeles",
        "altitude_ft": 340,
        "lat": 34.0430,
        "lon": -118.2673,
    },
    "MEM": {
        "arena_name": "FedExForum",
        "city": "Memphis",
        "altitude_ft": 337,
        "lat": 35.1382,
        "lon": -90.0506,
    },
    "MIA": {
        "arena_name": "Kaseya Center",
        "city": "Miami",
        "altitude_ft": 7,
        "lat": 25.7814,
        "lon": -80.1870,
    },
    "MIL": {
        "arena_name": "Fiserv Forum",
        "city": "Milwaukee",
        "altitude_ft": 635,
        "lat": 43.0451,
        "lon": -87.9174,
    },
    "MIN": {
        "arena_name": "Target Center",
        "city": "Minneapolis",
        "altitude_ft": 815,
        "lat": 44.9795,
        "lon": -93.2761,
    },
    "NOP": {
        "arena_name": "Smoothie King Center",
        "city": "New Orleans",
        "altitude_ft": 3,
        "lat": 29.9490,
        "lon": -90.0821,
    },
    "NYK": {
        "arena_name": "Madison Square Garden",
        "city": "New York",
        "altitude_ft": 33,
        "lat": 40.7505,
        "lon": -73.9934,
    },
    "OKC": {
        "arena_name": "Paycom Center",
        "city": "Oklahoma City",
        "altitude_ft": 1201,
        "lat": 35.4634,
        "lon": -97.5151,
    },
    "ORL": {
        "arena_name": "Kia Center",
        "city": "Orlando",
        "altitude_ft": 82,
        "lat": 28.5392,
        "lon": -81.3839,
    },
    "PHI": {
        "arena_name": "Wells Fargo Center",
        "city": "Philadelphia",
        "altitude_ft": 20,
        "lat": 39.9012,
        "lon": -75.1720,
    },
    "PHX": {
        "arena_name": "Footprint Center",
        "city": "Phoenix",
        "altitude_ft": 1086,
        "lat": 33.4457,
        "lon": -112.0712,
    },
    "POR": {
        "arena_name": "Moda Center",
        "city": "Portland",
        "altitude_ft": 50,
        "lat": 45.5316,
        "lon": -122.6668,
    },
    "SAC": {
        "arena_name": "Golden 1 Center",
        "city": "Sacramento",
        "altitude_ft": 30,
        "lat": 38.5802,
        "lon": -121.4997,
    },
    "SAS": {
        "arena_name": "Frost Bank Center",
        "city": "San Antonio",
        "altitude_ft": 650,
        "lat": 29.4270,
        "lon": -98.4375,
    },
    "TOR": {
        "arena_name": "Scotiabank Arena",
        "city": "Toronto",
        "altitude_ft": 249,
        "lat": 43.6435,
        "lon": -79.3791,
    },
    "UTA": {
        "arena_name": "Delta Center",
        "city": "Salt Lake City",
        "altitude_ft": 4226,
        "lat": 40.7683,
        "lon": -111.9011,
    },
    "WAS": {
        "arena_name": "Capital One Arena",
        "city": "Washington",
        "altitude_ft": 25,
        "lat": 38.8981,
        "lon": -77.0209,
    },
}

# Default for unknown teams
_UNKNOWN_ARENA = {
    "arena_name": "Unknown",
    "city": "Unknown",
    "altitude_ft": 0,
    "lat": 0.0,
    "lon": 0.0,
}

# High-altitude threshold — venues above this get the altitude edge tag
_ALTITUDE_THRESHOLD_FT = 3500


def get_arena(team_abbr: str) -> dict:
    """Sync lookup for a team's arena data.

    Returns full arena dict with arena_name, city, altitude_ft, lat, lon.
    Returns defaults for unknown teams.
    """
    return NBA_ARENAS.get(team_abbr, _UNKNOWN_ARENA).copy()


def _is_altitude_venue(team_abbr: str) -> bool:
    """Return True if the team plays at a high-altitude arena (DEN, UTA)."""
    arena = NBA_ARENAS.get(team_abbr, _UNKNOWN_ARENA)
    return arena["altitude_ft"] >= _ALTITUDE_THRESHOLD_FT


def _home_tag(win_pct: float) -> str:
    """Return a descriptive tag for home court performance."""
    if win_pct >= 0.80:
        return "ELITE HOME"
    elif win_pct >= 0.70:
        return "STRONG HOME"
    elif win_pct >= 0.50:
        return "AVERAGE HOME"
    else:
        return "WEAK HOME"


def _fmt_pct(pct: float) -> str:
    """Format a win percentage as .XXX (e.g. 0.733 -> '.733')."""
    return f"{pct:.3f}"


def _road_tag(win_pct: float) -> str:
    """Return a descriptive tag for road performance."""
    if win_pct >= 0.60:
        return "STRONG ROAD"
    elif win_pct >= 0.40:
        return "AVERAGE ROAD"
    else:
        return "WEAK ROAD"


async def fetch_home_road_records(team_abbr: str, games_archive: list = None) -> dict:
    """Compute home and road records from the scores archive.

    Args:
        team_abbr: NBA team abbreviation (e.g. "LAL", "BOS").
        games_archive: List of completed game dicts from the scores archive.
            Each dict should have: "home", "away", "home_score", "away_score".
            If None, returns zeroed-out placeholder.

    Returns:
        {
            "home_w": 22, "home_l": 8, "home_pct": 0.733,
            "road_w": 15, "road_l": 16, "road_pct": 0.484,
            "l10_home_ats_w": 7, "l10_home_ats_l": 3,
            "l10_road_ats_w": 4, "l10_road_ats_l": 6,
        }

    Cache: 6 hours
    """
    cache_key = f"nba_home_road:{team_abbr}"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    result = {
        "home_w": 0, "home_l": 0, "home_pct": 0.0,
        "road_w": 0, "road_l": 0, "road_pct": 0.0,
        "l10_home_ats_w": 0, "l10_home_ats_l": 0,
        "l10_road_ats_w": 0, "l10_road_ats_l": 0,
    }

    if not games_archive:
        return result

    home_games = []
    road_games = []

    for g in games_archive:
        home_team = g.get("home", "")
        away_team = g.get("away", "")
        home_score = g.get("home_score")
        away_score = g.get("away_score")

        if home_score is None or away_score is None:
            continue

        try:
            h_score = int(home_score)
            a_score = int(away_score)
        except (ValueError, TypeError):
            continue

        if home_team == team_abbr:
            won = h_score > a_score
            home_games.append({
                "won": won,
                "spread_result": g.get("spread_result"),  # "W" or "L" ATS
            })
            if won:
                result["home_w"] += 1
            else:
                result["home_l"] += 1

        elif away_team == team_abbr:
            won = a_score > h_score
            road_games.append({
                "won": won,
                "spread_result": g.get("spread_result"),
            })
            if won:
                result["road_w"] += 1
            else:
                result["road_l"] += 1

    # Win percentages
    home_total = result["home_w"] + result["home_l"]
    road_total = result["road_w"] + result["road_l"]
    result["home_pct"] = result["home_w"] / home_total if home_total > 0 else 0.0
    result["road_pct"] = result["road_w"] / road_total if road_total > 0 else 0.0

    # Last 10 home ATS
    recent_home = home_games[-10:] if home_games else []
    for g in recent_home:
        sr = g.get("spread_result", "")
        if sr == "W":
            result["l10_home_ats_w"] += 1
        elif sr == "L":
            result["l10_home_ats_l"] += 1

    # Last 10 road ATS
    recent_road = road_games[-10:] if road_games else []
    for g in recent_road:
        sr = g.get("spread_result", "")
        if sr == "W":
            result["l10_road_ats_w"] += 1
        elif sr == "L":
            result["l10_road_ats_l"] += 1

    _set_cache(cache_key, result)
    return result


async def build_homecourt_context(games: list, games_archive: list = None) -> str:
    """Build the home court advantage context block for the AI analysis prompt.

    Args:
        games: List of today's game dicts with "away" and "home" team abbreviations.
               e.g. [{"away": "BOS", "home": "LAL"}, {"away": "GSW", "home": "DEN"}]
        games_archive: List of completed game dicts from scores archive for record
                       computation. If None, only static arena data is shown.

    Returns formatted text block:
        === HOME COURT ===
        RULE: NBA home teams win ~58% overall. ...

          LAL (home): 22-8 at home (.733) | L10 home: 7-3 ATS -- STRONG HOME
          BOS (away): 15-16 on road (.484) | L10 road: 4-6 ATS -- AVERAGE ROAD
          Altitude: N/A (LAL at 340ft)
          Home edge: LAL strong home (.733) vs BOS average road (.484)
    """
    if not games:
        return "=== HOME COURT ===\nNo games provided for home court context."

    lines = ["=== HOME COURT ==="]
    lines.append(
        "RULE: NBA home teams win ~58% overall. Denver altitude (5280ft) adds 3-5% "
        "visitor disadvantage. Strong home record (70%+ at home) vs poor road record "
        "(< 40% on road) = score home_away 8-9. Altitude games (DEN/UTA) with "
        "sea-level visitors = additional +1."
    )
    lines.append("")

    for game in games:
        home_abbr = game.get("home", "?")
        away_abbr = game.get("away", "?")

        home_arena = get_arena(home_abbr)
        arena_name = home_arena["arena_name"]
        altitude = home_arena["altitude_ft"]

        home_rec = await fetch_home_road_records(home_abbr, games_archive)
        away_rec = await fetch_home_road_records(away_abbr, games_archive)

        # Home team line
        h_w = home_rec["home_w"]
        h_l = home_rec["home_l"]
        h_pct = home_rec["home_pct"]
        h_ats_w = home_rec["l10_home_ats_w"]
        h_ats_l = home_rec["l10_home_ats_l"]
        h_tag = _home_tag(h_pct)

        # Away team line
        a_w = away_rec["road_w"]
        a_l = away_rec["road_l"]
        a_pct = away_rec["road_pct"]
        a_ats_w = away_rec["l10_road_ats_w"]
        a_ats_l = away_rec["l10_road_ats_l"]
        a_tag = _road_tag(a_pct)

        lines.append(f"  {away_abbr} @ {home_abbr} ({arena_name}):")
        lines.append(
            f"  {home_abbr} (home): {h_w}-{h_l} at home ({_fmt_pct(h_pct)}) "
            f"| L10 home: {h_ats_w}-{h_ats_l} ATS -- {h_tag}"
        )
        lines.append(
            f"  {away_abbr} (away): {a_w}-{a_l} on road ({_fmt_pct(a_pct)}) "
            f"| L10 road: {a_ats_w}-{a_ats_l} ATS -- {a_tag}"
        )

        # Altitude check
        if _is_altitude_venue(home_abbr) and not _is_altitude_venue(away_abbr):
            lines.append(
                f"  Altitude: ALTITUDE EDGE -- {home_abbr} at {altitude}ft, "
                f"{away_abbr} is a sea-level team"
            )
        elif _is_altitude_venue(home_abbr):
            lines.append(
                f"  Altitude: Both teams are high-altitude ({home_abbr} at {altitude}ft)"
            )
        else:
            lines.append(f"  Altitude: N/A ({home_abbr} at {altitude}ft)")

        # Summary edge line
        lines.append(
            f"  Home edge: {home_abbr} {h_tag.lower()} ({_fmt_pct(h_pct)}) "
            f"vs {away_abbr} {a_tag.lower()} ({_fmt_pct(a_pct)})"
        )
        lines.append("")

    return "\n".join(lines)

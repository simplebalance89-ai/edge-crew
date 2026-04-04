"""
Soccer Home/Away & Venue Engine — Home/away splits + altitude/pitch factors.

ESPN Standings API: https://site.api.espn.com/apis/v2/sports/soccer/{league}/standings
Free, public, no API key required.

Used by: server.py analysis prompt (soccer home/away context injection)
Cache: 6 hours for standings/splits data
"""

import httpx
import logging
from datetime import datetime, timezone

from engines.mlb_pitcher_engine import _get_cached, _set_cache

logger = logging.getLogger("edge-crew")

ESPN_SOCCER_STANDINGS = "https://site.api.espn.com/apis/v2/sports/soccer/{league}/standings"

# ---------------------------------------------------------------------------
# SUPPORTED LEAGUES — ESPN slug mapping
# ---------------------------------------------------------------------------

SOCCER_LEAGUES = {
    "epl": "eng.1",          # English Premier League
    "eng.1": "eng.1",
    "la_liga": "esp.1",      # La Liga
    "esp.1": "esp.1",
    "bundesliga": "ger.1",   # Bundesliga
    "ger.1": "ger.1",
    "serie_a": "ita.1",      # Serie A
    "ita.1": "ita.1",
    "ligue_1": "fra.1",      # Ligue 1
    "fra.1": "fra.1",
    "mls": "usa.1",          # MLS
    "usa.1": "usa.1",
    "liga_mx": "mex.1",      # Liga MX
    "mex.1": "mex.1",
    "eredivisie": "ned.1",   # Eredivisie
    "ned.1": "ned.1",
    "primeira_liga": "por.1",  # Primeira Liga
    "por.1": "por.1",
    "spl": "sco.1",          # Scottish Premiership
    "sco.1": "sco.1",
    "copa_libertadores": "conmebol.libertadores",
    "conmebol.libertadores": "conmebol.libertadores",
    "allsvenskan": "swe.1",  # Allsvenskan (artificial turf)
    "swe.1": "swe.1",
}

# ---------------------------------------------------------------------------
# ALTITUDE VENUES — Stadiums above 3000ft (elevation in feet)
#
# Altitude impacts match fitness, ball flight, and visiting team performance.
# Goals scored 15-20% higher at altitude venues vs sea-level visitors.
# ---------------------------------------------------------------------------

ALTITUDE_VENUES = {
    # Liga MX — Mexico City metro (7350ft)
    "Club America": {"city": "Mexico City", "altitude_ft": 7350, "country": "MEX"},
    "Cruz Azul": {"city": "Mexico City", "altitude_ft": 7350, "country": "MEX"},
    "Pumas UNAM": {"city": "Mexico City", "altitude_ft": 7350, "country": "MEX"},
    "Club Universidad Nacional": {"city": "Mexico City", "altitude_ft": 7350, "country": "MEX"},
    # Liga MX — other altitude venues
    "Toluca": {"city": "Toluca", "altitude_ft": 8750, "country": "MEX"},
    "Club Atletico San Luis": {"city": "San Luis Potosi", "altitude_ft": 6070, "country": "MEX"},
    "Pachuca": {"city": "Pachuca", "altitude_ft": 7960, "country": "MEX"},
    "Leon": {"city": "Leon", "altitude_ft": 5955, "country": "MEX"},
    "Puebla": {"city": "Puebla", "altitude_ft": 7005, "country": "MEX"},
    # Colombia
    "Millonarios": {"city": "Bogota", "altitude_ft": 8660, "country": "COL"},
    "Independiente Santa Fe": {"city": "Bogota", "altitude_ft": 8660, "country": "COL"},
    "Santa Fe": {"city": "Bogota", "altitude_ft": 8660, "country": "COL"},
    # Bolivia
    "Bolivar": {"city": "La Paz", "altitude_ft": 11900, "country": "BOL"},
    "The Strongest": {"city": "La Paz", "altitude_ft": 11900, "country": "BOL"},
    "Always Ready": {"city": "El Alto", "altitude_ft": 13325, "country": "BOL"},
    # Ecuador
    "LDU Quito": {"city": "Quito", "altitude_ft": 9350, "country": "ECU"},
    "Liga de Quito": {"city": "Quito", "altitude_ft": 9350, "country": "ECU"},
    "Aucas": {"city": "Quito", "altitude_ft": 9350, "country": "ECU"},
    "Independiente del Valle": {"city": "Sangolqui", "altitude_ft": 8200, "country": "ECU"},
    # MLS — altitude venues
    "Colorado Rapids": {"city": "Denver", "altitude_ft": 5280, "country": "USA"},
    "Real Salt Lake": {"city": "Salt Lake City", "altitude_ft": 4226, "country": "USA"},
    # MLS — dome (altitude negligible but enclosed)
    "Atlanta United FC": {"city": "Atlanta", "altitude_ft": 1050, "country": "USA", "dome": True},
    "Atlanta United": {"city": "Atlanta", "altitude_ft": 1050, "country": "USA", "dome": True},
    # Peru
    "Sporting Cristal": {"city": "Lima", "altitude_ft": 500, "country": "PER"},
    "Cienciano": {"city": "Cusco", "altitude_ft": 11150, "country": "PER"},
}

# Altitude threshold for "significant advantage"
_ALTITUDE_THRESHOLD_FT = 3000

# ---------------------------------------------------------------------------
# ARTIFICIAL TURF VENUES — Matters in MLS, Scandinavian, and Scottish leagues
#
# Artificial turf changes ball speed, bounce height, and injury risk.
# Home teams adapted to turf have a measurable edge vs grass-only visitors.
# ---------------------------------------------------------------------------

TURF_VENUES = {
    # MLS
    "Atlanta United FC": "artificial",
    "Atlanta United": "artificial",
    "Portland Timbers": "artificial",
    "Seattle Sounders FC": "artificial",
    "Seattle Sounders": "artificial",
    "New England Revolution": "artificial",
    "Vancouver Whitecaps FC": "artificial",
    "Vancouver Whitecaps": "artificial",
    # Allsvenskan (Sweden)
    "AIK": "artificial",
    "Hammarby": "artificial",
    "IF Elfsborg": "artificial",
    # Scottish Premiership
    "Kilmarnock": "artificial",
    "Livingston": "artificial",
    "Ross County": "artificial",
}

# Default pitch type for teams not in TURF_VENUES
_DEFAULT_PITCH = "natural grass"


def _get_pitch_type(team_name: str) -> str:
    """Return pitch type for a team (artificial or natural grass)."""
    return TURF_VENUES.get(team_name, _DEFAULT_PITCH)


def _get_altitude(team_name: str) -> dict:
    """Return altitude info for a team if they play at an altitude venue.

    Returns: {"city": "...", "altitude_ft": N, "country": "..."} or None.
    """
    return ALTITUDE_VENUES.get(team_name)


def _resolve_league(league_key: str) -> str:
    """Resolve a league alias to the ESPN slug.

    Returns the ESPN league slug (e.g. 'eng.1') or the original key if not found.
    """
    return SOCCER_LEAGUES.get(league_key.lower(), league_key)


# ---------------------------------------------------------------------------
# ESPN Standings Fetch — home/away splits
# ---------------------------------------------------------------------------

async def fetch_league_standings(league: str) -> list:
    """Fetch league standings from ESPN, returning per-team records.

    Args:
        league: ESPN league slug or alias (e.g. 'epl', 'eng.1', 'mls').

    Returns list of dicts:
        [{"team_name": "Arsenal", "team_abbr": "ARS",
          "overall": {"w": 20, "d": 5, "l": 3, "gf": 65, "ga": 25, "gp": 28},
          "home": {"w": 13, "d": 2, "l": 0, "gf": 42, "ga": 8, "gp": 15},
          "away": {"w": 7, "d": 3, "l": 3, "gf": 23, "ga": 17, "gp": 13}}, ...]

    Cache: 6 hours.
    """
    espn_slug = _resolve_league(league)
    cache_key = f"soccer_standings:{espn_slug}"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    url = ESPN_SOCCER_STANDINGS.format(league=espn_slug)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[SOCCER VENUE] Standings fetch failed for {espn_slug}: {e}")
        return []

    teams = []
    for child in data.get("children", [{}]):
        for entry in child.get("standings", {}).get("entries", []):
            teams.append(_parse_standing_entry(entry))

    # Fallback: some leagues use top-level standings (no children wrapper)
    if not teams:
        for entry in data.get("standings", {}).get("entries", []):
            teams.append(_parse_standing_entry(entry))

    _set_cache(cache_key, teams)
    logger.info(f"[SOCCER VENUE] Fetched {len(teams)} teams for {espn_slug}")
    return teams


def _parse_standing_entry(entry: dict) -> dict:
    """Parse a single ESPN standings entry into our standard format."""
    team_info = entry.get("team", {})
    team_name = team_info.get("displayName", team_info.get("name", "Unknown"))
    team_abbr = team_info.get("abbreviation", "???")

    stats_map = {}
    for stat in entry.get("stats", []):
        stats_map[stat.get("name", "")] = stat.get("value", 0)

    # ESPN stat names vary by league, but common ones:
    overall_w = int(stats_map.get("wins", 0))
    overall_d = int(stats_map.get("ties", stats_map.get("draws", 0)))
    overall_l = int(stats_map.get("losses", 0))
    overall_gf = int(stats_map.get("pointsFor", stats_map.get("goalsFor", 0)))
    overall_ga = int(stats_map.get("pointsAgainst", stats_map.get("goalsAgainst", 0)))
    overall_gp = int(stats_map.get("gamesPlayed", overall_w + overall_d + overall_l))

    # Home splits (ESPN sometimes includes these)
    home_w = int(stats_map.get("homeWins", 0))
    home_d = int(stats_map.get("homeTies", stats_map.get("homeDraws", 0)))
    home_l = int(stats_map.get("homeLosses", 0))
    home_gf = int(stats_map.get("homePointsFor", stats_map.get("homeGoalsFor", 0)))
    home_ga = int(stats_map.get("homePointsAgainst", stats_map.get("homeGoalsAgainst", 0)))
    home_gp = home_w + home_d + home_l

    # Away splits
    away_w = int(stats_map.get("awayWins", 0))
    away_d = int(stats_map.get("awayTies", stats_map.get("awayDraws", 0)))
    away_l = int(stats_map.get("awayLosses", 0))
    away_gf = int(stats_map.get("awayPointsFor", stats_map.get("awayGoalsFor", 0)))
    away_ga = int(stats_map.get("awayPointsAgainst", stats_map.get("awayGoalsAgainst", 0)))
    away_gp = away_w + away_d + away_l

    # If ESPN doesn't provide home/away splits, estimate from overall
    # (half home, half away — crude but better than nothing)
    if home_gp == 0 and away_gp == 0 and overall_gp > 0:
        home_gp = overall_gp // 2 or 1
        away_gp = overall_gp - home_gp or 1
        home_w = overall_w // 2
        away_w = overall_w - home_w
        home_d = overall_d // 2
        away_d = overall_d - home_d
        home_l = overall_l // 2
        away_l = overall_l - home_l
        home_gf = overall_gf // 2
        away_gf = overall_gf - home_gf
        home_ga = overall_ga // 2
        away_ga = overall_ga - home_ga

    return {
        "team_name": team_name,
        "team_abbr": team_abbr,
        "overall": {
            "w": overall_w, "d": overall_d, "l": overall_l,
            "gf": overall_gf, "ga": overall_ga, "gp": overall_gp,
        },
        "home": {
            "w": home_w, "d": home_d, "l": home_l,
            "gf": home_gf, "ga": home_ga, "gp": home_gp,
        },
        "away": {
            "w": away_w, "d": away_d, "l": away_l,
            "gf": away_gf, "ga": away_ga, "gp": away_gp,
        },
    }


# ---------------------------------------------------------------------------
# Team lookup helpers
# ---------------------------------------------------------------------------

def _find_team(standings: list, team_name: str) -> dict | None:
    """Find a team in standings by display name or abbreviation (case-insensitive).

    Tries exact match first, then substring/abbreviation match.
    """
    name_lower = team_name.lower().strip()

    # Pass 1: exact match on name or abbreviation
    for t in standings:
        if t["team_name"].lower() == name_lower:
            return t
        if t["team_abbr"].lower() == name_lower:
            return t

    # Pass 2: substring match (e.g. "Arsenal" matches "Arsenal FC")
    for t in standings:
        if name_lower in t["team_name"].lower() or t["team_name"].lower() in name_lower:
            return t

    return None


# ---------------------------------------------------------------------------
# Stat computation helpers
# ---------------------------------------------------------------------------

def _win_pct(record: dict) -> float:
    """Compute win percentage from a W-D-L record (draws count as 0.5 for pct)."""
    gp = record.get("gp", 0)
    if gp == 0:
        return 0.0
    # In soccer, "win %" for betting context: wins / games played
    return record["w"] / gp


def _goals_per_game(record: dict) -> float:
    """Compute goals scored per game."""
    gp = record.get("gp", 0)
    if gp == 0:
        return 0.0
    return record["gf"] / gp


def _clean_sheet_pct(record: dict) -> float:
    """Estimate clean sheet percentage from goals against.

    Clean sheets aren't directly in standings, so we estimate:
    if GA/GP < 0.5 -> ~50%+ clean sheets, etc.
    This is a rough proxy; actual clean sheet data would come from a
    detailed match-by-match source.

    Returns a float 0.0-1.0.
    """
    gp = record.get("gp", 0)
    ga = record.get("ga", 0)
    if gp == 0:
        return 0.0
    ga_per_game = ga / gp
    # Heuristic: CS% ~ max(0, 1 - ga_per_game) capped at 0.6
    # A team conceding 0.5 GPG has ~50% CS, conceding 1.5 has ~10%, etc.
    if ga_per_game <= 0.3:
        return 0.55
    elif ga_per_game <= 0.6:
        return 0.45
    elif ga_per_game <= 0.9:
        return 0.35
    elif ga_per_game <= 1.2:
        return 0.25
    elif ga_per_game <= 1.5:
        return 0.15
    elif ga_per_game <= 2.0:
        return 0.08
    else:
        return 0.03


def _home_tag(win_pct: float) -> str:
    """Return a descriptive tag for home performance."""
    if win_pct >= 0.80:
        return "FORTRESS HOME"
    elif win_pct >= 0.65:
        return "STRONG HOME"
    elif win_pct >= 0.50:
        return "AVERAGE HOME"
    else:
        return "WEAK HOME"


def _road_tag(win_pct: float) -> str:
    """Return a descriptive tag for away performance."""
    if win_pct >= 0.50:
        return "STRONG ROAD"
    elif win_pct >= 0.30:
        return "AVERAGE ROAD"
    else:
        return "WEAK ROAD"


def _fmt_pct(pct: float) -> str:
    """Format a win percentage as .XXX (e.g. 0.733 -> '.733')."""
    return f"{pct:.3f}"


def _fmt_record(record: dict) -> str:
    """Format a W-D-L record as 'XW-YD-ZL'."""
    return f"{record['w']}W-{record['d']}D-{record['l']}L"


# ---------------------------------------------------------------------------
# Context builder — main entry point
# ---------------------------------------------------------------------------

async def build_soccer_venue_context(games: list, league: str = "epl") -> str:
    """Build the home/away & venue context block for the AI analysis prompt.

    Args:
        games: List of match dicts with "home" and "away" team names.
               e.g. [{"home": "Arsenal", "away": "Fulham"}, ...]
        league: ESPN league slug or alias (e.g. 'epl', 'mls', 'liga_mx').

    Returns formatted text block:
        === HOME & AWAY ===
        RULE: Soccer home advantage is ~46% win rate ...

          Arsenal (home): 13W-2D-0L at home (.867) | 2.8 home GPG | 40% home CS -- FORTRESS HOME
          Fulham (away): 4W-4D-7L away (.267) | 0.9 away GPG | 13% away CS -- WEAK ROAD
          Home edge: Arsenal fortress home (.867) vs Fulham weak road (.267)
          Altitude: N/A (London, sea level)
    """
    if not games:
        return "=== HOME & AWAY ===\nNo games provided for home/away context."

    standings = await fetch_league_standings(league)

    lines = ["=== HOME & AWAY ==="]
    lines.append(
        "RULE: Soccer home advantage is ~46% win rate (strongest in European football). "
        "Home team with 75%+ home win rate vs away team with < 30% away win rate = "
        "strong home edge, score 8-9. Altitude > 5000ft with sea-level visitors = "
        "significant advantage (goals scored 15-20% higher at altitude)."
    )
    lines.append("")

    for game in games:
        home_name = game.get("home", "?")
        away_name = game.get("away", "?")

        home_data = _find_team(standings, home_name) if standings else None
        away_data = _find_team(standings, away_name) if standings else None

        # --- Home team line ---
        if home_data:
            h_rec = home_data["home"]
            h_pct = _win_pct(h_rec)
            h_gpg = _goals_per_game(h_rec)
            h_cs = _clean_sheet_pct(h_rec)
            h_tag = _home_tag(h_pct)
            h_display = home_data["team_name"]

            lines.append(
                f"  {h_display} (home): {_fmt_record(h_rec)} at home ({_fmt_pct(h_pct)}) "
                f"| {h_gpg:.1f} home GPG | {h_cs:.0%} home CS -- {h_tag}"
            )
        else:
            h_pct = 0.0
            h_tag = "UNKNOWN"
            h_display = home_name
            lines.append(f"  {home_name} (home): standings data unavailable")

        # --- Away team line ---
        if away_data:
            a_rec = away_data["away"]
            a_pct = _win_pct(a_rec)
            a_gpg = _goals_per_game(a_rec)
            a_cs = _clean_sheet_pct(a_rec)
            a_tag = _road_tag(a_pct)
            a_display = away_data["team_name"]

            lines.append(
                f"  {a_display} (away): {_fmt_record(a_rec)} away ({_fmt_pct(a_pct)}) "
                f"| {a_gpg:.1f} away GPG | {a_cs:.0%} away CS -- {a_tag}"
            )
        else:
            a_pct = 0.0
            a_tag = "UNKNOWN"
            a_display = away_name
            lines.append(f"  {away_name} (away): standings data unavailable")

        # --- Home edge summary ---
        if home_data and away_data:
            lines.append(
                f"  Home edge: {h_display} {h_tag.lower()} ({_fmt_pct(h_pct)}) "
                f"vs {a_display} {a_tag.lower()} ({_fmt_pct(a_pct)})"
            )

        # --- Altitude check ---
        home_alt = _get_altitude(home_name) or _get_altitude(h_display if home_data else "")
        away_alt = _get_altitude(away_name) or _get_altitude(a_display if away_data else "")

        if home_alt and home_alt["altitude_ft"] >= _ALTITUDE_THRESHOLD_FT:
            h_alt_ft = home_alt["altitude_ft"]
            h_city = home_alt["city"]
            if away_alt and away_alt["altitude_ft"] >= _ALTITUDE_THRESHOLD_FT:
                lines.append(
                    f"  Altitude: Both teams at altitude "
                    f"({h_city} {h_alt_ft}ft vs {away_alt['city']} {away_alt['altitude_ft']}ft)"
                )
            else:
                lines.append(
                    f"  Altitude: ALTITUDE EDGE -- {h_display} at {h_city} ({h_alt_ft}ft), "
                    f"visitors from sea level"
                )
        else:
            lines.append(f"  Altitude: N/A (sea level)")

        # --- Pitch type check ---
        home_pitch = _get_pitch_type(home_name)
        if home_data:
            home_pitch = _get_pitch_type(home_data["team_name"]) if home_pitch == _DEFAULT_PITCH else home_pitch
        away_pitch = _get_pitch_type(away_name)
        if away_data:
            away_pitch = _get_pitch_type(away_data["team_name"]) if away_pitch == _DEFAULT_PITCH else away_pitch

        if home_pitch == "artificial" and away_pitch != "artificial":
            lines.append(
                f"  Pitch: TURF EDGE -- {h_display} plays on artificial turf, "
                f"{a_display} adapted to natural grass"
            )
        elif home_pitch == "artificial":
            lines.append(f"  Pitch: Both teams on artificial turf")
        else:
            lines.append(f"  Pitch: Natural grass")

        lines.append("")

    return "\n".join(lines)

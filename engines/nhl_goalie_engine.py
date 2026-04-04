"""
NHL Goalie Engine — Starting goalie matchups + goalie stats for NHL games.

NHL Stats API: https://api-web.nhle.com/v1/
Free, public, no API key required.

Fallback: ESPN scoreboard for game data.
ESPN NHL: https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard

Used by: server.py analysis prompt (goalie context injection)
Cache: 30 min for goalie confirmation (volatile), 6 hr for season stats (stable)
"""

import httpx
import logging
from datetime import datetime, timedelta, timezone

from engines.mlb_pitcher_engine import _get_cached, _set_cache

logger = logging.getLogger("edge-crew")

NHL_API_BASE = "https://api-web.nhle.com/v1"
ESPN_NHL_BASE = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"

# All 32 NHL teams: full name -> 3-letter abbreviation
NHL_TEAM_ABBREVS = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}

# Reverse lookup: abbreviation -> full name
NHL_ABBREV_TO_NAME = {v: k for k, v in NHL_TEAM_ABBREVS.items()}

# NHL team 3-letter codes used by the NHL API (triCode)
NHL_TRI_CODES = set(NHL_TEAM_ABBREVS.values())


def _goalie_tier(sv_pct: float) -> str:
    """Return a tier tag based on save percentage."""
    if sv_pct >= 0.920:
        return "ELITE"
    elif sv_pct >= 0.910:
        return "STRONG"
    elif sv_pct >= 0.900:
        return "AVERAGE"
    else:
        return "BELOW AVG"


# ---------------------------------------------------------------------------
# Fetch today's NHL schedule (games + probable goalies)
# ---------------------------------------------------------------------------

async def fetch_nhl_schedule(date_str: str = None) -> list:
    """Fetch today's NHL games with probable/confirmed starting goalies.

    Primary: NHL Stats API /schedule/now
    Fallback: ESPN NHL scoreboard

    Returns list of dicts:
        [{"game_id": 2024020001, "away": "NYR", "home": "PIT",
          "away_goalie": {"name": "Igor Shesterkin", "confirmed": True},
          "home_goalie": {"name": "Tristan Jarry", "confirmed": False}}, ...]
    """
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cache_key = f"nhl_schedule:{date_str}"
    cached = _get_cached(cache_key, ttl=1800)  # 30 min — goalie confirmations change
    if cached:
        return cached

    games = await _fetch_schedule_nhl_api(date_str)
    if not games:
        games = await _fetch_schedule_espn(date_str)

    _set_cache(cache_key, games)
    if games:
        logger.info(f"[NHL GOALIE] Fetched {len(games)} games for {date_str}")
    return games


async def _fetch_schedule_nhl_api(date_str: str) -> list:
    """Fetch schedule from NHL Stats API."""
    url = f"{NHL_API_BASE}/schedule/{date_str}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NHL GOALIE] NHL API schedule fetch failed: {e}")
        return []

    games = []
    for week in data.get("gameWeek", []):
        if week.get("date") != date_str:
            continue
        for game in week.get("games", []):
            state = game.get("gameState", "")
            if state in ("OFF", "FINAL"):
                continue

            away_team = game.get("awayTeam", {})
            home_team = game.get("homeTeam", {})
            away_abbr = away_team.get("abbrev", "?")
            home_abbr = home_team.get("abbrev", "?")

            # Probable goalies are sometimes nested under the team object
            away_goalie = _extract_goalie_nhl(away_team)
            home_goalie = _extract_goalie_nhl(home_team)

            games.append({
                "game_id": game.get("id"),
                "away": away_abbr,
                "home": home_abbr,
                "game_datetime": game.get("startTimeUTC"),
                "away_goalie": away_goalie,
                "home_goalie": home_goalie,
            })

    return games


def _extract_goalie_nhl(team_data: dict) -> dict:
    """Extract goalie info from NHL API team object in schedule."""
    # The NHL API sometimes includes placeholderGoalie or probableGoalie
    goalie = team_data.get("placeholderGoalie") or team_data.get("probableGoalie")
    if not goalie:
        return {"name": "TBD", "id": None, "confirmed": False}

    name_parts = []
    if goalie.get("firstName"):
        first = goalie["firstName"]
        name_parts.append(first.get("default", first) if isinstance(first, dict) else str(first))
    if goalie.get("lastName"):
        last = goalie["lastName"]
        name_parts.append(last.get("default", last) if isinstance(last, dict) else str(last))

    return {
        "name": " ".join(name_parts) if name_parts else "TBD",
        "id": goalie.get("id"),
        "confirmed": team_data.get("placeholderGoalie") is None and goalie is not None,
    }


async def _fetch_schedule_espn(date_str: str) -> list:
    """Fallback: fetch schedule from ESPN NHL scoreboard."""
    espn_date = date_str.replace("-", "")
    url = f"{ESPN_NHL_BASE}/scoreboard?dates={espn_date}"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NHL GOALIE] ESPN schedule fetch failed: {e}")
        return []

    games = []
    for event in data.get("events", []):
        competition = (event.get("competitions") or [{}])[0]
        competitors = competition.get("competitors", [])
        if len(competitors) < 2:
            continue

        status_type = competition.get("status", {}).get("type", {}).get("name", "")
        if status_type == "STATUS_FINAL":
            continue

        away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])

        away_abbr = away_comp.get("team", {}).get("abbreviation", "?")
        home_abbr = home_comp.get("team", {}).get("abbreviation", "?")

        games.append({
            "game_id": event.get("id"),
            "away": away_abbr,
            "home": home_abbr,
            "game_datetime": event.get("date"),
            "away_goalie": {"name": "TBD", "id": None, "confirmed": False},
            "home_goalie": {"name": "TBD", "id": None, "confirmed": False},
        })

    return games


# ---------------------------------------------------------------------------
# Fetch goalie season stats from NHL API
# ---------------------------------------------------------------------------

async def fetch_goalie_season_stats() -> dict:
    """Fetch season goalie stats leaders from NHL API.

    Endpoint: /goalie-stats-leaders/current?limit=100

    Returns dict keyed by goalie name (lowercase):
        {"igor shesterkin": {"sv_pct": 0.921, "gaa": 2.45, "wins": 25,
                             "losses": 10, "games_started": 40}, ...}
    Cache: 6 hours — season stats are stable
    """
    cache_key = "nhl_goalie_season_stats"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    url = f"{NHL_API_BASE}/goalie-stats-leaders/current"
    params = {"categories": "savePctg", "limit": 100}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NHL GOALIE] Season stats fetch failed: {e}")
        return {}

    goalies = {}
    for category in data.get("categories", []):
        for leader in category.get("leaders", []):
            player = leader.get("player", {})
            first = player.get("firstName", {})
            last = player.get("lastName", {})
            first_name = first.get("default", first) if isinstance(first, dict) else str(first)
            last_name = last.get("default", last) if isinstance(last, dict) else str(last)
            name = f"{first_name} {last_name}".strip().lower()

            if name in goalies:
                continue

            goalies[name] = {
                "sv_pct": _safe_float(leader.get("value"), 0.0),
                "gaa": 0.0,  # Not always in leaders endpoint
                "wins": 0,
                "losses": 0,
                "games_started": 0,
                "player_id": player.get("id"),
                "team_abbr": leader.get("teamAbbrev", {}).get("default", "?") if isinstance(leader.get("teamAbbrev"), dict) else leader.get("teamAbbrev", "?"),
            }

    # Also try to get GAA leaders to enrich
    await _enrich_goalie_gaa(goalies)

    _set_cache(cache_key, goalies)
    logger.info(f"[NHL GOALIE] Cached season stats for {len(goalies)} goalies")
    return goalies


async def _enrich_goalie_gaa(goalies: dict):
    """Enrich goalie dict with GAA from the GAA leaders endpoint."""
    url = f"{NHL_API_BASE}/goalie-stats-leaders/current"
    params = {"categories": "goalsAgainstAverage", "limit": 100}

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NHL GOALIE] GAA leaders fetch failed: {e}")
        return

    for category in data.get("categories", []):
        for leader in category.get("leaders", []):
            player = leader.get("player", {})
            first = player.get("firstName", {})
            last = player.get("lastName", {})
            first_name = first.get("default", first) if isinstance(first, dict) else str(first)
            last_name = last.get("default", last) if isinstance(last, dict) else str(last)
            name = f"{first_name} {last_name}".strip().lower()

            if name in goalies:
                goalies[name]["gaa"] = _safe_float(leader.get("value"), 0.0)
            else:
                goalies[name] = {
                    "sv_pct": 0.0,
                    "gaa": _safe_float(leader.get("value"), 0.0),
                    "wins": 0,
                    "losses": 0,
                    "games_started": 0,
                    "player_id": player.get("id"),
                    "team_abbr": leader.get("teamAbbrev", {}).get("default", "?") if isinstance(leader.get("teamAbbrev"), dict) else leader.get("teamAbbrev", "?"),
                }


# ---------------------------------------------------------------------------
# Fetch club roster goalie stats (for L5, backup, games started in 7 days)
# ---------------------------------------------------------------------------

async def fetch_club_goalie_stats(team_abbr: str) -> list:
    """Fetch goalie stats for a specific team's roster.

    Endpoint: /club-stats/{team}/now

    Returns list of goalie dicts with detailed stats.
    Cache: 6 hours
    """
    cache_key = f"nhl_club_goalies:{team_abbr}"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    url = f"{NHL_API_BASE}/club-stats/{team_abbr}/now"

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NHL GOALIE] Club stats fetch failed for {team_abbr}: {e}")
        return []

    goalies = []
    for goalie in data.get("goalies", []):
        first = goalie.get("firstName", {})
        last = goalie.get("lastName", {})
        first_name = first.get("default", first) if isinstance(first, dict) else str(first)
        last_name = last.get("default", last) if isinstance(last, dict) else str(last)

        goalies.append({
            "name": f"{first_name} {last_name}".strip(),
            "player_id": goalie.get("playerId"),
            "games_played": goalie.get("gamesPlayed", 0),
            "games_started": goalie.get("gamesStarted", 0),
            "wins": goalie.get("wins", 0),
            "losses": goalie.get("losses", 0),
            "sv_pct": _safe_float(goalie.get("savePctg"), 0.0),
            "gaa": _safe_float(goalie.get("goalsAgainstAvg"), 0.0),
            "shutouts": goalie.get("shutouts", 0),
        })

    # Sort by games started descending (starter first)
    goalies.sort(key=lambda g: g["games_started"], reverse=True)

    _set_cache(cache_key, goalies)
    return goalies


# ---------------------------------------------------------------------------
# Back-to-back detection
# ---------------------------------------------------------------------------

async def _fetch_yesterday_goalies(date_str: str) -> set:
    """Return set of goalie names (lowercase) who started yesterday.

    Uses yesterday's schedule to detect B2B goalie starts.
    Cache: 30 min
    """
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        target = datetime.now(timezone.utc)
    prev_date = (target - timedelta(days=1)).strftime("%Y-%m-%d")

    cache_key = f"nhl_yesterday_goalies:{prev_date}"
    cached = _get_cached(cache_key, ttl=1800)
    if cached is not None:
        return cached

    yesterday_games = await _fetch_schedule_nhl_api(prev_date)
    if not yesterday_games:
        yesterday_games = await _fetch_schedule_espn(prev_date)

    starters = set()
    for game in yesterday_games:
        for side in ("away_goalie", "home_goalie"):
            name = game.get(side, {}).get("name", "TBD")
            if name != "TBD":
                starters.add(name.lower())

    _set_cache(cache_key, starters)
    return starters


# ---------------------------------------------------------------------------
# Build goalie matchup context for AI prompt
# ---------------------------------------------------------------------------

async def build_goalie_context(date_str: str = None) -> str:
    """Build the goalie matchup context block for the AI analysis prompt.

    Returns formatted text:
    === GOALIE MATCHUP ===
    RULE: Goalie is THE #1 variable in NHL (weight 10). ...

      NYR: Igor Shesterkin (confirmed) | .921 SV% | 2.45 GAA | L5: .928 SV% — ELITE
      PIT: Tristan Jarry (expected) | .898 SV% | 3.12 GAA | L5: .891 SV% — BELOW AVG
      Goalie edge: NYR (elite confirmed vs below-avg expected)
    """
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    games = await fetch_nhl_schedule(date_str)
    if not games:
        return "=== GOALIE MATCHUP ===\nNo NHL games scheduled or goalie data unavailable."

    season_stats = await fetch_goalie_season_stats()
    yesterday_goalies = await _fetch_yesterday_goalies(date_str)

    lines = [
        "=== GOALIE MATCHUP ===",
        "RULE: Goalie is THE #1 variable in NHL (weight 10). "
        "Confirmed starter = full weight. Unconfirmed = reduce confidence by 30%. "
        "SV% > .920 vs SV% < .900 = massive edge. B2B goalie start = fatigue, score 3-4.",
        "",
    ]

    for game in games:
        matchup = f"{game['away']} @ {game['home']}"
        lines.append(f"{matchup}:")

        away_info = await _build_team_goalie_line(
            game["away"], game["away_goalie"], season_stats, yesterday_goalies
        )
        home_info = await _build_team_goalie_line(
            game["home"], game["home_goalie"], season_stats, yesterday_goalies
        )

        lines.append(f"  {away_info['line']}")
        lines.append(f"  {home_info['line']}")

        # Goalie edge summary
        edge = _determine_edge(
            game["away"], away_info, game["home"], home_info
        )
        lines.append(f"  {edge}")
        lines.append("")

    return "\n".join(lines)


async def _build_team_goalie_line(
    team_abbr: str,
    goalie_info: dict,
    season_stats: dict,
    yesterday_goalies: set,
) -> dict:
    """Build a single team's goalie line for the context block.

    Returns dict with 'line' (formatted string) and metadata for edge calc.
    """
    name = goalie_info.get("name", "TBD")
    confirmed = goalie_info.get("confirmed", False)
    status_tag = "confirmed" if confirmed else "expected"

    # Look up season stats
    name_lower = name.lower()
    stats = season_stats.get(name_lower, {})

    sv_pct = stats.get("sv_pct", 0.0)
    gaa = stats.get("gaa", 0.0)

    # Try club stats for richer data and backup goalie info
    club_goalies = await fetch_club_goalie_stats(team_abbr)
    starter_club = None
    backup_club = None

    for cg in club_goalies:
        if cg["name"].lower() == name_lower:
            starter_club = cg
            # Use club stats if season leaders didn't have this goalie
            if sv_pct == 0.0:
                sv_pct = cg.get("sv_pct", 0.0)
            if gaa == 0.0:
                gaa = cg.get("gaa", 0.0)
        elif backup_club is None and cg["name"].lower() != name_lower:
            backup_club = cg

    # If name is TBD, use the team's #1 starter from club stats
    if name == "TBD" and club_goalies:
        starter_club = club_goalies[0]
        name = starter_club["name"]
        sv_pct = starter_club.get("sv_pct", 0.0)
        gaa = starter_club.get("gaa", 0.0)
        status_tag = "expected"
        if len(club_goalies) > 1:
            backup_club = club_goalies[1]

    # B2B detection
    is_b2b = name_lower in yesterday_goalies

    # Tier tag
    tier = _goalie_tier(sv_pct) if sv_pct > 0 else "N/A"

    # Format SV% and GAA
    sv_str = f".{int(sv_pct * 1000):03d}" if sv_pct > 0 else "-"
    gaa_str = f"{gaa:.2f}" if gaa > 0 else "-"

    # Build the line
    parts = [f"{team_abbr}: {name} ({status_tag})"]
    parts.append(f"{sv_str} SV%")
    parts.append(f"{gaa_str} GAA")

    # L5 approximation from club stats (if available, use season as proxy)
    # NHL API doesn't expose L5 directly in leaders; club stats are season-level
    # We note this as season stats (L5 would require game log parsing)
    if starter_club:
        l5_sv = starter_club.get("sv_pct", sv_pct)
        l5_sv_str = f".{int(l5_sv * 1000):03d}" if l5_sv > 0 else "-"
        parts.append(f"L5: {l5_sv_str} SV%")

    # Flags
    flags = []
    if is_b2b:
        flags.append("B2B START")
        tier = "B2B START"
    if not confirmed:
        flags.append("UNCONFIRMED")

    parts.append(tier)
    if flags:
        parts[-1] = " | ".join(flags) + f" | {tier}" if tier not in flags else " | ".join(flags)

    line = " | ".join(parts)

    # Backup info
    if backup_club:
        backup_sv = backup_club.get("sv_pct", 0.0)
        backup_gaa = backup_club.get("gaa", 0.0)
        backup_sv_str = f".{int(backup_sv * 1000):03d}" if backup_sv > 0 else "-"
        backup_gaa_str = f"{backup_gaa:.2f}" if backup_gaa > 0 else "-"
        line += f"\n    Backup: {backup_club['name']} ({backup_sv_str} SV% | {backup_gaa_str} GAA)"

    return {
        "line": line,
        "sv_pct": sv_pct,
        "gaa": gaa,
        "tier": _goalie_tier(sv_pct) if sv_pct > 0 else "N/A",
        "confirmed": confirmed,
        "is_b2b": is_b2b,
        "name": name,
    }


def _determine_edge(
    away_abbr: str, away_info: dict,
    home_abbr: str, home_info: dict,
) -> str:
    """Determine which team has the goalie edge."""
    away_sv = away_info["sv_pct"]
    home_sv = home_info["sv_pct"]
    away_tier = away_info["tier"].lower()
    home_tier = home_info["tier"].lower()
    away_conf = "confirmed" if away_info["confirmed"] else "expected"
    home_conf = "confirmed" if home_info["confirmed"] else "expected"

    diff = abs(away_sv - home_sv)

    if diff < 0.005 and away_sv > 0 and home_sv > 0:
        return "Goalie edge: EVEN (similar caliber)"

    if away_sv > home_sv:
        edge_team = away_abbr
        edge_tier = away_tier
        edge_conf = away_conf
        opp_tier = home_tier
        opp_conf = home_conf
    else:
        edge_team = home_abbr
        edge_tier = home_tier
        edge_conf = home_conf
        opp_tier = away_tier
        opp_conf = away_conf

    # Factor in B2B fatigue
    if away_info["is_b2b"] and not home_info["is_b2b"]:
        edge_team = home_abbr
        edge_tier = home_tier
        edge_conf = home_conf
        opp_tier = away_tier
        opp_conf = away_conf
        return f"Goalie edge: {edge_team} ({edge_tier} {edge_conf} vs {opp_tier} {opp_conf} + B2B fatigue)"
    elif home_info["is_b2b"] and not away_info["is_b2b"]:
        edge_team = away_abbr
        edge_tier = away_tier
        edge_conf = away_conf
        opp_tier = home_tier
        opp_conf = home_conf
        return f"Goalie edge: {edge_team} ({edge_tier} {edge_conf} vs {opp_tier} {opp_conf} + B2B fatigue)"

    return f"Goalie edge: {edge_team} ({edge_tier} {edge_conf} vs {opp_tier} {opp_conf})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

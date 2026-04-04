"""
NBA Player Usage Engine — Star minutes, load management, and role player absorption.

ESPN APIs (free, public, no key required):
  Roster:   https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{id}/roster
  Gamelog:  https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{id}/gamelog

Used by: server.py analysis prompt (player usage context injection)
Cache: 2 hours for roster + player stats
"""

import httpx
import logging
from datetime import datetime, timezone

from engines.mlb_pitcher_engine import _get_cached, _set_cache

logger = logging.getLogger("edge-crew")

ESPN_NBA_ROSTER_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
ESPN_NBA_GAMELOG_BASE = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes"

# ESPN NBA team slug → numeric ID (all 30 teams)
NBA_TEAM_IDS = {
    "ATL": 1, "BOS": 2, "BKN": 17, "CHA": 30, "CHI": 4,
    "CLE": 5, "DAL": 6, "DEN": 7, "DET": 8, "GS": 9,
    "HOU": 10, "IND": 11, "LAC": 12, "LAL": 13, "MEM": 29,
    "MIA": 14, "MIL": 15, "MIN": 16, "NO": 3, "NY": 18,
    "OKC": 25, "ORL": 19, "PHI": 20, "PHX": 21, "POR": 22,
    "SAC": 23, "SA": 24, "TOR": 28, "UTAH": 26, "WSH": 27,
}

# Alternate abbreviation lookups (ESPN sometimes uses these)
NBA_TEAM_ALIASES = {
    "GSW": "GS", "NOP": "NO", "NYK": "NY", "SAS": "SA",
    "UTA": "UTAH", "BRK": "BKN", "PHO": "PHX", "CHP": "CHA",
}


def _resolve_team_id(abbr: str) -> int | None:
    """Resolve a team abbreviation to ESPN numeric ID."""
    upper = abbr.upper()
    if upper in NBA_TEAM_IDS:
        return NBA_TEAM_IDS[upper]
    alias = NBA_TEAM_ALIASES.get(upper)
    if alias:
        return NBA_TEAM_IDS.get(alias)
    return None


async def fetch_roster(team_abbr: str) -> list:
    """Fetch team roster from ESPN.

    Returns list of dicts:
        [{"id": "1966", "name": "LeBron James", "pos": "SF", "jersey": "6"}, ...]
    Cache: 2 hours
    """
    team_id = _resolve_team_id(team_abbr)
    if not team_id:
        return []

    cache_key = f"nba_roster:{team_abbr}"
    cached = _get_cached(cache_key, ttl=7200)  # 2 hours
    if cached:
        return cached

    url = f"{ESPN_NBA_ROSTER_BASE}/{team_id}/roster"

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NBA USAGE] Roster fetch failed for {team_abbr}: {e}")
        return []

    players = []
    for athlete in data.get("athletes", []):
        players.append({
            "id": athlete.get("id", ""),
            "name": athlete.get("displayName", "?"),
            "pos": athlete.get("position", {}).get("abbreviation", "?"),
            "jersey": athlete.get("jersey", "?"),
        })

    _set_cache(cache_key, players)
    return players


async def fetch_player_gamelog(player_id: str) -> dict:
    """Fetch season game log for an NBA player from ESPN.

    Endpoint: /athletes/{id}/gamelog

    Returns: {"season_avg": {"pts": 27.1, "reb": 7.3, "ast": 7.8, "min": 36.2},
              "last5_min": [34, 36, 33, 35, 36],
              "games": [...raw game entries...]}
    Cache: 2 hours
    """
    if not player_id:
        return {}

    cache_key = f"nba_gamelog:{player_id}"
    cached = _get_cached(cache_key, ttl=7200)  # 2 hours
    if cached:
        return cached

    url = f"{ESPN_NBA_GAMELOG_BASE}/{player_id}/gamelog"

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NBA USAGE] Gamelog fetch failed for player {player_id}: {e}")
        return {}

    result = _parse_gamelog(data)
    if result:
        _set_cache(cache_key, result)
    return result


def _parse_gamelog(data: dict) -> dict:
    """Parse ESPN gamelog response into structured stats.

    ESPN gamelog format has:
      - seasonTypes[].categories[].events[] with stats[]
      - labels[] mapping stat column positions
    """
    result = {
        "season_avg": {"pts": 0.0, "reb": 0.0, "ast": 0.0, "min": 0.0},
        "last5_min": [],
        "games": [],
    }

    # Try to find the regular season stats
    season_types = data.get("seasonTypes", [])
    if not season_types:
        return result

    # Find regular season (usually type "2") or use first available
    categories = []
    for st in season_types:
        cats = st.get("categories", [])
        if cats:
            categories = cats
            break

    if not categories:
        return result

    # Find the main stats category (usually "offense" or first one)
    events = []
    labels = []
    for cat in categories:
        cat_events = cat.get("events", [])
        cat_labels = cat.get("labels", [])
        if cat_events and cat_labels:
            events = cat_events
            labels = cat_labels
            break

    if not events or not labels:
        return result

    # Build label index for columns we care about
    label_lower = [l.lower() for l in labels]
    idx_min = _find_label_idx(label_lower, ["min", "mins", "minutes"])
    idx_pts = _find_label_idx(label_lower, ["pts", "points"])
    idx_reb = _find_label_idx(label_lower, ["reb", "rebounds"])
    idx_ast = _find_label_idx(label_lower, ["ast", "assists"])

    # Parse each game event
    all_min = []
    all_pts = []
    all_reb = []
    all_ast = []

    for event in events:
        stats = event.get("stats", [])
        if not stats:
            continue

        game_min = _safe_float(stats, idx_min)
        game_pts = _safe_float(stats, idx_pts)
        game_reb = _safe_float(stats, idx_reb)
        game_ast = _safe_float(stats, idx_ast)

        # Skip DNP entries (0 minutes)
        if game_min <= 0:
            continue

        all_min.append(game_min)
        all_pts.append(game_pts)
        all_reb.append(game_reb)
        all_ast.append(game_ast)

        result["games"].append({
            "min": game_min,
            "pts": game_pts,
            "reb": game_reb,
            "ast": game_ast,
        })

    # Compute season averages
    if all_min:
        n = len(all_min)
        result["season_avg"] = {
            "pts": round(sum(all_pts) / n, 1),
            "reb": round(sum(all_reb) / n, 1),
            "ast": round(sum(all_ast) / n, 1),
            "min": round(sum(all_min) / n, 1),
        }

    # Last 5 games minutes (most recent first)
    result["last5_min"] = all_min[:5]

    return result


def _find_label_idx(labels: list, candidates: list) -> int:
    """Find index of first matching label from candidates list."""
    for candidate in candidates:
        for i, label in enumerate(labels):
            if candidate in label:
                return i
    return -1


def _safe_float(stats: list, idx: int) -> float:
    """Safely extract a float from stats list at index."""
    if idx < 0 or idx >= len(stats):
        return 0.0
    try:
        val = str(stats[idx]).strip()
        # Handle "MM:SS" minute format
        if ":" in val:
            parts = val.split(":")
            return float(parts[0]) + float(parts[1]) / 60.0
        return float(val)
    except (ValueError, TypeError, IndexError):
        return 0.0


async def fetch_team_usage(team_abbr: str) -> list:
    """Fetch usage data for top players on a team.

    Returns list of top 5 players sorted by PPG:
        [{"name": "LeBron James", "ppg": 27.1, "rpg": 7.3, "apg": 7.8,
          "mpg": 36.2, "l5_min_avg": 34.8, "tag": "NORMAL USAGE"}, ...]
    Cache: 2 hours (via sub-fetches)
    """
    cache_key = f"nba_team_usage:{team_abbr}"
    cached = _get_cached(cache_key, ttl=7200)  # 2 hours
    if cached:
        return cached

    roster = await fetch_roster(team_abbr)
    if not roster:
        return []

    players = []
    async with httpx.AsyncClient(timeout=12) as client:
        for athlete in roster:
            pid = athlete.get("id")
            if not pid:
                continue

            gamelog = await fetch_player_gamelog(pid)
            if not gamelog or not gamelog.get("games"):
                continue

            avg = gamelog["season_avg"]
            l5_min = gamelog.get("last5_min", [])
            l5_min_avg = round(sum(l5_min) / len(l5_min), 1) if l5_min else 0.0
            season_mpg = avg["min"]

            players.append({
                "name": athlete.get("name", "?"),
                "ppg": avg["pts"],
                "rpg": avg["reb"],
                "apg": avg["ast"],
                "mpg": season_mpg,
                "l5_min_avg": l5_min_avg,
                "l5_min": l5_min,
            })

    # Sort by PPG descending, take top 5
    players.sort(key=lambda p: p["ppg"], reverse=True)
    top5 = players[:5]

    # Tag each player
    for p in top5:
        p["tag"] = _compute_usage_tag(p, top5)

    _set_cache(cache_key, top5)
    return top5


def _compute_usage_tag(player: dict, team_top5: list) -> str:
    """Determine usage tag for a player.

    Tags:
      STAR OUT           — star (25+ PPG) with 0 games in L5 (placeholder for injury data)
      LOAD MANAGEMENT RISK — star (25+ PPG, 32+ MPG) whose L5 avg is 4+ min below season avg
      TRENDING UP        — non-star whose L5 minutes are 15%+ above season avg
      NORMAL USAGE       — default
    """
    ppg = player["ppg"]
    mpg = player["mpg"]
    l5_avg = player["l5_min_avg"]

    is_star = ppg >= 25.0

    # Load management: star plays 32+ MPG season but L5 avg dropped 4+ minutes
    if is_star and mpg >= 32.0 and l5_avg > 0:
        if l5_avg < (mpg - 4.0):
            return "LOAD MANAGEMENT RISK"

    # Star with no recent minutes (sat out)
    if is_star and l5_avg == 0:
        return "STAR OUT"

    # Role player absorption: non-star minutes trending up 15%+
    if not is_star and mpg > 0 and l5_avg > 0:
        pct_change = (l5_avg - mpg) / mpg
        if pct_change >= 0.15:
            return "TRENDING UP"

    return "NORMAL USAGE"


async def build_usage_context(games: list) -> str:
    """Build player usage context for AI prompt.

    Args:
        games: list of dicts with "away" and "home" team abbreviations.
               e.g. [{"away": "LAL", "home": "BOS"}, ...]

    Returns formatted text like:
    === PLAYER USAGE ===
    RULE: Star player (25+ PPG) missing or on minutes restriction = ...

      LAL Top 5:
        LeBron James: 27.1 PPG | 36.2 MPG | L5 avg 34.8 min — NORMAL USAGE
        Anthony Davis: 25.4 PPG | 34.1 MPG | L5 avg 28.2 min — LOAD MANAGEMENT RISK
      BOS Top 5:
        Jayson Tatum: 28.3 PPG | 35.8 MPG | L5 avg 36.1 min — NORMAL USAGE
    """
    if not games:
        return "=== PLAYER USAGE ===\nNo games to evaluate."

    lines = ["=== PLAYER USAGE ==="]
    lines.append(
        "RULE: Star player (25+ PPG) missing or on minutes restriction = "
        "score star_player_status 3-4. Load management risk on B2B = flag. "
        "Role player minutes trending up 20%+ with star out = absorption play for props."
    )
    lines.append("")

    for game in games:
        for side_key in ("away", "home"):
            abbr = game.get(side_key, "?")
            top5 = await fetch_team_usage(abbr)

            if not top5:
                lines.append(f"  {abbr} Top 5: data unavailable")
                lines.append("")
                continue

            lines.append(f"  {abbr} Top 5:")
            for p in top5:
                lines.append(
                    f"    {p['name']}: {p['ppg']} PPG | {p['mpg']} MPG | "
                    f"L5 avg {p['l5_min_avg']} min — {p['tag']}"
                )
            lines.append("")

    return "\n".join(lines)

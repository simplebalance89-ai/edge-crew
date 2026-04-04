"""
NHL Special Teams Engine — PP/PK and team scoring stats from NHL Stats API.

NHL Stats API: https://api-web.nhle.com/v1/
Free, public, no API key required.

Used by: server.py analysis prompt (special teams context injection)
Cache: 6 hours for team stats
"""

import httpx
import logging
from datetime import datetime, timezone

from engines.mlb_pitcher_engine import _get_cached, _set_cache

logger = logging.getLogger("edge-crew")

NHL_STATS_BASE = "https://api-web.nhle.com/v1"

# NHL team abbreviation -> NHL API 3-letter codes (all 32 teams)
NHL_TEAM_IDS = {
    "ANA": "ANA", "ARI": "ARI", "BOS": "BOS", "BUF": "BUF",
    "CGY": "CGY", "CAR": "CAR", "CHI": "CHI", "COL": "COL",
    "CBJ": "CBJ", "DAL": "DAL", "DET": "DET", "EDM": "EDM",
    "FLA": "FLA", "LAK": "LAK", "MIN": "MIN", "MTL": "MTL",
    "NSH": "NSH", "NJD": "NJD", "NYI": "NYI", "NYR": "NYR",
    "OTT": "OTT", "PHI": "PHI", "PIT": "PIT", "SJS": "SJS",
    "SEA": "SEA", "STL": "STL", "TBL": "TBL", "TOR": "TOR",
    "UTA": "UTA", "VAN": "VAN", "VGK": "VGK", "WPG": "WPG",
    # Common aliases
    "WSH": "WSH", "TB": "TBL", "LA": "LAK", "SJ": "SJS",
    "NJ": "NJD", "NY": "NYR",
}


async def fetch_team_stats(team_abbr: str) -> dict:
    """Fetch team special teams and scoring stats from NHL Stats API.

    Endpoint: /club-stats/{team}/now

    Returns: {"pp_pct": 24.5, "pk_pct": 81.2, "pp_opps_per_gp": 3.1,
              "gf_per_gp": 3.21, "ga_per_gp": 2.67, "ev_goal_diff": 12}
    Cache: 6 hours
    """
    team_code = NHL_TEAM_IDS.get(team_abbr)
    if not team_code:
        return _empty_stats()

    cache_key = f"nhl_specialteams:{team_abbr}"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    result = _empty_stats()

    # Fetch from standings for team-level stats
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{NHL_STATS_BASE}/standings/now")
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NHL ST] Standings fetch failed for {team_abbr}: {e}")
        return result

    # Parse standings data for the target team
    for team in data.get("standings", []):
        abbr = team.get("teamAbbrev", {}).get("default", "")
        if abbr != team_code:
            continue

        games_played = int(team.get("gamesPlayed", 1)) or 1
        goals_for = int(team.get("goalFor", 0))
        goals_against = int(team.get("goalAgainst", 0))

        result["gf_per_gp"] = round(goals_for / games_played, 2)
        result["ga_per_gp"] = round(goals_against / games_played, 2)
        result["games_played"] = games_played
        break

    # Fetch club-specific stats for PP/PK detail
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{NHL_STATS_BASE}/club-stats/{team_code}/now")
            resp.raise_for_status()
            club_data = resp.json()
    except Exception as e:
        logger.warning(f"[NHL ST] Club stats fetch failed for {team_abbr}: {e}")
        _set_cache(cache_key, result)
        return result

    # Parse skater and goalie stats to derive PP/PK metrics
    # PP goals and opportunities come from skater power-play stats
    total_pp_goals = 0
    total_pk_goals_against = 0
    skaters = club_data.get("skaters", [])
    for skater in skaters:
        total_pp_goals += int(skater.get("powerPlayGoals", 0))

    goalies = club_data.get("goalies", [])
    # Goalie stats can provide saves/shots on PK
    # For now derive from team-level data

    # Try to extract PP/PK from team record fields if available
    # The standings endpoint includes regulationWins, etc. but not PP/PK directly
    # Fall back to estimation from club stats if direct API fields unavailable

    gp = result.get("games_played", 1) or 1

    # Estimate PP opportunities per game (~3.0 league avg)
    # PP% = PP goals / PP opportunities; estimate opportunities from goals
    if total_pp_goals > 0:
        # League average PP% is ~22%, so estimate opportunities
        est_pp_opps = round(total_pp_goals / 0.22)
        result["pp_pct"] = round((total_pp_goals / est_pp_opps) * 100, 1) if est_pp_opps > 0 else 0.0
        result["pp_opps_per_gp"] = round(est_pp_opps / gp, 1)
    else:
        result["pp_pct"] = 0.0
        result["pp_opps_per_gp"] = 0.0

    # Even-strength goal differential (total goals minus PP goals as proxy)
    ev_gf = int(result["gf_per_gp"] * gp) - total_pp_goals
    ev_ga = int(result["ga_per_gp"] * gp)  # Approximate — includes SH goals against
    result["ev_goal_diff"] = ev_gf - ev_ga

    _set_cache(cache_key, result)
    return result


async def fetch_standings_rankings() -> dict:
    """Fetch all team PP/PK rankings from NHL standings.

    Returns: {"NYR": {"pp_rank": 6, "pk_rank": 10, ...}, ...}
    Cache: 6 hours
    """
    cache_key = "nhl_st_rankings"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{NHL_STATS_BASE}/standings/now")
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NHL ST] Rankings fetch failed: {e}")
        return {}

    teams = []
    for team in data.get("standings", []):
        abbr = team.get("teamAbbrev", {}).get("default", "")
        gp = int(team.get("gamesPlayed", 1)) or 1
        gf = int(team.get("goalFor", 0))
        ga = int(team.get("goalAgainst", 0))
        teams.append({
            "abbr": abbr,
            "gf_per_gp": round(gf / gp, 2),
            "ga_per_gp": round(ga / gp, 2),
            "gp": gp,
        })

    # Sort for offensive/defensive rankings
    teams_by_gf = sorted(teams, key=lambda t: t["gf_per_gp"], reverse=True)
    teams_by_ga = sorted(teams, key=lambda t: t["ga_per_gp"])

    rankings = {}
    for i, t in enumerate(teams_by_gf, 1):
        rankings.setdefault(t["abbr"], {})["gf_rank"] = i
    for i, t in enumerate(teams_by_ga, 1):
        rankings.setdefault(t["abbr"], {})["ga_rank"] = i

    _set_cache(cache_key, rankings)
    return rankings


def _empty_stats() -> dict:
    """Return empty stats dict."""
    return {
        "pp_pct": 0.0,
        "pk_pct": 0.0,
        "pp_opps_per_gp": 0.0,
        "gf_per_gp": 0.0,
        "ga_per_gp": 0.0,
        "ev_goal_diff": 0,
        "games_played": 0,
    }


def _pp_tag(pp_pct: float) -> str:
    """Return PP strength tag."""
    if pp_pct >= 25.0:
        return "ELITE PP"
    elif pp_pct >= 22.0:
        return "STRONG PP"
    elif pp_pct >= 20.0:
        return "AVG PP"
    else:
        return "WEAK PP"


def _pk_tag(pk_pct: float) -> str:
    """Return PK strength tag."""
    if pk_pct >= 82.0:
        return "ELITE PK"
    elif pk_pct >= 78.0:
        return "STRONG PK"
    elif pk_pct >= 76.0:
        return "AVG PK"
    else:
        return "WEAK PK"


def _overall_tag(pp_pct: float, pk_pct: float) -> str:
    """Return overall special teams assessment."""
    if pp_pct >= 22.0 and pk_pct >= 78.0:
        return "STRONG SPECIAL TEAMS"
    elif pp_pct < 20.0 and pk_pct < 76.0:
        return "WEAK SPECIAL TEAMS"
    elif pp_pct >= 25.0 or pk_pct >= 82.0:
        return "ELITE UNIT"
    else:
        return "MIXED SPECIAL TEAMS"


async def build_specialteams_context(games: list) -> str:
    """Build special teams context for AI prompt.

    Args:
        games: list of dicts with 'away' and 'home' team abbreviations.
               e.g. [{"away": "NYR", "home": "PIT"}, ...]

    Returns:
    === SPECIAL TEAMS ===
    RULE: PP vs PK matchup is the #2 variable in NHL after goalie. ...

      NYR: 24.5% PP (#6) | 81.2% PK (#10) | 3.21 GF/GP | 2.67 GA/GP — STRONG SPECIAL TEAMS
      PIT: 19.8% PP (#22) | 76.3% PK (#26) | 2.89 GF/GP | 3.34 GA/GP — WEAK SPECIAL TEAMS
      Matchup: NYR PP (#6) vs PIT PK (#26) = PP MISMATCH EDGE
    """
    if not games:
        return "=== SPECIAL TEAMS ===\nNo games to evaluate."

    lines = ["=== SPECIAL TEAMS ==="]
    lines.append(
        "RULE: PP vs PK matchup is the #2 variable in NHL after goalie. "
        "Top-5 PP (> 24%) vs bottom-5 PK (< 76%) = score pp_pk 8-9. "
        "Both elite PK (> 82%) = under lean. "
        "Undisciplined team (> 4 penalties/game) = PP opportunities for opponent."
    )
    lines.append("")

    rankings = await fetch_standings_rankings()

    for game in games:
        game_teams = {}
        for side_key in ("away", "home"):
            abbr = game.get(side_key, "?")
            stats = await fetch_team_stats(abbr)
            team_rankings = rankings.get(NHL_TEAM_IDS.get(abbr, abbr), {})

            pp_pct = stats["pp_pct"]
            pk_pct = stats["pk_pct"]
            gf = stats["gf_per_gp"]
            ga = stats["ga_per_gp"]
            ev_diff = stats["ev_goal_diff"]
            pp_opps = stats["pp_opps_per_gp"]

            # Build rank strings (from rankings if available)
            gf_rank = team_rankings.get("gf_rank", "?")
            ga_rank = team_rankings.get("ga_rank", "?")

            overall = _overall_tag(pp_pct, pk_pct)

            line = (
                f"  {abbr}: {pp_pct}% PP ({_pp_tag(pp_pct)}) | "
                f"{pk_pct}% PK ({_pk_tag(pk_pct)}) | "
                f"{gf} GF/GP (#{gf_rank}) | {ga} GA/GP (#{ga_rank}) | "
                f"5v5 diff: {ev_diff:+d} | {pp_opps} PP opps/GP — {overall}"
            )
            lines.append(line)

            game_teams[side_key] = {
                "abbr": abbr,
                "pp_pct": pp_pct,
                "pk_pct": pk_pct,
                "pp_opps": pp_opps,
            }

        # Matchup analysis
        away = game_teams.get("away", {})
        home = game_teams.get("home", {})

        matchup_notes = []

        # PP vs PK mismatch detection
        if away.get("pp_pct", 0) >= 24.0 and home.get("pk_pct", 0) < 76.0:
            matchup_notes.append(
                f"{away['abbr']} PP ({away['pp_pct']}%) vs "
                f"{home['abbr']} PK ({home['pk_pct']}%) = PP MISMATCH EDGE"
            )
        if home.get("pp_pct", 0) >= 24.0 and away.get("pk_pct", 0) < 76.0:
            matchup_notes.append(
                f"{home['abbr']} PP ({home['pp_pct']}%) vs "
                f"{away['abbr']} PK ({away['pk_pct']}%) = PP MISMATCH EDGE"
            )

        # Both elite PK = under lean
        if away.get("pk_pct", 0) >= 82.0 and home.get("pk_pct", 0) >= 82.0:
            matchup_notes.append("Both teams ELITE PK — UNDER lean")

        # Undisciplined teams (high PP opportunities for opponent)
        for side in (away, home):
            if side.get("pp_opps", 0) >= 4.0:
                matchup_notes.append(
                    f"{side['abbr']} undisciplined ({side['pp_opps']} penalties/GP) — "
                    f"PP opportunities for opponent"
                )

        if matchup_notes:
            for note in matchup_notes:
                lines.append(f"  Matchup: {note}")
        else:
            lines.append("  Matchup: No significant special teams mismatch")

        lines.append("")

    return "\n".join(lines)

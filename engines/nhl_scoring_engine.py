"""
NHL Scoring Depth Engine — Team scoring and shot metrics from NHL Stats API.

NHL Stats API: https://api-web.nhle.com/v1/
Free, public, no API key required.

Used by: server.py analysis prompt (NHL scoring depth context injection)
Cache: 6 hours for team scoring stats
"""

import httpx
import logging
from datetime import datetime, timezone

from engines.mlb_pitcher_engine import _get_cached, _set_cache

logger = logging.getLogger("edge-crew")

NHL_STANDINGS_URL = "https://api-web.nhle.com/v1/standings/now"


async def fetch_nhl_standings() -> list:
    """Fetch current NHL standings with GF, GA, GP per team.

    Returns list of dicts:
        [{"abbr": "NYR", "gp": 72, "gf": 246, "ga": 181, "w": 44, "l": 20, "otl": 8}, ...]

    Cache: 6 hours
    """
    cache_key = "nhl_standings"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(NHL_STANDINGS_URL)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NHL SCORING] Standings fetch failed: {e}")
        return []

    teams = []
    for team in data.get("standings", []):
        abbr = team.get("teamAbbrev", {}).get("default", "???")
        gp = int(team.get("gamesPlayed", 0)) or 1
        gf = int(team.get("goalFor", 0))
        ga = int(team.get("goalAgainst", 0))
        w = int(team.get("wins", 0))
        l = int(team.get("losses", 0))
        otl = int(team.get("otLosses", 0))

        teams.append({
            "abbr": abbr,
            "gp": gp,
            "gf": gf,
            "ga": ga,
            "w": w,
            "l": l,
            "otl": otl,
            "gf_per_gp": round(gf / gp, 2),
            "ga_per_gp": round(ga / gp, 2),
            "goal_diff_per_gp": round((gf - ga) / gp, 2),
        })

    _set_cache(cache_key, teams)
    logger.info(f"[NHL SCORING] Fetched standings for {len(teams)} teams")
    return teams


async def fetch_team_scoring(team_abbr: str) -> dict:
    """Get scoring metrics for a single team from cached standings.

    Returns: {"gf_per_gp": 3.42, "ga_per_gp": 2.51, "goal_diff_per_gp": 0.91,
              "gf_rank": 5, "ga_rank": 8, "gp": 72, "gf": 246, "ga": 181}
    Cache: uses standings cache (6 hours)
    """
    cache_key = f"nhl_team_scoring:{team_abbr}"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    standings = await fetch_nhl_standings()
    if not standings:
        return _empty_scoring()

    # Compute league-wide ranks
    sorted_gf = sorted(standings, key=lambda t: t["gf_per_gp"], reverse=True)
    sorted_ga = sorted(standings, key=lambda t: t["ga_per_gp"])  # lower is better

    gf_ranks = {t["abbr"]: i + 1 for i, t in enumerate(sorted_gf)}
    ga_ranks = {t["abbr"]: i + 1 for i, t in enumerate(sorted_ga)}

    team_data = next((t for t in standings if t["abbr"] == team_abbr), None)
    if not team_data:
        return _empty_scoring()

    result = {
        "gf_per_gp": team_data["gf_per_gp"],
        "ga_per_gp": team_data["ga_per_gp"],
        "goal_diff_per_gp": team_data["goal_diff_per_gp"],
        "gf_rank": gf_ranks.get(team_abbr, 0),
        "ga_rank": ga_ranks.get(team_abbr, 0),
        "gp": team_data["gp"],
        "gf": team_data["gf"],
        "ga": team_data["ga"],
    }

    _set_cache(cache_key, result)
    return result


async def fetch_recent_form(team_abbr: str) -> dict:
    """Fetch last-5-games scoring form from NHL club stats.

    Endpoint: https://api-web.nhle.com/v1/club-stats-season/{team_abbr}
    Fallback: https://api-web.nhle.com/v1/scoreboard/now (recent scores)

    Returns: {"l5_gf_per_gp": 3.80, "l5_ga_per_gp": 2.20, "l5_games": 5}
    Cache: 6 hours
    """
    cache_key = f"nhl_recent_form:{team_abbr}"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    # Try to get recent game results from the schedule/results endpoint
    url = f"https://api-web.nhle.com/v1/club-schedule-season/{team_abbr}/now"

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NHL SCORING] Recent form fetch failed for {team_abbr}: {e}")
        return {"l5_gf_per_gp": None, "l5_ga_per_gp": None, "l5_games": 0}

    # Filter to completed games, take last 5
    completed = []
    for game in data.get("games", []):
        game_state = game.get("gameState", "")
        if game_state in ("OFF", "FINAL"):
            completed.append(game)

    last_5 = completed[-5:] if len(completed) >= 5 else completed

    if not last_5:
        return {"l5_gf_per_gp": None, "l5_ga_per_gp": None, "l5_games": 0}

    total_gf = 0
    total_ga = 0
    for game in last_5:
        home_abbr = game.get("homeTeam", {}).get("abbrev", "")
        away_abbr = game.get("awayTeam", {}).get("abbrev", "")
        home_score = int(game.get("homeTeam", {}).get("score", 0))
        away_score = int(game.get("awayTeam", {}).get("score", 0))

        if home_abbr == team_abbr:
            total_gf += home_score
            total_ga += away_score
        elif away_abbr == team_abbr:
            total_gf += away_score
            total_ga += home_score

    n = len(last_5)
    result = {
        "l5_gf_per_gp": round(total_gf / n, 2) if n else None,
        "l5_ga_per_gp": round(total_ga / n, 2) if n else None,
        "l5_games": n,
    }

    _set_cache(cache_key, result)
    return result


def _empty_scoring() -> dict:
    """Return empty scoring dict when data is unavailable."""
    return {
        "gf_per_gp": 0.0,
        "ga_per_gp": 0.0,
        "goal_diff_per_gp": 0.0,
        "gf_rank": 0,
        "ga_rank": 0,
        "gp": 0,
        "gf": 0,
        "ga": 0,
    }


def _offense_tag(gf_per_gp: float) -> str:
    """Tag offensive output level."""
    if gf_per_gp >= 3.5:
        return "ELITE OFFENSE"
    elif gf_per_gp >= 3.0:
        return "STRONG OFFENSE"
    elif gf_per_gp >= 2.8:
        return "AVERAGE OFFENSE"
    else:
        return "WEAK OFFENSE"


def _defense_tag(ga_per_gp: float) -> str:
    """Tag defensive quality level."""
    if ga_per_gp <= 2.5:
        return "ELITE DEFENSE"
    elif ga_per_gp <= 2.8:
        return "STRONG DEFENSE"
    elif ga_per_gp <= 3.2:
        return "AVERAGE DEFENSE"
    else:
        return "WEAK DEFENSE"


def _form_tag(l5_gf: float | None, season_gf: float) -> str:
    """Tag recent scoring form vs season baseline."""
    if l5_gf is None:
        return ""
    diff = l5_gf - season_gf
    if diff >= 0.5:
        return "HOT SCORING"
    elif diff <= -0.5:
        return "COLD SCORING"
    return ""


async def build_scoring_depth_context(games: list) -> str:
    """Build NHL scoring depth context for AI prompt.

    Args:
        games: list of dicts with {"away": "NYR", "home": "PIT", ...}

    Returns formatted text:
    === SCORING DEPTH ===
    RULE: Goal differential is the best predictor ...
      NYR: 3.42 GF/GP (#5) | 2.51 GA/GP (#8) | +0.91 diff | L5: 3.80 GF/GP — HOT SCORING
      PIT: 2.89 GF/GP (#20) | 3.34 GA/GP (#25) | -0.45 diff | L5: 2.40 GF/GP — COLD SCORING
      Matchup: NYR (+0.91 diff, #5 offense) vs PIT (-0.45 diff, #25 defense) = SCORING MISMATCH
    """
    if not games:
        return "=== SCORING DEPTH ===\nNo games to evaluate."

    lines = ["=== SCORING DEPTH ==="]
    lines.append(
        "RULE: Goal differential is the best predictor of future NHL results. "
        "GF/GP > 3.5 = elite offense. GA/GP < 2.5 = elite defense. "
        "L5 form diverging from season = regression or momentum signal."
    )
    lines.append("")

    for game in games:
        team_lines = {}
        for side_key in ("away", "home"):
            abbr = game.get(side_key, "?")
            scoring = await fetch_team_scoring(abbr)
            form = await fetch_recent_form(abbr)

            gf_gp = scoring["gf_per_gp"]
            ga_gp = scoring["ga_per_gp"]
            diff = scoring["goal_diff_per_gp"]
            gf_rank = scoring["gf_rank"]
            ga_rank = scoring["ga_rank"]
            l5_gf = form.get("l5_gf_per_gp")
            l5_ga = form.get("l5_ga_per_gp")

            # Build diff string with sign
            diff_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"

            # Tags
            off_tag = _offense_tag(gf_gp)
            def_tag = _defense_tag(ga_gp)
            scoring_form = _form_tag(l5_gf, gf_gp)

            # L5 display
            l5_display = f"{l5_gf:.2f} GF/GP" if l5_gf is not None else "N/A"

            # Combine tags
            tags = [off_tag, def_tag]
            if scoring_form:
                tags.append(scoring_form)
            tag_str = " | ".join(tags)

            line = (
                f"  {abbr}: {gf_gp:.2f} GF/GP (#{gf_rank}) | "
                f"{ga_gp:.2f} GA/GP (#{ga_rank}) | "
                f"{diff_str} diff | L5: {l5_display}"
            )
            if scoring_form:
                line += f" — {scoring_form}"

            lines.append(line)
            team_lines[side_key] = {
                "abbr": abbr,
                "diff": diff,
                "diff_str": diff_str,
                "gf_rank": gf_rank,
                "ga_rank": ga_rank,
                "off_tag": off_tag,
                "def_tag": def_tag,
            }

        # Matchup summary
        away = team_lines.get("away", {})
        home = team_lines.get("home", {})
        if away and home:
            away_abbr = away["abbr"]
            home_abbr = home["abbr"]
            away_diff = away["diff"]
            home_diff = home["diff"]

            # Determine if there's a scoring mismatch
            gap = abs(away_diff - home_diff)
            if gap >= 0.8:
                matchup_tag = "SCORING MISMATCH"
            elif gap >= 0.4:
                matchup_tag = "MODERATE EDGE"
            else:
                matchup_tag = "EVEN MATCHUP"

            lines.append(
                f"  Matchup: {away_abbr} ({away['diff_str']} diff, #{away['gf_rank']} offense) "
                f"vs {home_abbr} ({home['diff_str']} diff, #{home['ga_rank']} defense) "
                f"= {matchup_tag}"
            )

        lines.append("")

    return "\n".join(lines)

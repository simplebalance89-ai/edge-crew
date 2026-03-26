"""
MLB Defense Engine — Team fielding stats from MLB Stats API.

MLB Stats API: https://statsapi.mlb.com/api/v1/
Free, public, no API key required. Rate limit: ~60 req/min.

Used by: server.py analysis prompt (defensive context injection)
Cache: 6 hours for fielding stats
"""

import httpx
import logging
from datetime import datetime, timezone

from engines.mlb_pitcher_engine import (
    MLB_STATS_BASE,
    MLB_TEAM_IDS,
    _get_cached,
    _set_cache,
)

logger = logging.getLogger("edge-crew")


async def fetch_team_fielding(team_abbr: str) -> dict:
    """Fetch team fielding stats from MLB Stats API.

    Endpoint: /teams/{id}/stats?stats=season&season=YYYY&group=fielding

    Returns: {"fielding_pct": ".985", "errors": 42, "errors_per_game": "0.8",
              "double_plays": 45, "assists": 620, "putouts": 1850}
    Cache: 6 hours
    """
    team_id = MLB_TEAM_IDS.get(team_abbr)
    if not team_id:
        return {
            "fielding_pct": "-",
            "errors": 0,
            "errors_per_game": "-",
            "double_plays": 0,
            "assists": 0,
            "putouts": 0,
        }

    cache_key = f"mlb_fielding:{team_abbr}"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    season = datetime.now(timezone.utc).year
    url = f"{MLB_STATS_BASE}/teams/{team_id}/stats"
    params = {
        "stats": "season",
        "season": season,
        "group": "fielding",
    }

    result = {
        "fielding_pct": "-",
        "errors": 0,
        "errors_per_game": "-",
        "double_plays": 0,
        "assists": 0,
        "putouts": 0,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[MLB DEFENSE] Fielding fetch failed for {team_abbr}: {e}")
        return result

    for stat_group in data.get("stats", []):
        if stat_group.get("type", {}).get("displayName", "") != "season":
            continue
        splits = stat_group.get("splits", [])
        if not splits:
            continue

        s = splits[0].get("stat", {})
        errors = int(s.get("errors", 0))
        games_played = int(s.get("gamesPlayed", 1)) or 1
        double_plays = int(s.get("doublePlays", 0))
        assists = int(s.get("assists", 0))
        putouts = int(s.get("putOuts", 0))
        fielding_pct = s.get("fielding", "-")

        result = {
            "fielding_pct": fielding_pct,
            "errors": errors,
            "errors_per_game": f"{errors / games_played:.1f}",
            "double_plays": double_plays,
            "assists": assists,
            "putouts": putouts,
        }
        break

    _set_cache(cache_key, result)
    return result


async def build_defense_context(games: list) -> str:
    """Build defensive context for AI prompt.

    Returns:
    === DEFENSIVE METRICS ===
    NYY: .987 FLD% | 0.6 E/G | 52 DP — STRONG defense
    BOS: .978 FLD% | 1.2 E/G | 38 DP — BELOW AVG defense
    """
    if not games:
        return "=== DEFENSIVE METRICS ===\nNo games to evaluate."

    lines = ["=== DEFENSIVE METRICS ==="]
    lines.append(
        "RULE: Defense matters most in low-scoring pitcher duels. "
        "FLD% < .980 or E/G > 1.0 = liability. Weight defense 5-7 in tight matchups."
    )
    lines.append("")

    for game in games:
        for side_key in ("away", "home"):
            abbr = game.get(side_key, "?")
            fielding = await fetch_team_fielding(abbr)

            fld_pct = fielding["fielding_pct"]
            epg = fielding["errors_per_game"]
            dp = fielding["double_plays"]

            # Determine rating
            try:
                fld_val = float(fld_pct)
                epg_val = float(epg)
            except (ValueError, TypeError):
                fld_val = 0.0
                epg_val = 99.0

            if fld_val >= 0.985 and epg_val <= 0.7:
                tag = "STRONG defense"
            elif fld_val < 0.980 or epg_val > 1.0:
                tag = "BELOW AVG defense"
            else:
                tag = "AVERAGE defense"

            lines.append(f"  {abbr}: {fld_pct} FLD% | {epg} E/G | {dp} DP — {tag}")

        lines.append("")

    return "\n".join(lines)

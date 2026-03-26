"""
MLB Pitcher Engine — Probable starters + pitcher stats from MLB Stats API.

MLB Stats API: https://statsapi.mlb.com/api/v1/
Free, public, no API key required. Rate limit: ~60 req/min.

Used by: server.py analysis prompt (pitcher context injection)
Cache: 30 min for probable pitchers, 2 hr for season stats
"""

import httpx
import logging
from datetime import datetime, timezone

logger = logging.getLogger("edge-crew")

MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"

# MLB team ID → abbreviation (all 30 teams)
MLB_TEAM_ABBR = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

# Module-level cache (same pattern as server.py _cache)
_pitcher_cache = {}


def _get_cached(key, ttl=1800):
    if key in _pitcher_cache:
        data, ts = _pitcher_cache[key]
        if (datetime.now(timezone.utc) - ts).total_seconds() < ttl:
            return data
    return None


def _set_cache(key, data):
    _pitcher_cache[key] = (data, datetime.now(timezone.utc))


async def fetch_probable_pitchers(date_str: str = None) -> list:
    """Fetch today's probable pitchers from MLB Stats API.

    Endpoint: /schedule?sportId=1&date=YYYY-MM-DD&hydrate=probablePitcher(note)

    Returns list of dicts:
        [{"game_id": 12345, "away": "NYY", "home": "BOS",
          "away_pitcher": {"name": "Gerrit Cole", "id": 543037, ...},
          "home_pitcher": {"name": "Chris Sale", "id": 519242, ...}}, ...]
    """
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cache_key = f"mlb_probable:{date_str}"
    cached = _get_cached(cache_key, ttl=1800)  # 30 min
    if cached:
        return cached

    url = f"{MLB_STATS_BASE}/schedule"
    params = {
        "sportId": 1,
        "date": date_str,
        "hydrate": "probablePitcher(note),linescore",
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error(f"[MLB PITCHER] Schedule fetch failed: {e}")
        return []

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            game_id = game.get("gamePk")
            status = game.get("status", {}).get("abstractGameState", "")
            if status == "Final":
                continue  # Skip finished games

            away_team = game.get("teams", {}).get("away", {})
            home_team = game.get("teams", {}).get("home", {})

            away_id = away_team.get("team", {}).get("id", 0)
            home_id = home_team.get("team", {}).get("id", 0)
            away_abbr = MLB_TEAM_ABBR.get(away_id, away_team.get("team", {}).get("name", "?"))
            home_abbr = MLB_TEAM_ABBR.get(home_id, home_team.get("team", {}).get("name", "?"))

            away_pitcher = _extract_pitcher(away_team.get("probablePitcher"))
            home_pitcher = _extract_pitcher(home_team.get("probablePitcher"))

            games.append({
                "game_id": game_id,
                "away": away_abbr,
                "home": home_abbr,
                "away_pitcher": away_pitcher,
                "home_pitcher": home_pitcher,
            })

    _set_cache(cache_key, games)
    logger.info(f"[MLB PITCHER] Fetched {len(games)} games with probable pitchers for {date_str}")
    return games


def _extract_pitcher(pitcher_data: dict) -> dict:
    """Extract pitcher info from MLB Stats API probablePitcher object.
    Note: schedule endpoint only gives id/fullName/link. Hand comes from stats call."""
    if not pitcher_data:
        return {"name": "TBD", "id": None, "hand": "?"}

    return {
        "name": pitcher_data.get("fullName", "TBD"),
        "id": pitcher_data.get("id"),
        "hand": "?",  # Populated later from /people/{id} endpoint
    }


async def _fetch_pitcher_hand(pitcher_id: int) -> str:
    """Fetch pitcher throwing hand from /people/{id}."""
    if not pitcher_id:
        return "?"
    cache_key = f"mlb_hand:{pitcher_id}"
    cached = _get_cached(cache_key, ttl=86400)  # 24hr — hand doesn't change
    if cached:
        return cached
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(f"{MLB_STATS_BASE}/people/{pitcher_id}")
            resp.raise_for_status()
            data = resp.json()
            people = data.get("people", [])
            if people:
                hand = people[0].get("pitchHand", {}).get("code", "?")
                _set_cache(cache_key, hand)
                return hand
    except Exception:
        pass
    return "?"


async def fetch_pitcher_season_stats(pitcher_id: int, season: int = None) -> dict:
    """Fetch season + recent stats for a pitcher.

    Endpoint: /people/{id}/stats?stats=season,gameLog&season=YYYY&group=pitching

    Returns: {"era": "3.41", "whip": "1.12", "k_per_9": "10.2", "w": 5, "l": 2,
              "ip": "89.1", "l5_era": "2.89", "l5_k_per_9": "11.1", "games_started": 15}
    """
    if not pitcher_id:
        return {}

    if not season:
        season = datetime.now(timezone.utc).year

    cache_key = f"mlb_pitcher_stats:{pitcher_id}:{season}"
    cached = _get_cached(cache_key, ttl=7200)  # 2 hours
    if cached:
        return cached

    url = f"{MLB_STATS_BASE}/people/{pitcher_id}/stats"
    params = {
        "stats": "season,gameLog",
        "season": season,
        "group": "pitching",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[MLB PITCHER] Stats fetch failed for pitcher {pitcher_id}: {e}")
        return {}

    stats = {}
    for stat_group in data.get("stats", []):
        stat_type = stat_group.get("type", {}).get("displayName", "")

        if stat_type == "season":
            splits = stat_group.get("splits", [])
            if splits:
                s = splits[0].get("stat", {})
                stats["era"] = s.get("era", "-")
                stats["whip"] = s.get("whip", "-")
                stats["k_per_9"] = s.get("strikeoutsPer9Inn", "-")
                stats["bb_per_9"] = s.get("walksPer9Inn", "-")
                stats["w"] = s.get("wins", 0)
                stats["l"] = s.get("losses", 0)
                stats["ip"] = s.get("inningsPitched", "0")
                stats["games_started"] = s.get("gamesStarted", 0)
                stats["avg"] = s.get("avg", "-")
                stats["ops"] = s.get("ops", "-")

        elif stat_type == "gameLog":
            splits = stat_group.get("splits", [])
            # Last 5 starts
            recent = splits[:5] if splits else []
            if recent:
                total_er = 0
                total_ip = 0.0
                total_k = 0
                for g in recent:
                    gs = g.get("stat", {})
                    ip_str = gs.get("inningsPitched", "0")
                    try:
                        ip_val = float(ip_str)
                    except (ValueError, TypeError):
                        ip_val = 0.0
                    total_ip += ip_val
                    total_er += int(gs.get("earnedRuns", 0))
                    total_k += int(gs.get("strikeOuts", 0))

                if total_ip > 0:
                    stats["l5_era"] = f"{(total_er / total_ip * 9):.2f}"
                    stats["l5_k_per_9"] = f"{(total_k / total_ip * 9):.1f}"
                    stats["l5_ip_avg"] = f"{(total_ip / len(recent)):.1f}"
                else:
                    stats["l5_era"] = "-"
                    stats["l5_k_per_9"] = "-"
                    stats["l5_ip_avg"] = "-"

                stats["l5_starts"] = len(recent)

    _set_cache(cache_key, stats)
    return stats


async def build_pitcher_context(date_str: str = None) -> str:
    """Build the pitcher context block for the AI analysis prompt.

    Returns formatted text like:
    === PROBABLE PITCHERS (MLB Stats API — verified) ===
    NYY @ BOS:
      Away SP: Gerrit Cole (R) | ERA 3.41 | WHIP 1.12 | K/9 10.2 | L5: 2.89 ERA
      Home SP: Chris Sale (L) | ERA 2.89 | WHIP 0.98 | K/9 11.4 | L5: 2.45 ERA
    """
    games = await fetch_probable_pitchers(date_str)
    if not games:
        return "=== PROBABLE PITCHERS ===\nNo MLB games scheduled or pitcher data unavailable."

    lines = ["=== PROBABLE PITCHERS (MLB Stats API — verified) ==="]
    lines.append("RULE: Starting pitcher is the #1 edge variable in MLB. Score starting_pitcher 8-10 for elite SPs (ERA < 3.00, K/9 > 9). Score 2-4 for weak SPs (ERA > 5.00).")
    lines.append("")

    for game in games:
        matchup = f"{game['away']} @ {game['home']}"
        lines.append(f"{matchup}:")

        for side, key in [("Away", "away_pitcher"), ("Home", "home_pitcher")]:
            pitcher = game[key]
            if pitcher["id"]:
                stats = await fetch_pitcher_season_stats(pitcher["id"])
                hand = await _fetch_pitcher_hand(pitcher["id"])
                era = stats.get("era", "-")
                whip = stats.get("whip", "-")
                k9 = stats.get("k_per_9", "-")
                record = f"{stats.get('w', 0)}-{stats.get('l', 0)}"
                l5_era = stats.get("l5_era", "-")
                l5_k9 = stats.get("l5_k_per_9", "-")
                ip = stats.get("ip", "-")

                lines.append(
                    f"  {side} SP: {pitcher['name']} ({hand}) | "
                    f"ERA {era} | WHIP {whip} | K/9 {k9} | W-L {record} | IP {ip} | "
                    f"L5: {l5_era} ERA, {l5_k9} K/9"
                )
            else:
                lines.append(f"  {side} SP: TBD — not yet announced")

        lines.append("")

    return "\n".join(lines)

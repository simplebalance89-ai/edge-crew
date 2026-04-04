"""
Soccer Attack & Defense Metrics Engine — Team goals, BTTS, O2.5, clean sheets.

ESPN Standings API: https://site.api.espn.com/apis/v2/sports/soccer/{league}/standings
Free, public, no API key required.

Used by: server.py analysis prompt (soccer attack/defense context injection)
Cache: 6 hours for season stats
"""

import httpx
import logging
from datetime import datetime, timezone

from engines.mlb_pitcher_engine import _get_cached, _set_cache

logger = logging.getLogger("edge-crew")

# ESPN soccer league slugs → display names
SOCCER_LEAGUES = {
    "eng.1": "Premier League",
    "esp.1": "La Liga",
    "ger.1": "Bundesliga",
    "ita.1": "Serie A",
    "fra.1": "Ligue 1",
    "usa.1": "MLS",
    "eng.2": "Championship",
    "uefa.champions": "Champions League",
    "uefa.europa": "Europa League",
}

ESPN_SOCCER_BASE = "https://site.api.espn.com/apis/v2/sports/soccer"

# ESPN scoreboard base (for completed match scores → BTTS / O2.5 calculation)
ESPN_SCOREBOARD_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"


async def _fetch_standings(league: str) -> list:
    """Fetch team standings from ESPN soccer standings API.

    Returns list of dicts with team season stats:
        [{"name": "Arsenal", "abbr": "ARS", "gp": 30, "gf": 72, "ga": 24,
          "wins": 22, "draws": 5, "losses": 3, "rank": 1}, ...]

    Cache: 6 hours
    """
    cache_key = f"soccer_standings:{league}"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    url = f"{ESPN_SOCCER_BASE}/{league}/standings"

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[SOCCER ATTACK] Standings fetch failed for {league}: {e}")
        return []

    teams = []
    for child in data.get("children", []):
        for entry in child.get("standings", {}).get("entries", []):
            team_info = entry.get("team", {})
            name = team_info.get("displayName", team_info.get("name", "?"))
            abbr = team_info.get("abbreviation", "?")

            # Parse stats from the standings entry
            stats_map = {}
            for stat in entry.get("stats", []):
                stats_map[stat.get("name", "")] = stat.get("value", 0)

            gp = int(stats_map.get("gamesPlayed", 0))
            gf = int(stats_map.get("pointsFor", 0))
            ga = int(stats_map.get("pointsAgainst", 0))
            wins = int(stats_map.get("wins", 0))
            draws = int(stats_map.get("ties", stats_map.get("draws", 0)))
            losses = int(stats_map.get("losses", 0))
            rank = int(stats_map.get("rank", 0))

            teams.append({
                "name": name,
                "abbr": abbr,
                "gp": gp,
                "gf": gf,
                "ga": ga,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "rank": rank,
            })

    _set_cache(cache_key, teams)
    logger.info(f"[SOCCER ATTACK] Fetched standings for {league}: {len(teams)} teams")
    return teams


async def _fetch_team_scores(league: str, team_name: str, season: int = None) -> list:
    """Fetch completed match scores for a team from ESPN scoreboard.

    Scans recent scoreboard data to extract final scores for BTTS/O2.5 calculation.

    Returns list of tuples: [(goals_for, goals_against), ...]
    Cache: 6 hours
    """
    if not season:
        season = datetime.now(timezone.utc).year

    cache_key = f"soccer_scores:{league}:{team_name}:{season}"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    url = f"{ESPN_SCOREBOARD_BASE}/{league}/scoreboard"
    params = {
        "limit": 200,
        "dates": str(season),
    }

    scores = []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[SOCCER ATTACK] Scoreboard fetch failed for {league}: {e}")
        return scores

    team_lower = team_name.lower()

    for event in data.get("events", []):
        status = event.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FULL_TIME" and status != "STATUS_FINAL":
            continue

        competitors = event.get("competitions", [{}])[0].get("competitors", [])
        if len(competitors) < 2:
            continue

        team_found = False
        team_goals = 0
        opp_goals = 0

        for comp in competitors:
            comp_name = comp.get("team", {}).get("displayName", "").lower()
            comp_abbr = comp.get("team", {}).get("abbreviation", "").lower()
            comp_score = int(comp.get("score", 0))

            if team_lower in comp_name or team_lower == comp_abbr:
                team_found = True
                team_goals = comp_score
            else:
                opp_goals = comp_score

        if team_found:
            scores.append((team_goals, opp_goals))

    _set_cache(cache_key, scores)
    return scores


def _compute_attack_defense_metrics(team: dict, scores: list) -> dict:
    """Compute attack/defense metrics from standings + match scores.

    Returns dict with all calculated metrics for a single team.
    """
    gp = team["gp"] or 1
    gf = team["gf"]
    ga = team["ga"]

    gpg = round(gf / gp, 2)   # Goals per game
    gcpg = round(ga / gp, 2)  # Goals conceded per game

    # Clean sheets and BTTS / O2.5 from individual match scores
    clean_sheets = 0
    btts_count = 0
    over25_count = 0
    total_matches = len(scores) if scores else 0

    for gf_match, ga_match in scores:
        if ga_match == 0:
            clean_sheets += 1
        if gf_match > 0 and ga_match > 0:
            btts_count += 1
        if gf_match + ga_match >= 3:
            over25_count += 1

    # Fall back to standings-based estimates if no individual scores available
    if total_matches == 0:
        total_matches = gp
        # Estimate clean sheets from GA (rough: CS ~ games where GA/GP < 1)
        clean_sheets = max(0, gp - ga) if ga < gp else 0
        # Estimate BTTS: if team scores AND concedes on average
        btts_count = int(gp * 0.55) if gpg > 0.5 and gcpg > 0.5 else int(gp * 0.35)
        # Estimate O2.5: based on combined goal rate
        combined = gpg + gcpg
        if combined > 3.0:
            over25_count = int(gp * 0.70)
        elif combined > 2.5:
            over25_count = int(gp * 0.55)
        else:
            over25_count = int(gp * 0.40)

    cs_pct = round((clean_sheets / total_matches) * 100) if total_matches else 0
    btts_pct = round((btts_count / total_matches) * 100) if total_matches else 0
    over25_pct = round((over25_count / total_matches) * 100) if total_matches else 0

    # xG proxy: goals scored vs expected from shot volume
    # Without shot data, use GPG vs league average (~1.3) as rough proxy
    xg_proxy = round(gpg / 1.3, 2) if gpg > 0 else 0.0

    # Attack tag
    if gpg >= 2.0:
        attack_tag = "ELITE ATTACK"
    elif gpg >= 1.5:
        attack_tag = "STRONG ATTACK"
    elif gpg >= 1.2:
        attack_tag = "AVERAGE ATTACK"
    else:
        attack_tag = "WEAK ATTACK"

    # Defense tag
    if gcpg < 1.0:
        defense_tag = "FORTRESS DEFENSE"
    elif gcpg <= 1.2:
        defense_tag = "SOLID DEFENSE"
    elif gcpg <= 1.5:
        defense_tag = "AVERAGE DEFENSE"
    else:
        defense_tag = "LEAKY DEFENSE"

    return {
        "name": team["name"],
        "abbr": team["abbr"],
        "gp": gp,
        "gpg": gpg,
        "gcpg": gcpg,
        "gf": gf,
        "ga": ga,
        "clean_sheets": clean_sheets,
        "cs_pct": cs_pct,
        "btts_pct": btts_pct,
        "over25_pct": over25_pct,
        "xg_proxy": xg_proxy,
        "attack_tag": attack_tag,
        "defense_tag": defense_tag,
        "rank": team.get("rank", 0),
    }


async def fetch_team_attack_defense(
    team_name: str, league: str = "eng.1"
) -> dict:
    """Fetch attack & defense metrics for a single team.

    Looks up team in standings, fetches match scores, computes metrics.

    Returns: {"name": "Arsenal", "abbr": "ARS", "gpg": 2.4, "gcpg": 0.8,
              "cs_pct": 47, "btts_pct": 53, "over25_pct": 67,
              "attack_tag": "ELITE ATTACK", "defense_tag": "FORTRESS DEFENSE", ...}
    Cache: 6 hours (via standings + scores caches)
    """
    cache_key = f"soccer_attack_defense:{league}:{team_name}"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    standings = await _fetch_standings(league)
    if not standings:
        return _empty_metrics(team_name)

    # Find team in standings (fuzzy match on name or abbreviation)
    team_lower = team_name.lower()
    team = None
    for t in standings:
        if (team_lower in t["name"].lower()
                or team_lower == t["abbr"].lower()
                or t["name"].lower() in team_lower):
            team = t
            break

    if not team:
        logger.warning(f"[SOCCER ATTACK] Team '{team_name}' not found in {league} standings")
        return _empty_metrics(team_name)

    # Fetch individual match scores for BTTS / O2.5 calculation
    scores = await _fetch_team_scores(league, team["name"])

    result = _compute_attack_defense_metrics(team, scores)
    _set_cache(cache_key, result)
    return result


def _empty_metrics(team_name: str) -> dict:
    """Return empty metrics dict when data is unavailable."""
    return {
        "name": team_name,
        "abbr": "?",
        "gp": 0,
        "gpg": 0.0,
        "gcpg": 0.0,
        "gf": 0,
        "ga": 0,
        "clean_sheets": 0,
        "cs_pct": 0,
        "btts_pct": 0,
        "over25_pct": 0,
        "xg_proxy": 0.0,
        "attack_tag": "N/A",
        "defense_tag": "N/A",
        "rank": 0,
    }


def _rank_suffix(rank: int) -> str:
    """Return rank with ordinal suffix: 1 → '#1', 14 → '#14'."""
    return f"#{rank}" if rank else "??"


async def build_attack_defense_context(games: list, league: str = "eng.1") -> str:
    """Build attack & defense context for AI prompt.

    Args:
        games: list of dicts with {"away": "Team Name", "home": "Team Name"}
        league: ESPN league slug (default: eng.1 for Premier League)

    Returns formatted text like:
    === ATTACK & DEFENSE ===
    RULE: BTTS rate > 65% for both teams = BTTS LOCK ...
      Arsenal: 2.4 GPG (#2) | 0.8 GC/GP (#1) | 47% CS | BTTS: 53% | O2.5: 67% — ELITE ATTACK + FORTRESS DEFENSE
      Fulham: 1.3 GPG (#14) | 1.5 GC/GP (#12) | 20% CS | BTTS: 70% | O2.5: 60% — AVERAGE ATTACK, LEAKY DEFENSE
      BTTS signal: ...
      Over/Under: ...
    """
    if not games:
        return "=== ATTACK & DEFENSE ===\nNo games to evaluate."

    lines = ["=== ATTACK & DEFENSE ==="]
    lines.append(
        "RULE: BTTS rate > 65% for both teams = BTTS LOCK chain signal. "
        "Over 2.5 rate > 70% for both = over lean. "
        "Clean sheet rate > 40% = elite defense, under lean. "
        "Goals per game > 2.0 = attacking team. "
        "Goals conceded < 1.0 = fortress defense."
    )
    lines.append("")

    for game in games:
        away_name = game.get("away", "?")
        home_name = game.get("home", "?")

        away = await fetch_team_attack_defense(away_name, league)
        home = await fetch_team_attack_defense(home_name, league)

        # Format each team line
        for metrics in (home, away):
            tag = f"{metrics['attack_tag']} + {metrics['defense_tag']}"
            lines.append(
                f"  {metrics['name']}: "
                f"{metrics['gpg']} GPG ({_rank_suffix(metrics['rank'])}) | "
                f"{metrics['gcpg']} GC/GP | "
                f"{metrics['cs_pct']}% clean sheets | "
                f"BTTS: {metrics['btts_pct']}% | "
                f"O2.5: {metrics['over25_pct']}% — {tag}"
            )

        # BTTS signal
        btts_signal = _btts_signal(home["btts_pct"], away["btts_pct"])
        lines.append(f"  BTTS signal: {home['name']} BTTS {home['btts_pct']}% vs "
                      f"{away['name']} BTTS {away['btts_pct']}% = {btts_signal}")

        # Over/Under signal
        combined_gpg = round(home["gpg"] + away["gpg"], 1)
        combined_gcpg = round(home["gcpg"] + away["gcpg"], 1)
        ou_signal = _over_under_signal(home, away)
        lines.append(f"  Over/Under: Combined {combined_gpg} GPG + "
                      f"{combined_gcpg} GC/GP = {ou_signal}")

        lines.append("")

    return "\n".join(lines)


def _btts_signal(home_btts: int, away_btts: int) -> str:
    """Determine BTTS betting signal from both teams' BTTS rates."""
    if home_btts >= 65 and away_btts >= 65:
        return "BTTS LOCK"
    elif home_btts >= 65 or away_btts >= 65:
        return "MIXED"
    elif home_btts <= 40 and away_btts <= 40:
        return "BTTS NO LEAN"
    else:
        return "NEUTRAL"


def _over_under_signal(home: dict, away: dict) -> str:
    """Determine over/under signal from combined metrics."""
    avg_over25 = (home["over25_pct"] + away["over25_pct"]) / 2
    combined_gpg = home["gpg"] + away["gpg"]

    # Strong over indicators
    if avg_over25 >= 70 and combined_gpg >= 3.0:
        return "STRONG OVER LEAN"
    elif avg_over25 >= 60 or combined_gpg >= 3.0:
        return "OVER LEAN"

    # Strong under indicators
    if home["cs_pct"] >= 40 and away["cs_pct"] >= 40:
        return "UNDER LEAN (both elite defense)"
    if avg_over25 <= 40 and combined_gpg <= 2.0:
        return "STRONG UNDER LEAN"
    elif avg_over25 <= 45 or combined_gpg <= 2.2:
        return "UNDER LEAN"

    return "NEUTRAL"

"""
NBA Pace & Matchup Engine — Team offensive/defensive ratings and pace context.

ESPN API: https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/
Free, public, no API key required.

Used by: server.py analysis prompt (pace/matchup context injection)
Cache: 6 hours for season stats (move slowly)
"""

import httpx
import logging
from datetime import datetime, timezone

from engines.mlb_pitcher_engine import _get_cached, _set_cache

logger = logging.getLogger("edge-crew")

ESPN_NBA_STATS_URL = (
    "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/"
    "statistics/byteam"
)
ESPN_NBA_STANDINGS_URL = (
    "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
)

# ESPN NBA team abbreviation mapping (displayName -> abbreviation)
# ESPN uses its own abbreviations; we normalize to common 3-letter codes.
NBA_TEAM_ABBR_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

# Reverse: abbreviation -> full name (for display flexibility)
NBA_ABBR_TO_NAME = {v: k for k, v in NBA_TEAM_ABBR_MAP.items()}


def _normalize_nba_abbr(raw: str) -> str:
    """Normalize team name/abbreviation to a standard 3-letter code."""
    raw = raw.strip()
    # Already a known abbreviation
    if raw.upper() in NBA_ABBR_TO_NAME:
        return raw.upper()
    # Full name lookup
    if raw in NBA_TEAM_ABBR_MAP:
        return NBA_TEAM_ABBR_MAP[raw]
    # Partial match
    raw_lower = raw.lower()
    for name, abbr in NBA_TEAM_ABBR_MAP.items():
        if raw_lower in name.lower():
            return abbr
    return raw.upper()[:3]


async def fetch_nba_team_stats() -> dict:
    """Fetch season team stats from ESPN NBA statistics endpoint.

    Returns dict keyed by team abbreviation:
        {"LAL": {"ppg": 114.2, "opp_ppg": 109.5, "rank_ppg": 8, "rank_opp_ppg": 12,
                 "fg3a_pg": 35.1, "fg3_pct": 0.367}, ...}
    Cache: 6 hours
    """
    cache_key = "nba_team_stats:season"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    season = datetime.now(timezone.utc).year
    params = {"season": season, "seasontype": 2}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(ESPN_NBA_STATS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NBA MATCHUP] ESPN team stats fetch failed: {e}")
        return {}

    result = {}

    # ESPN returns categories -> each has a list of teams with stats
    categories = data.get("categories", [])
    # Build a working dict keyed by team displayName
    team_data = {}

    for cat in categories:
        cat_name = cat.get("name", "")
        for team_entry in cat.get("teams", []):
            team_info = team_entry.get("team", {})
            team_name = team_info.get("displayName", "")
            if not team_name:
                continue
            if team_name not in team_data:
                team_data[team_name] = {}

            # Each team has stats array aligned with category labels
            stats = team_entry.get("stats", [])
            labels = cat.get("labels", [])
            for i, label in enumerate(labels):
                if i < len(stats):
                    team_data[team_name][f"{cat_name}:{label}"] = stats[i]

    # Parse the relevant stats out of the collected data
    for team_name, stats in team_data.items():
        abbr = _normalize_nba_abbr(team_name)
        ppg = _safe_float(stats.get("offensive:PTS", stats.get("scoring:PTS")))
        opp_ppg = _safe_float(stats.get("defensive:PTS", stats.get("scoring:OPTS")))
        fg3a_pg = _safe_float(stats.get("offensive:3PA", stats.get("scoring:3PA")))
        fg3_pct = _safe_float(stats.get("offensive:3P%", stats.get("scoring:3P%")))

        if ppg > 0:
            result[abbr] = {
                "ppg": ppg,
                "opp_ppg": opp_ppg if opp_ppg > 0 else 0.0,
                "fg3a_pg": fg3a_pg,
                "fg3_pct": fg3_pct,
            }

    # If ESPN stats endpoint didn't yield data, try standings as fallback
    if not result:
        result = await _fetch_stats_from_standings()

    # Compute rankings
    if result:
        _compute_rankings(result)

    _set_cache(cache_key, result)
    logger.info(f"[NBA MATCHUP] Fetched stats for {len(result)} teams")
    return result


async def _fetch_stats_from_standings() -> dict:
    """Fallback: derive PPG and OppPPG from ESPN standings data."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(ESPN_NBA_STANDINGS_URL)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[NBA MATCHUP] Standings fallback failed: {e}")
        return {}

    result = {}
    for child in data.get("children", []):
        for entry in child.get("standings", {}).get("entries", []):
            team_info = entry.get("team", {})
            team_name = team_info.get("displayName", "")
            abbr = _normalize_nba_abbr(team_name) if team_name else ""
            if not abbr:
                continue

            stats_map = {}
            for s in entry.get("stats", []):
                stats_map[s.get("name", "")] = s.get("value", 0)

            ppg = _safe_float(stats_map.get("pointsFor", 0))
            pa = _safe_float(stats_map.get("pointsAgainst", 0))
            gp = _safe_float(stats_map.get("gamesPlayed", 1)) or 1

            if ppg > 0:
                result[abbr] = {
                    "ppg": round(ppg / gp, 1) if ppg > 200 else ppg,
                    "opp_ppg": round(pa / gp, 1) if pa > 200 else pa,
                    "fg3a_pg": 0.0,
                    "fg3_pct": 0.0,
                }

    return result


def _compute_rankings(stats: dict):
    """Add rank_ppg and rank_opp_ppg to each team entry."""
    # Rank PPG: highest = #1
    sorted_ppg = sorted(stats.keys(), key=lambda a: stats[a].get("ppg", 0), reverse=True)
    for i, abbr in enumerate(sorted_ppg, 1):
        stats[abbr]["rank_ppg"] = i

    # Rank OppPPG: lowest = #1 (best defense)
    sorted_opp = sorted(stats.keys(), key=lambda a: stats[a].get("opp_ppg", 999))
    for i, abbr in enumerate(sorted_opp, 1):
        stats[abbr]["rank_opp_ppg"] = i


def _safe_float(val) -> float:
    """Safely convert a value to float."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


async def fetch_nba_recent_form(team_abbr: str, games: list = None) -> dict:
    """Compute L5 PPG and L5 OppPPG from the games list passed by the pipeline.

    If games list is provided (from server.py scores archive), compute from that.
    Otherwise return empty defaults.

    Returns: {"l5_ppg": 118.4, "l5_opp_ppg": 106.2, "l5_games": 5}
    Cache: 6 hours
    """
    cache_key = f"nba_recent:{team_abbr}"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    result = {"l5_ppg": 0.0, "l5_opp_ppg": 0.0, "l5_games": 0}

    if not games:
        return result

    # games expected as list of dicts with "home", "away", "home_score", "away_score"
    team_games = []
    for g in games:
        home = _normalize_nba_abbr(g.get("home_team", g.get("home", "")))
        away = _normalize_nba_abbr(g.get("away_team", g.get("away", "")))
        home_score = _safe_float(g.get("home_score", 0))
        away_score = _safe_float(g.get("away_score", 0))

        if home == team_abbr and home_score > 0:
            team_games.append({"scored": home_score, "allowed": away_score})
        elif away == team_abbr and away_score > 0:
            team_games.append({"scored": away_score, "allowed": home_score})

    recent = team_games[:5]
    if recent:
        total_scored = sum(g["scored"] for g in recent)
        total_allowed = sum(g["allowed"] for g in recent)
        count = len(recent)
        result = {
            "l5_ppg": round(total_scored / count, 1),
            "l5_opp_ppg": round(total_allowed / count, 1),
            "l5_games": count,
        }

    _set_cache(cache_key, result)
    return result


def _offense_tag(season_ppg: float, l5_ppg: float) -> str:
    """Qualitative tag for offensive form."""
    if l5_ppg > 0 and l5_ppg > season_ppg + 5:
        return "HOT OFFENSE"
    if l5_ppg > 0 and l5_ppg < season_ppg - 5:
        return "COLD OFFENSE"
    return ""


def _defense_tag(opp_ppg: float) -> str:
    """Qualitative tag for defensive quality."""
    if opp_ppg > 0 and opp_ppg < 107:
        return "ELITE DEFENSE"
    if opp_ppg > 115:
        return "WEAK DEFENSE"
    return ""


def _matchup_tag(off_rank: int, def_rank: int, total_teams: int = 30) -> str:
    """Detect mismatch: top-10 offense vs bottom-10 defense."""
    if off_rank <= 10 and def_rank > (total_teams - 10):
        return "MISMATCH EDGE"
    return ""


async def build_matchup_context(games: list, recent_scores: list = None) -> str:
    """Build pace and matchup context for AI prompt.

    Args:
        games: list of game dicts with "away" and "home" keys (team abbreviations)
        recent_scores: optional list of recent completed game dicts for L5 form

    Returns formatted text block:
    === PACE & MATCHUP ===
    RULE: Top-10 offense vs bottom-10 defense = score off_vs_def 8-9. ...

      LAL: 114.2 PPG (#8) | 109.5 OppPPG (#12) | Net +4.7 | L5: 118.4 PPG — HOT OFFENSE
      BOS: 108.1 PPG (#18) | 112.8 OppPPG (#22) | Net -4.7 | L5: 104.2 PPG — COLD OFFENSE
      Matchup: LAL offense (#8) vs BOS defense (#22) = MISMATCH EDGE
    """
    if not games:
        return "=== PACE & MATCHUP ===\nNo games to evaluate."

    team_stats = await fetch_nba_team_stats()
    if not team_stats:
        return "=== PACE & MATCHUP ===\nTeam stats unavailable — ESPN fetch failed."

    total_teams = len(team_stats) or 30

    lines = ["=== PACE & MATCHUP ==="]
    lines.append(
        "RULE: Top-10 offense vs bottom-10 defense = score off_vs_def 8-9. "
        "Pace mismatch > 8 PPG differential = tempo edge. "
        "Both elite defenses (< 105 PPG allowed) = under lean."
    )
    lines.append("")

    for game in games:
        away_abbr = _normalize_nba_abbr(game.get("away", "?"))
        home_abbr = _normalize_nba_abbr(game.get("home", "?"))

        matchup_lines = []

        for abbr in (away_abbr, home_abbr):
            ts = team_stats.get(abbr, {})
            ppg = ts.get("ppg", 0.0)
            opp_ppg = ts.get("opp_ppg", 0.0)
            rank_ppg = ts.get("rank_ppg", "?")
            rank_opp = ts.get("rank_opp_ppg", "?")
            net = round(ppg - opp_ppg, 1) if ppg and opp_ppg else 0.0
            net_str = f"+{net}" if net > 0 else str(net)

            # Recent form
            recent = await fetch_nba_recent_form(abbr, recent_scores)
            l5_ppg = recent.get("l5_ppg", 0.0)
            l5_opp_ppg = recent.get("l5_opp_ppg", 0.0)

            # Tags
            tags = []
            o_tag = _offense_tag(ppg, l5_ppg)
            if o_tag:
                tags.append(o_tag)
            d_tag = _defense_tag(opp_ppg)
            if d_tag:
                tags.append(d_tag)
            tag_str = " — " + ", ".join(tags) if tags else ""

            l5_str = f"L5: {l5_ppg} PPG" if l5_ppg > 0 else "L5: N/A"

            line = (
                f"  {abbr}: {ppg} PPG (#{rank_ppg}) | "
                f"{opp_ppg} OppPPG (#{rank_opp}) | "
                f"Net {net_str} | {l5_str}{tag_str}"
            )
            matchup_lines.append(line)

        # Add team lines
        lines.extend(matchup_lines)

        # Matchup edge detection: away offense vs home defense, and vice versa
        away_stats = team_stats.get(away_abbr, {})
        home_stats = team_stats.get(home_abbr, {})

        away_off_rank = away_stats.get("rank_ppg", 99)
        away_def_rank = away_stats.get("rank_opp_ppg", 99)
        home_off_rank = home_stats.get("rank_ppg", 99)
        home_def_rank = home_stats.get("rank_opp_ppg", 99)

        # Check away offense vs home defense
        edge1 = _matchup_tag(away_off_rank, home_def_rank, total_teams)
        if edge1:
            lines.append(
                f"  Matchup: {away_abbr} offense (#{away_off_rank}) vs "
                f"{home_abbr} defense (#{home_def_rank}) = {edge1}"
            )

        # Check home offense vs away defense
        edge2 = _matchup_tag(home_off_rank, away_def_rank, total_teams)
        if edge2:
            lines.append(
                f"  Matchup: {home_abbr} offense (#{home_off_rank}) vs "
                f"{away_abbr} defense (#{away_def_rank}) = {edge2}"
            )

        # Pace differential check
        away_ppg = away_stats.get("ppg", 0)
        home_ppg = home_stats.get("ppg", 0)
        if away_ppg and home_ppg:
            pace_diff = abs(away_ppg - home_ppg)
            if pace_diff > 8:
                faster = away_abbr if away_ppg > home_ppg else home_abbr
                lines.append(
                    f"  Pace gap: {pace_diff:.1f} PPG differential — "
                    f"{faster} tempo edge"
                )

        # Both elite defenses -> under lean
        away_opp = away_stats.get("opp_ppg", 999)
        home_opp = home_stats.get("opp_ppg", 999)
        if away_opp < 105 and home_opp < 105:
            lines.append(
                f"  Both elite defenses ({away_abbr} {away_opp}, "
                f"{home_abbr} {home_opp} OppPPG) — UNDER lean"
            )

        lines.append("")

    return "\n".join(lines)

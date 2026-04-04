"""
Soccer Form & League Engine — Standings, form, and positional data.

ESPN Soccer API: https://site.api.espn.com/apis/v2/sports/soccer/{league}/standings
Free, public, no API key required.

Fallback: API-Sports (v3.football.api-sports.io) — requires API_SPORTS_KEY env var.

Used by: server.py analysis prompt (soccer form context injection)
Cache: 6 hours for standings (move slowly)
"""

import httpx
import logging
import os
from datetime import datetime, timezone

from engines.mlb_pitcher_engine import _get_cached, _set_cache

logger = logging.getLogger("edge-crew")

ESPN_SOCCER_BASE = "https://site.api.espn.com/apis/v2/sports/soccer"
API_SPORTS_BASE = "https://v3.football.api-sports.io"

# League key -> (display name, ESPN slug, API-Sports league ID)
SOCCER_LEAGUES = {
    "eng.1":            ("EPL",        "eng.1",            39),
    "esp.1":            ("La Liga",    "esp.1",            140),
    "ger.1":            ("Bundesliga", "ger.1",            78),
    "ita.1":            ("Serie A",    "ita.1",            135),
    "fra.1":            ("Ligue 1",    "fra.1",            61),
    "usa.1":            ("MLS",        "usa.1",            253),
    "uefa.champions":   ("UCL",        "uefa.champions",   2),
}


def _form_tag(l5_form: str) -> str:
    """Derive a form tag from L5 string like 'WWDLW'."""
    wins = l5_form.upper().count("W")
    if wins >= 4:
        return "ELITE FORM"
    elif wins == 3:
        return "STRONG FORM"
    elif wins == 2:
        return "MIXED FORM"
    else:
        return "POOR FORM"


def _position_tag(rank: int, total_teams: int) -> str:
    """Derive positional tag from league rank."""
    if rank <= 4:
        return "TITLE CONTENDER"
    if total_teams and rank > total_teams - 3:
        return "RELEGATION ZONE"
    return ""


def _parse_espn_record(record_items: list) -> dict:
    """Parse ESPN record items into structured stats."""
    stats = {
        "wins": 0, "draws": 0, "losses": 0,
        "goals_for": 0, "goals_against": 0,
        "home_wins": 0, "home_draws": 0, "home_losses": 0,
        "away_wins": 0, "away_draws": 0, "away_losses": 0,
    }
    for item in record_items:
        rec_type = item.get("type", "")
        s = {st.get("name", ""): st.get("value", 0) for st in item.get("stats", [])}
        if rec_type == "total":
            stats["wins"] = int(s.get("wins", 0))
            stats["draws"] = int(s.get("ties", s.get("draws", 0)))
            stats["losses"] = int(s.get("losses", 0))
            stats["goals_for"] = int(s.get("pointsFor", s.get("goalsFor", 0)))
            stats["goals_against"] = int(s.get("pointsAgainst", s.get("goalsAgainst", 0)))
            stats["points"] = int(s.get("points", 0))
            stats["games_played"] = int(s.get("gamesPlayed", 0))
        elif rec_type == "home":
            stats["home_wins"] = int(s.get("wins", 0))
            stats["home_draws"] = int(s.get("ties", s.get("draws", 0)))
            stats["home_losses"] = int(s.get("losses", 0))
        elif rec_type == "away":
            stats["away_wins"] = int(s.get("wins", 0))
            stats["away_draws"] = int(s.get("ties", s.get("draws", 0)))
            stats["away_losses"] = int(s.get("losses", 0))
    return stats


async def fetch_league_standings(league_key: str) -> list:
    """Fetch standings for a soccer league from ESPN.

    Returns list of dicts per team:
        [{"team": "Arsenal", "rank": 1, "points": 72, "wins": 22, "draws": 6,
          "losses": 2, "goals_for": 68, "goals_against": 23, "goal_diff": 45,
          "home_record": "13W-2D-0L", "away_record": "9W-4D-2L",
          "l5_form": "WWDWW", "l5_gf": 12, "l5_ga": 3, "ppg": 2.40,
          "games_played": 30, "form_tag": "ELITE FORM", "position_tag": "TITLE CONTENDER"}, ...]
    Cache: 6 hours
    """
    league_info = SOCCER_LEAGUES.get(league_key)
    if not league_info:
        logger.warning(f"[SOCCER FORM] Unknown league key: {league_key}")
        return []

    display_name, espn_slug, api_sports_id = league_info

    cache_key = f"soccer_standings:{league_key}"
    cached = _get_cached(cache_key, ttl=21600)  # 6 hours
    if cached:
        return cached

    # Try ESPN first (free, no key)
    standings = await _fetch_espn_standings(espn_slug, league_key)

    # Fallback to API-Sports if ESPN returns nothing
    if not standings:
        standings = await _fetch_api_sports_standings(api_sports_id, league_key)

    if standings:
        _set_cache(cache_key, standings)
    return standings


async def _fetch_espn_standings(espn_slug: str, league_key: str) -> list:
    """Fetch standings from ESPN Soccer API."""
    url = f"{ESPN_SOCCER_BASE}/{espn_slug}/standings"

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[SOCCER FORM] ESPN standings fetch failed for {league_key}: {e}")
        return []

    standings = []
    children = data.get("children", [])
    # Some leagues have groups (e.g. UCL); flatten all entries
    all_entries = []
    if children:
        for group in children:
            for entry in group.get("standings", {}).get("entries", []):
                all_entries.append(entry)
    else:
        for entry in data.get("standings", {}).get("entries", []):
            all_entries.append(entry)

    total_teams = len(all_entries)

    for entry in all_entries:
        team_name = entry.get("team", {}).get("displayName", "Unknown")
        # Stats are in a flat list under 'stats'
        stat_map = {}
        for st in entry.get("stats", []):
            stat_map[st.get("name", "")] = st.get("value", 0)
            if st.get("displayValue"):
                stat_map[st.get("name", "") + "_display"] = st.get("displayValue", "")

        rank = int(stat_map.get("rank", 0))
        points = int(stat_map.get("points", 0))
        wins = int(stat_map.get("wins", 0))
        draws = int(stat_map.get("ties", stat_map.get("draws", 0)))
        losses = int(stat_map.get("losses", 0))
        goals_for = int(stat_map.get("pointsFor", stat_map.get("goalsFor", 0)))
        goals_against = int(stat_map.get("pointsAgainst", stat_map.get("goalsAgainst", 0)))
        goal_diff = int(stat_map.get("pointDifferential", stat_map.get("goalDifference", goals_for - goals_against)))
        games_played = int(stat_map.get("gamesPlayed", 0))

        # Home/away from record items
        record_items = entry.get("records", [])
        rec = _parse_espn_record(record_items)

        home_record = f"{rec['home_wins']}W-{rec['home_draws']}D-{rec['home_losses']}L"
        away_record = f"{rec['away_wins']}W-{rec['away_draws']}D-{rec['away_losses']}L"

        # L5 form string — ESPN provides it under 'form' or recent results
        # Check stats for a form display value
        l5_form = ""
        for st in entry.get("stats", []):
            if st.get("name") in ("deductions", "form"):
                continue
            if st.get("abbreviation") == "F" or st.get("name") == "form":
                l5_form = st.get("displayValue", "")
                break

        # If no form found in stats, try the 'note' or build from record
        if not l5_form:
            l5_form = "-----"

        # Normalize form to W/D/L characters only (ESPN uses W, D, L)
        l5_clean = "".join(c for c in l5_form.upper() if c in "WDL")[:5]
        if not l5_clean:
            l5_clean = "-----"

        # Estimate L5 goals (not always available from ESPN standings)
        l5_gf = 0
        l5_ga = 0

        ppg = round(points / games_played, 2) if games_played > 0 else 0.0

        form_tag = _form_tag(l5_clean)
        pos_tag = _position_tag(rank, total_teams)

        standings.append({
            "team": team_name,
            "rank": rank,
            "points": points,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "goal_diff": goal_diff,
            "home_record": home_record,
            "away_record": away_record,
            "l5_form": l5_clean,
            "l5_gf": l5_gf,
            "l5_ga": l5_ga,
            "ppg": ppg,
            "games_played": games_played,
            "form_tag": form_tag,
            "position_tag": pos_tag,
        })

    # Sort by rank
    standings.sort(key=lambda x: x.get("rank", 999))
    return standings


async def _fetch_api_sports_standings(league_id: int, league_key: str) -> list:
    """Fallback: Fetch standings from API-Sports (requires API_SPORTS_KEY)."""
    api_key = os.environ.get("API_SPORTS_KEY", "")
    if not api_key:
        logger.warning("[SOCCER FORM] API_SPORTS_KEY not set — skipping fallback.")
        return []

    season = datetime.now(timezone.utc).year
    url = f"{API_SPORTS_BASE}/standings"
    params = {"league": league_id, "season": season}
    headers = {"x-apisports-key": api_key}

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[SOCCER FORM] API-Sports standings fetch failed for {league_key}: {e}")
        return []

    standings = []
    response = data.get("response", [])
    if not response:
        return []

    league_data = response[0].get("league", {})
    groups = league_data.get("standings", [])

    for group in groups:
        total_teams = len(group)
        for entry in group:
            team_name = entry.get("team", {}).get("name", "Unknown")
            rank = entry.get("rank", 0)
            points = entry.get("points", 0)

            all_stats = entry.get("all", {})
            home_stats = entry.get("home", {})
            away_stats = entry.get("away", {})

            wins = all_stats.get("win", 0)
            draws = all_stats.get("draw", 0)
            losses = all_stats.get("lose", 0)
            goals_for = all_stats.get("goals", {}).get("for", 0)
            goals_against = all_stats.get("goals", {}).get("against", 0)
            goal_diff = entry.get("goalsDiff", goals_for - goals_against)
            games_played = all_stats.get("played", 0)

            hw = home_stats.get("win", 0)
            hd = home_stats.get("draw", 0)
            hl = home_stats.get("lose", 0)
            aw = away_stats.get("win", 0)
            ad = away_stats.get("draw", 0)
            al = away_stats.get("lose", 0)

            home_record = f"{hw}W-{hd}D-{hl}L"
            away_record = f"{aw}W-{ad}D-{al}L"

            # Form string from API-Sports
            l5_form = entry.get("form", "-----") or "-----"
            l5_clean = "".join(c for c in l5_form.upper() if c in "WDL")[:5]
            if not l5_clean:
                l5_clean = "-----"

            ppg = round(points / games_played, 2) if games_played > 0 else 0.0

            form_tag = _form_tag(l5_clean)
            pos_tag = _position_tag(rank, total_teams)

            standings.append({
                "team": team_name,
                "rank": rank,
                "points": points,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "goal_diff": goal_diff,
                "home_record": home_record,
                "away_record": away_record,
                "l5_form": l5_clean,
                "l5_gf": 0,
                "l5_ga": 0,
                "ppg": ppg,
                "games_played": games_played,
                "form_tag": form_tag,
                "position_tag": pos_tag,
            })

    standings.sort(key=lambda x: x.get("rank", 999))
    return standings


def _find_team_in_standings(standings: list, team_name: str) -> dict | None:
    """Fuzzy-match a team name against standings entries."""
    if not standings or not team_name:
        return None
    name_lower = team_name.lower().strip()
    for entry in standings:
        if name_lower in entry["team"].lower() or entry["team"].lower() in name_lower:
            return entry
    # Try partial token match
    tokens = name_lower.split()
    for entry in standings:
        entry_lower = entry["team"].lower()
        if any(tok in entry_lower for tok in tokens if len(tok) > 3):
            return entry
    return None


async def build_form_context(games: list) -> str:
    """Build soccer form & league position context for AI prompt.

    Expects games list with dicts containing:
        {"home": "Arsenal", "away": "Fulham", "league": "eng.1", ...}

    Returns formatted text like:
    === FORM & LEAGUE POSITION ===
    RULE: League position gap > 10 places = class gap, score 8-9 for higher team. ...

      Arsenal (EPL #1): 72 pts | 22W-6D-2L | GD +45 | Home: 13W-2D-0L | L5: WWDWW — ELITE FORM
      Fulham (EPL #12): 41 pts | 11W-8D-11L | GD +2 | Home: 7W-4D-4L | L5: LWDLW — MIXED FORM
      Position gap: 11 places | Class edge: Arsenal
    """
    if not games:
        return "=== FORM & LEAGUE POSITION ===\nNo games to evaluate."

    lines = ["=== FORM & LEAGUE POSITION ==="]
    lines.append(
        "RULE: League position gap > 10 places = class gap, score 8-9 for higher team. "
        "L5 form diverging from season position = momentum signal. "
        "Home record vs away record differential > 30% = significant home/away split. "
        "PPG > 2.0 = title contender. PPG < 1.0 = relegation form."
    )
    lines.append("")

    # Group games by league to avoid redundant fetches
    league_standings_cache = {}

    for game in games:
        league_key = game.get("league", "eng.1")
        league_info = SOCCER_LEAGUES.get(league_key)
        league_display = league_info[0] if league_info else league_key

        # Fetch standings for this league (cached across games)
        if league_key not in league_standings_cache:
            league_standings_cache[league_key] = await fetch_league_standings(league_key)
        standings = league_standings_cache[league_key]

        home_name = game.get("home", "?")
        away_name = game.get("away", "?")

        home_data = _find_team_in_standings(standings, home_name)
        away_data = _find_team_in_standings(standings, away_name)

        for team_name, data in [(home_name, home_data), (away_name, away_data)]:
            if data:
                gd_sign = "+" if data["goal_diff"] >= 0 else ""
                tags = data["form_tag"]
                if data["position_tag"]:
                    tags = f"{data['position_tag']} | {tags}"
                lines.append(
                    f"  {data['team']} ({league_display} #{data['rank']}): "
                    f"{data['points']} pts | "
                    f"{data['wins']}W-{data['draws']}D-{data['losses']}L | "
                    f"GD {gd_sign}{data['goal_diff']} | "
                    f"Home: {data['home_record']} | Away: {data['away_record']} | "
                    f"PPG {data['ppg']:.2f} | "
                    f"L5: {data['l5_form']} — {tags}"
                )
            else:
                lines.append(f"  {team_name}: standings data not available")

        # Position gap analysis
        if home_data and away_data:
            gap = abs(home_data["rank"] - away_data["rank"])
            if gap > 0:
                higher = home_data if home_data["rank"] < away_data["rank"] else away_data
                if gap > 10:
                    lines.append(f"  Position gap: {gap} places | Class edge: {higher['team']}")
                else:
                    lines.append(f"  Position gap: {gap} places")

        lines.append("")

    return "\n".join(lines)

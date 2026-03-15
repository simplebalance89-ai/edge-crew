"""Mathurin Screener — the core engine.

The Mathurin Test: If a player's FLOOR (min of L5 games) > the book's line,
that's guaranteed money. The gap between floor and line IS the edge.

Example: Mathurin L5 = [26, 28, 21, 22, 23]. Line = 17.5.
Floor (21) > Line (17.5) = 3.5 point gap = MISPRICED.
"""

import logging
from engines.odds import get_player_props, get_events
from engines.espn import fetch_player_game_logs, PROP_STAT_TO_ESPN, get_rosters
from engines.cache import get_cached, set_cache

logger = logging.getLogger("casa-ev")

# Map Odds API stat labels to ESPN stat header names
_STAT_MAP = {
    "Points": "PTS",
    "Rebounds": "REB",
    "Assists": "AST",
    "Threes": "3PM",
    "Three Pointers": "3PM",
    "Goals": "G",
    "Shots": "SOG",
    "Strikeouts": "K",
    "Total Bases": "TB",
    "Hits": "H",
    "Pass Yds": "YDS",
    "Rush Yds": "YDS",
    "Reception Yds": "YDS",
    "Pass Tds": "TD",
}


async def run_screener(sport: str):
    """Run the Mathurin Screener for every prop on tonight's slate.

    For each prop:
    1. Get the book's line (e.g., 17.5 points)
    2. Pull player's L5 raw values from ESPN
    3. Compute floor = min(L5)
    4. If floor > line → MISPRICED (guaranteed edge)
    5. Compute gap = floor - line

    Returns props sorted by gap size (biggest mispricings first).
    """
    sport_lower = sport.lower()

    cache_key = f"screener:{sport_lower}"
    cached = get_cached(cache_key, ttl=300)
    if cached:
        return cached

    # Step 1: Get all props from Odds API
    props_data = await get_player_props(sport)
    if "error" in props_data:
        return props_data
    all_props = props_data.get("props", [])

    if not all_props:
        return {"sport": sport.upper(), "screener": [], "count": 0, "message": "No props on slate"}

    # Step 2: Get teams playing tonight (from events)
    events = await get_events(sport)
    team_names = set()
    for e in events:
        team_names.add(e.get("away_team", ""))
        team_names.add(e.get("home_team", ""))

    # Step 3: Fetch game logs for all teams
    if not get_cached(f"rosters:{sport_lower}", ttl=3600):
        await get_rosters(sport)

    game_logs = await fetch_player_game_logs(sport, list(team_names))

    # Step 4: Screen each OVER prop against L5 floor
    screener_results = []
    for prop in all_props:
        if prop.get("side", "").lower() != "over":
            continue  # Only screen overs — that's where floor > line matters

        player = prop.get("player", "")
        stat = prop.get("stat", "")
        line = prop.get("line", 0)

        if line is None or line == 0:
            continue

        # Find this player in game logs
        player_data = game_logs.get(player)
        if not player_data or player_data.get("note"):
            continue

        # Map the prop stat to ESPN stat name
        espn_stat = _STAT_MAP.get(stat)
        if not espn_stat:
            continue

        raw = player_data.get("raw_stats", {}).get(espn_stat)
        if not raw or not raw.get("l5_raw"):
            continue

        l5_raw = raw["l5_raw"]
        l5_floor = min(l5_raw)
        l5_ceiling = max(l5_raw)
        l5_avg = raw.get("l5_avg", 0)
        l10_avg = raw.get("l10_avg", 0)

        # THE MATHURIN TEST
        gap = round(l5_floor - line, 1)
        mispriced = l5_floor > line

        # Hit rate: how many of L5 went over the line?
        hit_count = sum(1 for v in l5_raw if v > line)
        hit_rate = round(hit_count / len(l5_raw) * 100, 0)

        entry = {
            "player": player,
            "stat": stat,
            "team": player_data.get("team", "?"),
            "matchup": prop.get("matchup", ""),
            "commence": prop.get("commence", ""),
            "line": line,
            "l5_raw": l5_raw,
            "l5_floor": l5_floor,
            "l5_ceiling": l5_ceiling,
            "l5_avg": l5_avg,
            "l10_avg": l10_avg,
            "gap": gap,
            "mispriced": mispriced,
            "hit_rate": hit_rate,
            "best_odds": prop.get("best_odds"),
            "best_book": prop.get("best_book"),
            "best_prob": prop.get("best_prob"),
            "consensus_prob": prop.get("consensus_prob"),
            "edge": prop.get("edge", 0),
            "book_count": prop.get("book_count", 0),
            "books": prop.get("books", []),
            "verdict": "MISPRICED" if mispriced else ("CLOSE" if gap >= -1 else "PASS"),
        }
        screener_results.append(entry)

    # Sort by gap descending (biggest mispricings first)
    screener_results.sort(key=lambda x: -x["gap"])

    result = {
        "sport": sport.upper(),
        "screener": screener_results,
        "mispriced_count": sum(1 for s in screener_results if s["mispriced"]),
        "total_screened": len(screener_results),
        "games_on_slate": len(events),
        "players_with_logs": len(game_logs),
    }

    set_cache(cache_key, result)
    return result

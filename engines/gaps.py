"""Player Prop Gap Analyzer — Injury Absorption Engine.

Finds mispriced role player props when lineup gaps exist.
Books adjust star lines fast. They're slow on the role players who absorb those minutes.

Logic:
1. Pull injury report — who's OUT/GTD?
2. For each OUT player, calculate what they leave on the table (PPG, RPG, APG, MPG)
3. Identify the 2-3 players most likely to absorb (backup at same position, next in rotation)
4. Pull prop lines for those absorbers
5. Compare their L5 recent stats to the book's line
6. Flag where: recent avg >> book line AND there's a clear injury reason
"""

import logging
from engines.espn import fetch_injuries, fetch_player_game_logs, get_rosters
from engines.odds import get_player_props, get_events
from engines.cache import get_cached, set_cache

logger = logging.getLogger("casa-ev")

# Position groups — who backs up whom
_POSITION_GROUPS = {
    # NBA
    "PG": ["PG", "SG"],
    "SG": ["SG", "PG", "SF"],
    "SF": ["SF", "SG", "PF"],
    "PF": ["PF", "SF", "C"],
    "C": ["C", "PF"],
    # NHL
    "LW": ["LW", "RW", "C"],
    "RW": ["RW", "LW", "C"],
    "D": ["D"],
    "G": ["G"],
}


def _find_absorbers(injured_player_pos, roster_players, injured_name):
    """Find 2-3 players on same team most likely to absorb minutes.
    Returns players at same/adjacent positions, excluding the injured player."""
    eligible_positions = _POSITION_GROUPS.get(injured_player_pos, [injured_player_pos])
    absorbers = []
    for p in roster_players:
        if p["name"].lower() == injured_name.lower():
            continue
        if p.get("pos", "?") in eligible_positions:
            absorbers.append(p)
    return absorbers[:3]


async def find_gap_props(sport: str):
    """Find mispriced role player props driven by injury gaps.

    Returns props where:
    - A starter/key player is OUT or GTD
    - A role player is likely to absorb their minutes
    - That role player's recent stats (L5) exceed their current prop line
    - The book hasn't adjusted yet (line lag)
    """
    sport_lower = sport.lower()

    cache_key = f"gaps:{sport_lower}"
    cached = get_cached(cache_key, ttl=300)
    if cached:
        return cached

    # Step 1: Get injuries
    injuries = await fetch_injuries(sport)
    if not injuries:
        return {"sport": sport.upper(), "gaps": [], "count": 0, "message": "No injury data available"}

    # Filter to OUT and Day-to-Day only (not long-term IR)
    active_injuries = []
    for inj in injuries:
        status = (inj.get("status") or "").lower()
        if status in ["out", "day-to-day", "doubtful", "questionable"]:
            active_injuries.append(inj)

    if not active_injuries:
        return {"sport": sport.upper(), "gaps": [], "count": 0, "message": "No relevant injuries (OUT/DTD)"}

    # Step 2: Get rosters
    roster_data = await get_rosters(sport)
    rosters = roster_data.get("rosters", {})

    # Step 3: Get today's events and teams
    events = await get_events(sport)
    teams_playing = set()
    for e in events:
        teams_playing.add(e.get("away_team", ""))
        teams_playing.add(e.get("home_team", ""))

    # Step 4: Get prop lines
    props_data = await get_player_props(sport)
    all_props = props_data.get("props", [])

    # Build prop lookup: player_name_lower -> list of props
    prop_lookup = {}
    for p in all_props:
        pname = p.get("player", "").lower().strip()
        if pname not in prop_lookup:
            prop_lookup[pname] = []
        prop_lookup[pname].append(p)

    # Step 5: Get game logs for teams playing
    team_abbrevs = list(teams_playing)
    game_logs = await fetch_player_game_logs(sport, team_abbrevs)

    # Step 6: For each injury on a team playing tonight, find absorbers
    gap_props = []
    processed_injured = set()

    for inj in active_injuries:
        injured_name = inj.get("player", "")
        injured_team = inj.get("team", "")
        injury_status = inj.get("status", "")
        injury_detail = inj.get("detail", "")

        # Skip if team isn't playing tonight
        if injured_team not in teams_playing:
            continue

        # Skip duplicates
        inj_key = f"{injured_name}|{injured_team}"
        if inj_key in processed_injured:
            continue
        processed_injured.add(inj_key)

        # Get injured player's stats to quantify the gap
        injured_stats = game_logs.get(injured_name, {})
        injured_ppg = injured_stats.get("l5_pts", injured_stats.get("l10_pts", 0))
        injured_rpg = injured_stats.get("l5_reb", injured_stats.get("l10_reb", 0))
        injured_apg = injured_stats.get("l5_ast", injured_stats.get("l10_ast", 0))

        # Only care about impactful players being out (>10 PPG or significant role)
        if injured_ppg < 8 and injured_rpg < 5:
            continue

        # Find absorbers from roster
        team_roster = None
        injured_pos = "?"
        for abbr, info in rosters.items():
            if info.get("team", "").lower() == injured_team.lower() or abbr.lower() in injured_team.lower():
                team_roster = info
                # Find injured player's position
                for p in info.get("players", []):
                    if p["name"].lower() == injured_name.lower():
                        injured_pos = p.get("pos", "?")
                        break
                break

        if not team_roster:
            continue

        absorbers = _find_absorbers(injured_pos, team_roster.get("players", []), injured_name)

        # Step 7: Check each absorber's props vs their recent stats
        for absorber in absorbers:
            abs_name = absorber["name"]
            abs_props = prop_lookup.get(abs_name.lower().strip(), [])
            abs_logs = game_logs.get(abs_name, {})

            if not abs_props or not abs_logs or abs_logs.get("note"):
                continue

            # Skip if this player is already a high-minute starter (>30 MPG)
            raw_stats = abs_logs.get("raw_stats", {})
            min_stats = raw_stats.get("MIN", {})
            if min_stats and min_stats.get("l5_avg", 0) > 32:
                continue

            for prop in abs_props:
                if prop.get("side", "").lower() != "over":
                    continue

                stat = prop.get("stat", "")
                line = prop.get("line", 0)
                if not line:
                    continue

                # Map stat to ESPN header
                stat_map = {
                    "Points": "PTS", "Rebounds": "REB", "Assists": "AST",
                    "Threes": "3PM", "Goals": "G",
                }
                espn_stat = stat_map.get(stat)
                if not espn_stat:
                    continue

                raw = raw_stats.get(espn_stat, {})
                if not raw or not raw.get("l5_raw"):
                    continue

                l5_raw = raw["l5_raw"]
                l5_avg = raw.get("l5_avg", 0)
                l5_floor = min(l5_raw)

                # THE GAP TEST: Is their recent production above the line?
                gap = round(l5_avg - line, 1)
                floor_gap = round(l5_floor - line, 1)
                hit_count = sum(1 for v in l5_raw if v > line)
                hit_rate = round(hit_count / len(l5_raw) * 100, 0)

                # Only flag if average exceeds line OR hit rate >= 60%
                if l5_avg <= line and hit_rate < 60:
                    continue

                # Score the gap opportunity
                # Gap Size: how many stats the injured player leaves
                gap_size_score = min(10, round(injured_ppg / 3))
                # Absorption: hit rate as proxy
                absorption_score = min(10, round(hit_rate / 10))
                # Line Lag: bigger gap between recent avg and line = more lag
                line_lag_score = min(10, round(max(0, gap) * 2))
                # Combined
                combined_score = round((gap_size_score + absorption_score + line_lag_score) / 3, 1)

                # Grade
                if combined_score >= 8:
                    grade = "A"
                elif combined_score >= 6:
                    grade = "B"
                elif combined_score >= 4:
                    grade = "C"
                else:
                    grade = "D"

                gap_props.append({
                    "player": abs_name,
                    "team": abs_logs.get("team", "?"),
                    "position": absorber.get("pos", "?"),
                    "stat": stat,
                    "line": line,
                    "l5_raw": l5_raw,
                    "l5_avg": l5_avg,
                    "l5_floor": l5_floor,
                    "gap": gap,
                    "floor_gap": floor_gap,
                    "hit_rate": hit_rate,
                    "matchup": prop.get("matchup", ""),
                    "commence": prop.get("commence", ""),
                    "best_odds": prop.get("best_odds"),
                    "best_book": prop.get("best_book"),
                    "book_count": prop.get("book_count", 0),
                    # Gap context
                    "injured_player": injured_name,
                    "injured_status": injury_status,
                    "injured_detail": injury_detail,
                    "injured_ppg": injured_ppg,
                    "injured_rpg": injured_rpg,
                    "injured_pos": injured_pos,
                    # Scoring
                    "gap_size_score": gap_size_score,
                    "absorption_score": absorption_score,
                    "line_lag_score": line_lag_score,
                    "combined_score": combined_score,
                    "grade": grade,
                })

    # Sort by grade then combined score
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    gap_props.sort(key=lambda x: (grade_order.get(x["grade"], 9), -x["combined_score"]))

    result = {
        "sport": sport.upper(),
        "gaps": gap_props,
        "count": len(gap_props),
        "injuries_scanned": len(active_injuries),
        "teams_playing": len(teams_playing),
    }

    set_cache(cache_key, result)
    return result

"""
Prop Tracker — Frequency-Based Edge Scoring

Tracks player stat averages (L5/L10) vs book lines.
Finds where the book is lagging behind real performance.
Edge score stacks dimensions (1-8 scale):

DIMENSIONS:
  +1 = L5 avg 20-25% over line
  +2 = L5 avg 25-50% over line
  +3 = L5 avg 50%+ over line
  +1 = Crushed line last game
  +2 = Crushed line last 2 games
  +1 = Opponent bottom 10 defending that stat
  +2 = Opponent bottom 5
  +1 = Opposing starter OUT at same position
  +1 = Star OUT (affects all opposing players)
  +1 = Injury announced < 2 hours before tip

DECISION: 5+ = LOCK | 3-4 = Strong | 1-2 = Monitor

Stores data in data/prop_history.json (growing cache).
"""

import json
import os
import sys
import time
from datetime import datetime

import requests
from app_config import require_env
from paths import DATA_DIR

BDL_KEY = os.environ.get("BALLDONTLIE_API_KEY")
BDL_BASE = "https://api.balldontlie.io/v1"


# ─── Player Stats Cache ─────────────────────────────────────────────────────────

def load_prop_history() -> dict:
    """Load persistent prop tracking history."""
    hist_file = DATA_DIR / "prop_history.json"
    if hist_file.exists():
        with open(hist_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"players": {}, "last_updated": None}


def save_prop_history(history: dict):
    """Save prop tracking history."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history["last_updated"] = datetime.now().isoformat()
    with open(DATA_DIR / "prop_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# ─── BDL Player Stats Fetcher ───────────────────────────────────────────────────

def search_player(name: str) -> dict | None:
    """Search BDL for a player by name. Tries last name first, then first name."""
    headers = {"Authorization": require_env("BALLDONTLIE_API_KEY", "BallDontLie prop tracker")}
    # BDL search works best with last name
    parts = name.strip().split()
    search_terms = [name]  # Try full name first
    if len(parts) >= 2:
        search_terms.append(parts[-1])  # Last name
        search_terms.append(parts[0])   # First name

    for term in search_terms:
        try:
            resp = requests.get(
                f"{BDL_BASE}/players",
                headers=headers,
                params={"search": term, "per_page": 10},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if data:
                # Try to match full name
                name_lower = name.lower()
                for p in data:
                    full = f"{p.get('first_name', '')} {p.get('last_name', '')}".lower()
                    if name_lower in full or full in name_lower:
                        return p
                # If no exact match, try partial
                for p in data:
                    if parts[-1].lower() in p.get("last_name", "").lower():
                        return p
                return data[0]
        except Exception as e:
            print(f"[props] Error searching '{term}': {e}")
    return None


# ESPN player ID mapping (top players — grows over time)
ESPN_PLAYER_IDS = {
    "trae young": 4277905, "luka doncic": 4395725, "lebron james": 1966,
    "anthony davis": 6583, "nikola jokic": 3112335, "jamal murray": 3936299,
    "julius randle": 3064514, "anthony edwards": 4594327,
    "deandre ayton": 4395726, "anfernee simons": 4395648,
    "christian braun": 4683634, "jayson tatum": 4065648,
    "jalen green": 4432810, "collin sexton": 4278049,
    "keyonte george": 5105625, "bennedict mathurin": 4683023,
    "desmond bane": 4432158, "jaren jackson jr": 4395724,
    "michael porter jr": 4278104, "myles turner": 3064440,
    "clint capela": 2991230, "austin reaves": 4683750,
    "alperen sengun": 4698392, "scoot henderson": 5105610,
    "dejounte murray": 3978, "trey murphy iii": 4592180,
}


def fetch_player_stats_espn(player_name: str, n_games: int = 15) -> list[dict]:
    """Fetch recent game logs from ESPN (free, current season)."""
    # Try ESPN ID lookup
    espn_id = ESPN_PLAYER_IDS.get(player_name.lower())
    if not espn_id:
        # Try partial match
        for name, pid in ESPN_PLAYER_IDS.items():
            if player_name.lower() in name or name in player_name.lower():
                espn_id = pid
                break

    if not espn_id:
        print(f"[props] No ESPN ID for {player_name} — add to ESPN_PLAYER_IDS")
        return []

    url = f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{espn_id}/gamelog"
    try:
        resp = requests.get(url, params={"season": 2026}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[props] ESPN gamelog error for {player_name}: {e}")
        return []

    # Labels: MIN, FG, FG%, 3PT, 3P%, FT, FT%, REB, AST, BLK, STL, PF, TO, PTS
    label_map = {
        "PTS": 13, "REB": 7, "AST": 8, "BLK": 9, "STL": 10, "TO": 12,
        "3PM": 3,  # 3PT field is "made-attempted", need to parse
        "MIN": 0,
    }

    games = []
    for st in data.get("seasonTypes", []):
        if "Regular" not in st.get("displayName", ""):
            continue
        for cat in st.get("categories", []):
            for event in cat.get("events", []):
                stats_raw = event.get("stats", [])
                if len(stats_raw) < 14:
                    continue

                game = {"date": event.get("gameDate", "")}
                for stat_name, idx in label_map.items():
                    val = stats_raw[idx] if idx < len(stats_raw) else 0
                    if stat_name == "3PM":
                        # Parse "3-5" format to get makes
                        try:
                            game["fg3m"] = int(str(val).split("-")[0])
                        except (ValueError, IndexError):
                            game["fg3m"] = 0
                    elif stat_name == "MIN":
                        game["min"] = val
                    else:
                        try:
                            game[stat_name.lower()] = int(float(val))
                        except (ValueError, TypeError):
                            game[stat_name.lower()] = 0

                games.append(game)

    # Already ordered most recent first from ESPN
    return games[:n_games]


def fetch_player_stats(player_id: int, n_games: int = 15) -> list[dict]:
    """Fetch recent game logs — tries ESPN first, BDL fallback."""
    # This is now a passthrough — smart_track uses fetch_player_stats_espn directly
    headers = {"Authorization": require_env("BALLDONTLIE_API_KEY", "BallDontLie prop tracker")}
    try:
        resp = requests.get(
            f"{BDL_BASE}/stats",
            headers=headers,
            params={
                "player_ids[]": player_id,
                "per_page": n_games,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        data.sort(key=lambda g: g.get("game", {}).get("date", ""), reverse=True)
        return data[:n_games]
    except Exception as e:
        print(f"[props] Error fetching stats for player {player_id}: {e}")
        return []


def calculate_averages(game_logs: list[dict], stat: str) -> dict:
    """Calculate L5/L10/L15 averages for a stat."""
    stat_map = {
        "PTS": "pts", "REB": "reb", "AST": "ast",
        "3PM": "fg3m", "STL": "stl", "BLK": "blk",
        "TO": "turnover", "PRA": None,  # PTS+REB+AST combo
    }

    field = stat_map.get(stat)

    values = []
    for g in game_logs:
        if stat == "PRA":
            val = (g.get("pts") or 0) + (g.get("reb") or 0) + (g.get("ast") or 0)
        elif field:
            val = g.get(field)
        else:
            continue
        if val is not None:
            values.append(val)

    if not values:
        return {"L5": 0, "L10": 0, "L15": 0, "last": 0, "last2": [], "games": 0}

    l5 = values[:5]
    l10 = values[:10]
    l15 = values[:15]

    return {
        "L5": round(sum(l5) / max(len(l5), 1), 1),
        "L10": round(sum(l10) / max(len(l10), 1), 1),
        "L15": round(sum(l15) / max(len(l15), 1), 1),
        "last": values[0] if values else 0,
        "last2": values[:2],
        "last5_values": l5,
        "games": len(values),
    }


# ─── Edge Scoring (8 Dimensions) ────────────────────────────────────────────────

def score_prop_edge(
    avgs: dict,
    line: float,
    opp_def_rank: int | None = None,
    opp_injuries: list[dict] | None = None,
    fresh_injury: bool = False,
) -> dict:
    """
    Calculate edge score for a prop.
    Returns score (0-8+), breakdown, and decision.
    """
    edge = 0
    breakdown = []
    l5_avg = avgs.get("L5", 0)
    last = avgs.get("last", 0)
    last2 = avgs.get("last2", [])

    if not line or not l5_avg:
        return {"edge": 0, "breakdown": ["No data"], "decision": "SKIP"}

    # ─── Dimension 1: L5 Average vs Line ─────────────────────────────────
    gap_pct = (l5_avg - line) / line * 100 if line else 0

    if gap_pct >= 50:
        edge += 3
        breakdown.append(f"L5 avg {l5_avg} is {gap_pct:.0f}% OVER line {line} (+3)")
    elif gap_pct >= 25:
        edge += 2
        breakdown.append(f"L5 avg {l5_avg} is {gap_pct:.0f}% over line {line} (+2)")
    elif gap_pct >= 20:
        edge += 1
        breakdown.append(f"L5 avg {l5_avg} is {gap_pct:.0f}% over line {line} (+1)")
    elif gap_pct <= -20:
        breakdown.append(f"L5 avg {l5_avg} is {abs(gap_pct):.0f}% UNDER line {line} — fade")
    else:
        breakdown.append(f"L5 avg {l5_avg} vs line {line} ({gap_pct:+.0f}%) — no gap")

    # ─── Dimension 2: Crushed Line Recently ──────────────────────────────
    if last > line * 1.3:
        edge += 1
        breakdown.append(f"Crushed line last game: {last} vs {line} (+1)")
    if len(last2) >= 2 and all(v > line for v in last2):
        edge += 1
        breakdown.append(f"Over line last 2 games: {last2} (+1)")

    # ─── Dimension 3: Opponent Defensive Ranking ─────────────────────────
    if opp_def_rank is not None:
        if opp_def_rank >= 26:  # Bottom 5
            edge += 2
            breakdown.append(f"OPP ranked #{opp_def_rank} defending this stat (+2)")
        elif opp_def_rank >= 21:  # Bottom 10
            edge += 1
            breakdown.append(f"OPP ranked #{opp_def_rank} defending this stat (+1)")

    # ─── Dimension 4: Opposing Injuries ──────────────────────────────────
    if opp_injuries:
        starter_out = sum(1 for i in opp_injuries if i.get("status") == "OUT" and (i.get("ppg") or 0) >= 12)
        star_out = sum(1 for i in opp_injuries if i.get("status") == "OUT" and (i.get("ppg") or 0) >= 20)

        if star_out >= 1:
            edge += 1
            breakdown.append(f"Opposing star OUT — usage boost (+1)")
        if starter_out >= 2:
            edge += 1
            breakdown.append(f"{starter_out} opposing starters OUT — expanded role (+1)")

    # ─── Dimension 5: Fresh Injury (< 2 hours) ──────────────────────────
    if fresh_injury:
        edge += 1
        breakdown.append("FRESH injury on opponent — book hasn't adjusted (+1)")

    # ─── Consistency Check ───────────────────────────────────────────────
    l5_values = avgs.get("last5_values", [])
    if l5_values and line:
        times_over = sum(1 for v in l5_values if v > line)
        consistency = times_over / len(l5_values)
        breakdown.append(f"Hit rate L5: {times_over}/{len(l5_values)} ({consistency:.0%})")

        # Bonus for extreme consistency
        if consistency >= 0.8 and edge >= 2:
            edge += 1
            breakdown.append("80%+ hit rate with edge — reliability bonus (+1)")

    # ─── Decision ────────────────────────────────────────────────────────
    if edge >= 5:
        decision = "LOCK"
    elif edge >= 3:
        decision = "STRONG"
    elif edge >= 1:
        decision = "MONITOR"
    else:
        decision = "PASS"

    return {
        "edge": edge,
        "gap_pct": round(gap_pct, 1),
        "breakdown": breakdown,
        "decision": decision,
        "L5_avg": l5_avg,
        "L10_avg": avgs.get("L10", 0),
        "line": line,
        "last_game": last,
        "hit_rate_L5": f"{sum(1 for v in l5_values if v > line)}/{len(l5_values)}" if l5_values else "?",
    }


# ─── Track a Player Prop ────────────────────────────────────────────────────────

def track_prop(
    player_name: str,
    stat: str,
    line: float,
    opp_team: str | None = None,
    opp_def_rank: int | None = None,
    opp_injuries: list[dict] | None = None,
    fresh_injury: bool = False,
) -> dict:
    """
    Full prop tracking pipeline:
    1. Find player in BDL
    2. Pull last 15 game logs
    3. Calculate L5/L10/L15 averages
    4. Score edge (8 dimensions)
    5. Return structured result
    """
    # Search player
    player = search_player(player_name)
    if not player:
        return {"error": f"Player not found: {player_name}", "player": player_name, "stat": stat}

    player_id = player.get("id")
    full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()

    # Fetch game logs
    game_logs = fetch_player_stats(player_id, 15)
    if not game_logs:
        return {"error": f"No game logs for {full_name}", "player": full_name, "stat": stat}

    # Calculate averages
    avgs = calculate_averages(game_logs, stat)

    # Score edge
    edge_result = score_prop_edge(
        avgs, line,
        opp_def_rank=opp_def_rank,
        opp_injuries=opp_injuries,
        fresh_injury=fresh_injury,
    )

    result = {
        "player": full_name,
        "player_id": player_id,
        "team": player.get("team", {}).get("full_name", "?"),
        "stat": stat,
        "line": line,
        "opponent": opp_team,
        **edge_result,
        "tracked_at": datetime.now().isoformat(),
    }

    # Save to history
    history = load_prop_history()
    key = f"{full_name}_{stat}_{datetime.now().strftime('%Y%m%d')}"
    if "players" not in history:
        history["players"] = {}
    history["players"][key] = result
    save_prop_history(history)

    return result


# ─── Batch Track ─────────────────────────────────────────────────────────────────

def batch_track(props: list[dict]) -> list[dict]:
    """
    Track multiple props at once.
    Each prop: {"player": "Name", "stat": "PTS", "line": 28.5, "opponent": "DAL", ...}
    """
    results = []
    for prop in props:
        result = track_prop(
            player_name=prop["player"],
            stat=prop["stat"],
            line=prop["line"],
            opp_team=prop.get("opponent"),
            opp_def_rank=prop.get("opp_def_rank"),
            opp_injuries=prop.get("opp_injuries"),
            fresh_injury=prop.get("fresh_injury", False),
        )
        results.append(result)

        # Print summary
        if "error" not in result:
            print(f"[prop] {result['player']:20s} | {result['stat']:4s} | Line: {result['line']:5.1f} | L5: {result['L5_avg']:5.1f} | Gap: {result['gap_pct']:+5.1f}% | Edge: {result['edge']} | {result['decision']} | {result['hit_rate_L5']}")
        else:
            print(f"[prop] {prop['player']:20s} | {result.get('error', '?')}")

        time.sleep(0.5)  # Rate limit

    return results


# ─── Auto-Wire: Load Game Context (injuries + profiles) ─────────────────────────

def load_game_context(player_team: str, opponent: str, sport: str = "nba") -> dict:
    """
    Auto-load injuries and team profiles from today's game data.
    Returns opponent injuries + team profiles for the prop tracker.
    """
    date = datetime.now().strftime("%Y%m%d")
    data_file = DATA_DIR / f"games_{sport}_{date}.json"
    if not data_file.exists():
        return {}

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for game in data.get("games", []):
        away = game.get("away", "")
        home = game.get("home", "")

        # Match by team name substring
        player_side = None
        opp_side = None
        if player_team and (player_team.lower() in away.lower() or away.lower() in player_team.lower()):
            player_side = "away"
            opp_side = "home"
        elif player_team and (player_team.lower() in home.lower() or home.lower() in player_team.lower()):
            player_side = "home"
            opp_side = "away"
        elif opponent and (opponent.lower() in away.lower()):
            opp_side = "away"
            player_side = "home"
        elif opponent and (opponent.lower() in home.lower()):
            opp_side = "home"
            player_side = "away"
        else:
            continue

        opp_injuries = game.get("injuries", {}).get(opp_side, [])
        opp_profile = game.get(f"{opp_side}_profile", {})
        our_profile = game.get(f"{player_side}_profile", {})

        # Check for fresh injuries on opponent
        fresh_injuries = [
            i for i in opp_injuries
            if i.get("status") == "OUT" and i.get("freshness") == "FRESH"
        ]

        return {
            "opp_injuries": opp_injuries,
            "opp_profile": opp_profile,
            "our_profile": our_profile,
            "fresh_injury": len(fresh_injuries) > 0,
            "opp_ppg_allowed": opp_profile.get("opp_ppg_L5"),
            "opp_margin": opp_profile.get("margin_L5"),
            "game_id": game.get("game_id"),
            "matchup": f"{away} @ {home}",
        }

    return {}


def smart_track(
    player_name: str,
    stat: str,
    line: float,
    opponent: str | None = None,
    sport: str = "nba",
) -> dict:
    """
    Full smart tracking — auto-wires game context (injuries, profiles).
    This is what you call instead of track_prop() for game-aware analysis.
    """
    # Try ESPN game logs first (current season, accurate)
    espn_logs = fetch_player_stats_espn(player_name, 15)

    # Search BDL for team info
    player = search_player(player_name)
    team_name = player.get("team", {}).get("full_name", "") if player else ""

    # Auto-load game context
    ctx = load_game_context(team_name, opponent, sport)

    # Calculate averages from ESPN data
    if espn_logs:
        avgs = calculate_averages(espn_logs, stat)
        edge_result = score_prop_edge(
            avgs, line,
            opp_injuries=ctx.get("opp_injuries"),
            fresh_injury=ctx.get("fresh_injury", False),
        )
        result = {
            "player": player_name,
            "team": team_name,
            "stat": stat,
            "line": line,
            "opponent": opponent,
            "source": "ESPN",
            **edge_result,
            "last5_values": avgs.get("last5_values", []),
            "tracked_at": datetime.now().isoformat(),
        }
    else:
        result = track_prop(
            player_name=player_name,
            stat=stat,
            line=line,
            opp_team=opponent,
            opp_injuries=ctx.get("opp_injuries"),
            fresh_injury=ctx.get("fresh_injury", False),
        )

    # Enrich with team profiles
    if ctx:
        result["game_context"] = {
            "matchup": ctx.get("matchup"),
            "opp_ppg_allowed_L5": ctx.get("opp_ppg_allowed"),
            "opp_margin_L5": ctx.get("opp_margin"),
            "opp_out_count": len([i for i in ctx.get("opp_injuries", []) if i.get("status") == "OUT"]),
            "opp_stars_out": [
                f"{i['player']} ({i.get('ppg', '?')} PPG)"
                for i in ctx.get("opp_injuries", [])
                if i.get("status") == "OUT" and (i.get("ppg") or 0) >= 15
            ],
            "fresh_injury_on_opp": ctx.get("fresh_injury", False),
            "our_record": ctx.get("our_profile", {}).get("record"),
            "our_L5": ctx.get("our_profile", {}).get("L5"),
            "our_ppg_L5": ctx.get("our_profile", {}).get("ppg_L5"),
        }

    # Mathurin Test (from Alyssa's math model)
    l5_values = result.get("last5_values") or (
        result.get("L5_avg") and []  # fallback
    )
    # Pull from history
    hist = load_prop_history()
    key = f"{player_name}_{stat}_{datetime.now().strftime('%Y%m%d')}"
    saved = hist.get("players", {}).get(key, {})

    return result


# ─── Mathurin Screener ───────────────────────────────────────────────────────────

def mathurin_test(player_name: str, stat: str, line: float) -> dict:
    """
    The Mathurin Test: If L5 FLOOR > book line, the line is MISPRICED.
    Player hit the over in ALL of their last 5 games.
    """
    # Use ESPN game logs (current season, accurate)
    game_logs = fetch_player_stats_espn(player_name, 10)
    if not game_logs:
        # Fallback to BDL
        player = search_player(player_name)
        if not player:
            return {"player": player_name, "verdict": "NO_DATA"}
        game_logs = fetch_player_stats(player.get("id"), 10)

    avgs = calculate_averages(game_logs, stat)
    l5_values = avgs.get("last5_values", [])

    if not l5_values:
        return {"player": player_name, "verdict": "NO_DATA"}

    l5_floor = min(l5_values)
    l5_ceiling = max(l5_values)
    hit_count = sum(1 for v in l5_values if v > line)
    hit_rate = round(hit_count / len(l5_values) * 100, 0)

    if l5_floor > line:
        verdict = "MISPRICED"
    elif l5_floor >= line - 1:
        verdict = "CLOSE"
    else:
        verdict = "PASS"

    return {
        "player": player_name,
        "stat": stat,
        "line": line,
        "L5_floor": l5_floor,
        "L5_ceiling": l5_ceiling,
        "L5_avg": avgs.get("L5", 0),
        "L5_values": l5_values,
        "hit_count": hit_count,
        "hit_rate": f"{hit_rate:.0f}%",
        "gap": round(l5_floor - line, 1),
        "verdict": verdict,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python prop_tracker.py <player_name> <stat> <line> [opponent]")
        print("  Example: python prop_tracker.py 'Trae Young' AST 10.5 DAL")
        print("  Stats: PTS, REB, AST, 3PM, STL, BLK, PRA")
        sys.exit(1)

    player = sys.argv[1]
    stat = sys.argv[2].upper()
    line = float(sys.argv[3])
    opp = sys.argv[4] if len(sys.argv) > 4 else None

    result = smart_track(player, stat, line, opponent=opp)
    print(json.dumps(result, indent=2, default=str))

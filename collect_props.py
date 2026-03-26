"""
Layer 1: Props Collector (NBA only)
Uses BallDontLie API for player props.
Merges into existing data/games_nba_{date}.json
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

BDL_KEY = os.environ.get("BALLDONTLIE_API_KEY")
BDL_BASE = "https://api.balldontlie.io/v1"

# BDL team abbreviation mapping
BDL_TEAM_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}
ABBREV_TO_TEAM = {abbr: team for team, abbr in BDL_TEAM_ABBREV.items()}


def _normalize_stat_type(stat_type: str) -> str:
    stat = (stat_type or "").strip().lower().replace(" ", "_")
    aliases = {
        "pts": "points",
        "player_points": "points",
        "reb": "rebounds",
        "player_rebounds": "rebounds",
        "ast": "assists",
        "player_assists": "assists",
        "3pt_made": "three_pt",
        "threes": "three_pt",
        "three_pointers_made": "three_pt",
        "fg3m": "three_pt",
        "points_rebounds_assists": "pra",
    }
    return aliases.get(stat, stat)


def _extract_line(prop: dict):
    for key in ("line_score", "line", "value"):
        val = prop.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
    return None


def _extract_player_name(prop: dict) -> str:
    player = prop.get("player", {}) if isinstance(prop.get("player"), dict) else {}
    for key in ("name", "full_name", "display_name"):
        if player.get(key):
            return str(player[key]).strip()
    first = player.get("first_name", "")
    last = player.get("last_name", "")
    full = f"{first} {last}".strip()
    return full


def _extract_player_team(prop: dict) -> str:
    player = prop.get("player", {}) if isinstance(prop.get("player"), dict) else {}
    team = player.get("team", {}) if isinstance(player.get("team"), dict) else {}
    for source in (team, prop):
        for key in ("team_abbreviation", "team", "team_abbr", "teamAbbreviation"):
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
    if isinstance(team.get("abbreviation"), str):
        return team.get("abbreviation")
    if isinstance(team.get("full_name"), str):
        return team.get("full_name")
    return ""


def _extract_player_pos(prop: dict) -> str:
    player = prop.get("player", {}) if isinstance(prop.get("player"), dict) else {}
    for key in ("position", "pos"):
        val = player.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _team_tokens(team_name: str) -> set[str]:
    tokens = {team_name.lower()}
    abbr = BDL_TEAM_ABBREV.get(team_name)
    if abbr:
        tokens.add(abbr.lower())
    return tokens


def _prop_matches_game(prop: dict, game: dict) -> bool:
    raw_team = _extract_player_team(prop).lower()
    if not raw_team:
        return False
    away_tokens = _team_tokens(game.get("away", ""))
    home_tokens = _team_tokens(game.get("home", ""))
    return raw_team in away_tokens or raw_team in home_tokens


def _build_prop_entry(prop: dict, game: dict) -> dict | None:
    player_name = _extract_player_name(prop)
    stat = _normalize_stat_type(prop.get("stat_type", ""))
    line = _extract_line(prop)
    if not player_name or not stat or line is None:
        return None

    raw_team = _extract_player_team(prop)
    team_name = ABBREV_TO_TEAM.get(raw_team, raw_team)
    if team_name == game.get("away"):
        opponent = game.get("home", "")
    elif team_name == game.get("home"):
        opponent = game.get("away", "")
    else:
        away_tokens = _team_tokens(game.get("away", ""))
        team_name_l = team_name.lower()
        if team_name_l in away_tokens:
            team_name = game.get("away", "")
            opponent = game.get("home", "")
        else:
            team_name = game.get("home", "")
            opponent = game.get("away", "")

    return {
        "player": player_name,
        "team": team_name,
        "opponent": opponent,
        "pos": _extract_player_pos(prop),
        "stat": stat,
        "line": line,
        "raw_stat": prop.get("stat_type", ""),
    }


def fetch_props(game_date: str) -> list[dict]:
    """Fetch player props from BDL for a given date."""
    headers = {"Authorization": BDL_KEY}
    url = f"{BDL_BASE}/props"
    params = {"date": game_date}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 404:
            print("[props] BDL props endpoint returned 404 — trying alternate")
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        print(f"[props] ERROR fetching props: {e}")
        return []


def fetch_player_game_logs(player_id: int, n_games: int = 10) -> list[dict]:
    """Fetch recent game logs for a player from BDL."""
    headers = {"Authorization": BDL_KEY}
    url = f"{BDL_BASE}/stats"
    params = {
        "player_ids[]": player_id,
        "per_page": n_games,
        "sort": "-game.date",
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        print(f"[props] ERROR fetching game logs for player {player_id}: {e}")
        return []


def calculate_prop_edge(prop: dict, game_logs: list[dict]) -> dict:
    """Calculate edge score for a prop based on recent performance."""
    stat_key = prop.get("stat_type", "").lower()
    line = prop.get("line", 0)
    player_name = prop.get("player", {}).get("first_name", "") + " " + prop.get("player", {}).get("last_name", "")

    if not stat_key or not line:
        return {}

    # Map BDL stat types to game log fields
    stat_map = {
        "points": "pts",
        "rebounds": "reb",
        "assists": "ast",
        "threes": "fg3m",
        "pts": "pts",
        "reb": "reb",
        "ast": "ast",
    }
    field = stat_map.get(stat_key, stat_key)

    # Calculate L5 and L10 averages
    values = [g.get(field, 0) for g in game_logs if g.get(field) is not None]
    l5_vals = values[:5]
    l10_vals = values[:10]

    l5_avg = round(sum(l5_vals) / max(len(l5_vals), 1), 1) if l5_vals else 0
    l10_avg = round(sum(l10_vals) / max(len(l10_vals), 1), 1) if l10_vals else 0

    # Gap percentage
    gap_pct = round((l5_avg - line) / max(line, 0.1) * 100, 1) if line else 0

    # Edge scoring (1-8 scale)
    edge = 0
    if gap_pct >= 50:
        edge += 3
    elif gap_pct >= 25:
        edge += 2
    elif gap_pct >= 20:
        edge += 1

    # Crushed line recently
    if l5_vals and l5_vals[0] > line * 1.2:
        edge += 1
    if len(l5_vals) >= 2 and l5_vals[0] > line and l5_vals[1] > line:
        edge += 1

    return {
        "player": player_name,
        "stat": stat_key.upper(),
        "line": line,
        "L5_avg": l5_avg,
        "L10_avg": l10_avg,
        "gap_pct": gap_pct,
        "edge_score": min(edge, 8),
        "values_L5": l5_vals,
    }


def collect(date: str | None = None) -> int:
    """Collect props for NBA games. Returns number of props found."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    date_formatted = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    data_file = DATA_DIR / f"games_nba_{date}.json"

    if not data_file.exists():
        print(f"[props] No NBA data file: {data_file}")
        return 0

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for game in data.get("games", []):
        game["props"] = []

    # Try BDL props endpoint
    raw_props = fetch_props(date_formatted)
    if raw_props:
        print(f"[props] Got {len(raw_props)} raw props from BDL")
        matched = 0
        for prop in raw_props:
            for game in data.get("games", []):
                if not _prop_matches_game(prop, game):
                    continue
                entry = _build_prop_entry(prop, game)
                if entry is not None:
                    game["props"].append(entry)
                    matched += 1
                break
        print(f"[props] Matched {matched} props into game payloads")
    else:
        print("[props] No props available from BDL (may not be posted yet)")

    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[props] NBA: {len(raw_props)} props -> {data_file.name}")
    return len(raw_props)


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    collect(date_arg)

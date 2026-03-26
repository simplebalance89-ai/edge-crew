"""
Layer 1: Injury Collector
Uses Tank01 (RapidAPI) for NBA/NHL roster scans.
Adds freshness tier and impact notes.
Merges into existing data/games_{sport}_{date}.json
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

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "409e417a5amshc8f88f3da5eb1c8p1b356bjsn648bb9e45f03")

TANK01_HOSTS = {
    "NBA": "tank01-fantasy-stats.p.rapidapi.com",
    "NHL": "tank01-nhl-live-in-game-real-time-statistics-nhl.p.rapidapi.com",
}

# Team name -> Tank01 abbreviation mapping (common ones)
NBA_TEAM_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GS", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NO", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SA",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

NHL_TEAM_ABBREV = {
    "Anaheim Ducks": "ANA", "Arizona Coyotes": "ARI", "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF", "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL", "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA", "Los Angeles Kings": "LA", "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL", "Nashville Predators": "NSH", "New Jersey Devils": "NJ",
    "New York Islanders": "NYI", "New York Rangers": "NYR", "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJ",
    "Seattle Kraken": "SEA", "St. Louis Blues": "STL", "Tampa Bay Lightning": "TB",
    "Toronto Maple Leafs": "TOR", "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
}

TEAM_ABBREVS = {"NBA": NBA_TEAM_ABBREV, "NHL": NHL_TEAM_ABBREV}


def get_freshness_tier(injury_date_str: str | None) -> tuple[str, int]:
    """Calculate freshness tier from injury date."""
    if not injury_date_str:
        return "UNKNOWN", 0
    try:
        injury_date = datetime.strptime(injury_date_str, "%Y-%m-%d")
        days = (datetime.now() - injury_date).days
    except (ValueError, TypeError):
        return "UNKNOWN", 0

    if days <= 3:
        return "FRESH", days
    elif days <= 14:
        return "RECENT", days
    elif days <= 30:
        return "ESTABLISHED", days
    else:
        return "SEASON", days


def fetch_injuries_tank01(sport: str, team_abbrev: str) -> list[dict]:
    """Fetch injuries from Tank01 for a specific team."""
    host = TANK01_HOSTS.get(sport)
    if not host:
        return []

    if sport == "NBA":
        url = f"https://{host}/getNBATeamRoster"
        params = {"teamAbv": team_abbrev}
    elif sport == "NHL":
        url = f"https://{host}/getNHLTeamRoster"
        params = {"teamAbv": team_abbrev}
    else:
        return []

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": host,
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            print(f"[injuries] Rate limited on {team_abbrev}, waiting 2s...")
            time.sleep(2)
            resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[injuries] ERROR fetching {team_abbrev}: {e}")
        return []

    injuries = []
    roster = data.get("body", {}).get("roster", {}) if isinstance(data.get("body"), dict) else {}

    # Tank01 returns roster as dict with player IDs as keys
    if isinstance(roster, dict):
        players = list(roster.values())
    elif isinstance(roster, list):
        players = roster
    else:
        return []

    for player in players:
        injury = player.get("injury", {})
        if not injury or not isinstance(injury, dict):
            continue

        status = injury.get("designation", "").upper()
        if status not in ("OUT", "GTD", "DOUBTFUL", "DAY-TO-DAY", "D2D"):
            continue

        injury_date = injury.get("injDate", "")
        freshness, days_out = get_freshness_tier(injury_date)

        injuries.append({
            "player": player.get("longName", player.get("espnName", "Unknown")),
            "pos": player.get("pos", ""),
            "status": "OUT" if status in ("OUT", "DOUBTFUL") else "GTD",
            "injury": injury.get("description", ""),
            "freshness": freshness,
            "days_out": days_out,
            "ppg": _safe_float(player.get("stats", {}).get("pts") if isinstance(player.get("stats"), dict) else None),
            "rpg": _safe_float(player.get("stats", {}).get("reb") if isinstance(player.get("stats"), dict) else None),
            "apg": _safe_float(player.get("stats", {}).get("ast") if isinstance(player.get("stats"), dict) else None),
            "team_record_without": None,  # Calculated later with results data
        })

    return injuries


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return round(float(val), 1)
    except (ValueError, TypeError):
        return None


def collect(sports: list[str] | None = None, date: str | None = None) -> dict[str, int]:
    """Collect injuries for all teams on today's slate. Returns sport -> injury count."""
    if sports is None:
        sports = ["NBA", "NHL"]
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    results = {}

    for sport in sports:
        if sport == "NCAAB":
            print(f"[injuries] SKIP NCAAB — Tank01 doesn't cover college")
            continue

        sport_key = sport.lower()
        data_file = DATA_DIR / f"games_{sport_key}_{date}.json"
        if not data_file.exists():
            print(f"[injuries] No data file for {sport}: {data_file}")
            continue

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        abbrevs = TEAM_ABBREVS.get(sport, {})
        teams_needed = set()
        for game in data.get("games", []):
            teams_needed.add(game["away"])
            teams_needed.add(game["home"])

        # Fetch injuries per team (with rate limiting)
        team_injuries: dict[str, list] = {}
        total_injuries = 0
        for team_name in sorted(teams_needed):
            abbrev = abbrevs.get(team_name)
            if not abbrev:
                print(f"[injuries] No abbreviation for: {team_name}")
                continue

            injuries = fetch_injuries_tank01(sport, abbrev)
            team_injuries[team_name] = injuries
            total_injuries += len(injuries)
            if injuries:
                print(f"[injuries] {team_name}: {len(injuries)} injuries")
            time.sleep(0.5)  # Rate limit: ~2 req/sec

        # Merge into game data
        for game in data.get("games", []):
            game["injuries"] = {
                "away": team_injuries.get(game["away"], []),
                "home": team_injuries.get(game["home"], []),
            }

        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"[injuries] {sport}: {total_injuries} injuries across {len(teams_needed)} teams -> {data_file.name}")
        results[sport] = total_injuries

    return results


if __name__ == "__main__":
    sports_arg = sys.argv[1:] if len(sys.argv) > 1 else None
    collect(sports_arg)

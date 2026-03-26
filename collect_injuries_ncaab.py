"""
Layer 1: NCAAB Injury Collector
Uses ESPN injury API for college basketball (Tank01 doesn't cover college).
Merges into existing data/games_ncaab_{date}.json
"""

import json
import os
import sys
import time
from datetime import datetime

import requests
from paths import DATA_DIR


def fetch_espn_injuries_ncaab() -> dict:
    """Fetch NCAAB injuries from ESPN."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/injuries"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[injuries-ncaab] ESPN injuries error: {e}")
        # Try alternate endpoint
        try:
            url2 = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams"
            resp2 = requests.get(url2, params={"limit": 100}, timeout=15)
            resp2.raise_for_status()
            return resp2.json()
        except:
            pass
    return {}


def fetch_team_injuries_espn(team_id: int) -> list[dict]:
    """Fetch injuries for a specific NCAAB team from ESPN."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{team_id}/injuries"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()

        injuries = []
        for item in data.get("items", []):
            athlete = item.get("athlete", {})
            status = item.get("status", "")
            injury_type = item.get("type", {}).get("description", "")
            details = item.get("details", {})

            if status.upper() not in ("OUT", "DOUBTFUL", "DAY-TO-DAY", "QUESTIONABLE"):
                continue

            injuries.append({
                "player": athlete.get("displayName", "Unknown"),
                "pos": athlete.get("position", {}).get("abbreviation", ""),
                "status": "OUT" if status.upper() in ("OUT", "DOUBTFUL") else "GTD",
                "injury": injury_type or details.get("type", ""),
                "freshness": "UNKNOWN",
                "days_out": 0,
                "ppg": None,
                "rpg": None,
                "apg": None,
                "team_record_without": None,
            })

        return injuries
    except Exception as e:
        return []


# ESPN team name -> ID mapping for common NCAAB teams
NCAAB_ESPN_IDS = {
    "Duke Blue Devils": 150, "North Carolina Tar Heels": 153,
    "Kentucky Wildcats": 96, "Kansas Jayhawks": 2305,
    "Michigan State Spartans": 127, "Ohio State Buckeyes": 194,
    "Louisville Cardinals": 97, "Wisconsin Badgers": 275,
    "Arkansas Razorbacks": 8, "Vanderbilt Commodores": 238,
    "Wake Forest Demon Deacons": 154, "TCU Horned Frogs": 2628,
    "Nebraska Cornhuskers": 158, "Dayton Flyers": 2168,
    "Southern Methodist Mustangs": 2567, "Nevada Wolf Pack": 2440,
    "California Golden Bears": 25, "Colorado State Rams": 36,
    "New Mexico Lobos": 167, "Illinois State Redbirds": 2287,
    "Bradley Braves": 2803, "Lehigh Mountain Hawks": 2329,
}


def collect(date: str | None = None) -> int:
    """Collect NCAAB injuries and merge into game data."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    data_file = DATA_DIR / f"games_ncaab_{date}.json"
    if not data_file.exists():
        print(f"[injuries-ncaab] No data file: {data_file}")
        return 0

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    teams_checked = set()

    for game in data.get("games", []):
        for side in ["away", "home"]:
            team_name = game.get(side, "")
            if team_name in teams_checked:
                # Reuse cached injuries
                continue

            team_id = NCAAB_ESPN_IDS.get(team_name)
            if team_id:
                injuries = fetch_team_injuries_espn(team_id)
                if injuries:
                    print(f"[injuries-ncaab] {team_name}: {len(injuries)} injuries")
                    total += len(injuries)
                time.sleep(0.3)
            else:
                injuries = []

            # Merge
            if "injuries" not in game:
                game["injuries"] = {"away": [], "home": []}
            game["injuries"][side] = injuries
            teams_checked.add(team_name)

    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[injuries-ncaab] {total} injuries across {len(teams_checked)} teams -> {data_file.name}")
    return total


if __name__ == "__main__":
    collect()

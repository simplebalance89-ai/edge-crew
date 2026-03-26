"""
Layer 1: Results Collector
Fetches game results from ESPN for ATS calculation.
Caches results in data/results_{sport}.json (append-only, growing cache).
ATS = did the team cover the spread? Requires historical spread + final score.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


def fetch_espn_scoreboard(sport: str, date: str) -> list[dict]:
    """Fetch completed games from ESPN scoreboard for a date (YYYYMMDD)."""
    sport_type = "basketball" if sport in ("NBA", "NCAAB") else "hockey"
    league = sport.lower()
    if sport == "NCAAB":
        league = "mens-college-basketball"

    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport_type}/{league}/scoreboard"
    params = {"dates": date}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[results] ERROR fetching {sport} scoreboard for {date}: {e}")
        return []

    results = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})

        results.append({
            "date": date,
            "sport": sport,
            "home_team": home.get("team", {}).get("displayName", ""),
            "away_team": away.get("team", {}).get("displayName", ""),
            "home_score": int(home.get("score", 0)),
            "away_score": int(away.get("score", 0)),
            "home_id": home.get("team", {}).get("id"),
            "away_id": away.get("team", {}).get("id"),
        })

    return results


def load_results_cache(sport: str) -> list[dict]:
    """Load existing results cache."""
    cache_file = DATA_DIR / f"results_{sport.lower()}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results_cache(sport: str, results: list[dict]):
    """Save results cache (deduplicated by date + teams)."""
    cache_file = DATA_DIR / f"results_{sport.lower()}.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        key = f"{r['date']}_{r['home_team']}_{r['away_team']}"
        if key not in seen:
            seen.add(key)
            unique.append(r)

    unique.sort(key=lambda r: r["date"], reverse=True)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2)

    return len(unique)


def calculate_ats(team_name: str, results: list[dict], spreads_cache: dict, n: int | None = None) -> dict:
    """
    Calculate ATS record for a team.
    spreads_cache: dict of "date_home_away" -> spread (from perspective of away team).
    """
    team_games = [
        r for r in results
        if r["home_team"] == team_name or r["away_team"] == team_name
    ]
    if n:
        team_games = team_games[:n]

    covers = 0
    pushes = 0
    losses = 0

    for game in team_games:
        key = f"{game['date']}_{game['home_team']}_{game['away_team']}"
        spread = spreads_cache.get(key)
        if spread is None:
            continue  # No spread data for this game

        is_home = game["home_team"] == team_name
        margin = game["home_score"] - game["away_score"]
        if not is_home:
            margin = -margin
            spread = -spread  # Flip spread for away team perspective

        # Cover = team beats the spread
        result = margin + spread
        if result > 0:
            covers += 1
        elif result == 0:
            pushes += 1
        else:
            losses += 1

    total = covers + losses + pushes
    return {
        "record": f"{covers}-{losses}" + (f"-{pushes}" if pushes else ""),
        "covers": covers,
        "losses": losses,
        "pushes": pushes,
        "total": total,
        "pct": round(covers / max(total, 1) * 100, 1),
    }


def collect(sports: list[str] | None = None, days_back: int = 30) -> dict[str, int]:
    """Collect recent game results for ATS calculation. Returns sport -> games added."""
    if sports is None:
        sports = ["NBA", "NHL"]

    results_map = {}

    for sport in sports:
        existing = load_results_cache(sport)
        existing_dates = {r["date"] for r in existing}

        new_results = []
        for i in range(days_back):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
            if date in existing_dates:
                continue

            games = fetch_espn_scoreboard(sport, date)
            if games:
                new_results.extend(games)
                print(f"[results] {sport} {date}: {len(games)} games")
            time.sleep(0.3)

        all_results = existing + new_results
        total = save_results_cache(sport, all_results)
        print(f"[results] {sport}: {len(new_results)} new, {total} total cached")
        results_map[sport] = len(new_results)

    return results_map


if __name__ == "__main__":
    sports_arg = sys.argv[1:] if len(sys.argv) > 1 else None
    collect(sports_arg)

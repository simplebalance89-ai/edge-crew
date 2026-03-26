"""
Layer 1: Odds Collector
Reads from Edge Crew cache (tmp_ec_*.json) as primary source.
Falls back to SportsGameOdds API if cache is stale.
Writes to data/games_{sport}_{date}.json
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = Path(os.environ.get("EC_CACHE_DIR", r"C:\Users\GCTII"))

SPORT_MAP = {
    "NBA": "nba",
    "NHL": "nhl",
    "NCAAB": "ncaab",
}

SGO_KEY = os.environ.get("SPORTSGAMEODDS_KEY")


def load_cache(sport: str) -> dict | None:
    """Load Edge Crew cache file for a sport."""
    cache_file = CACHE_DIR / f"tmp_ec_{SPORT_MAP[sport]}.json"
    if not cache_file.exists():
        print(f"[odds] No cache file for {sport}: {cache_file}")
        return None

    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check freshness — cache should be from today
    fetched_at = data.get("fetched_at", "")
    today_str = datetime.now().strftime("%b %d, %Y")
    if today_str not in fetched_at:
        print(f"[odds] WARNING: Cache for {sport} may be stale: {fetched_at}")

    return data


def transform_game(game: dict, sport: str) -> dict:
    """Transform Edge Crew cache format to consensus engine schema."""
    shifts = game.get("shifts", {})
    spread_open = shifts.get("spread", {}).get("open")
    total_open = shifts.get("total", {}).get("open")
    away_ml_open = shifts.get("away_ml", {}).get("open")
    home_ml_open = shifts.get("home_ml", {}).get("open")
    away_spread = game.get("away_spread")
    home_spread = game.get("home_spread")
    total_current = game.get("total")
    away_ml_current = game.get("away_ml")
    home_ml_current = game.get("home_ml")

    return {
        "game_id": game.get("id", ""),
        "sport": sport,
        "away": game.get("away", ""),
        "home": game.get("home", ""),
        "time": game.get("time", ""),
        "odds": {
            "spread_away": away_spread,
            "spread_home": home_spread,
            "spread_away_val": away_spread,
            "spread_home_val": home_spread,
            "spread_open": spread_open,
            "spread_current": away_spread,
            "spread_away_odds": game.get("away_spread_odds"),
            "spread_home_odds": game.get("home_spread_odds"),
            "total": total_current,
            "total_open": total_open,
            "total_current": total_current,
            "over_odds": game.get("over_odds"),
            "under_odds": game.get("under_odds"),
            "ml_away": away_ml_current,
            "ml_home": home_ml_current,
            "away_ml_open": away_ml_open,
            "away_ml_current": away_ml_current,
            "home_ml_open": home_ml_open,
            "home_ml_current": home_ml_current,
            "fair_away_ml": game.get("fair_away_ml"),
            "fair_home_ml": game.get("fair_home_ml"),
            "bookmaker": game.get("bookmaker", ""),
        },
        "shifts": {
            "spread_delta": round(
                (away_spread or 0) - (spread_open or 0), 1
            ),
            "total_delta": round(
                (total_current or 0) - (total_open or 0), 1
            ),
            "ml_moved": (away_ml_current or 0) != (away_ml_open or 0) or (home_ml_current or 0) != (home_ml_open or 0),
        },
        # Placeholders for other collectors to fill
        "away_profile": {},
        "home_profile": {},
        "injuries": {"away": [], "home": []},
        "lineup": {"away": [], "home": []},
        "props": [],
    }


def collect(sports: list[str] | None = None, date: str | None = None) -> dict[str, str]:
    """Collect odds for specified sports. Returns dict of sport -> output file path."""
    if sports is None:
        sports = ["NBA", "NHL", "NCAAB"]
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for sport in sports:
        cache = load_cache(sport)
        if cache is None:
            print(f"[odds] SKIP {sport} — no data")
            continue

        games = []
        for game in cache.get("games", []):
            games.append(transform_game(game, sport))

        output = {
            "sport": sport,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "games": games,
            "source": cache.get("source", "Edge Crew cache"),
            "fetched_at": cache.get("fetched_at", ""),
            "count": len(games),
        }

        out_file = DATA_DIR / f"games_{SPORT_MAP[sport]}_{date}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"[odds] {sport}: {len(games)} games -> {out_file.name}")
        results[sport] = str(out_file)

    return results


if __name__ == "__main__":
    sports_arg = sys.argv[1:] if len(sys.argv) > 1 else None
    collect(sports_arg)

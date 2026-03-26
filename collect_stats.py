"""
Layer 1: Stats Collector
Pulls team stats from ESPN public endpoints (no API key needed).
Calculates L5/L10 records, PPG, OPP PPG, streaks.
Merges into existing data/games_{sport}_{date}.json
"""

import json
import os
import sys
import time
from datetime import datetime

import requests
from paths import DATA_DIR

# ESPN team ID mappings
NBA_ESPN_IDS = {
    "Atlanta Hawks": 1, "Boston Celtics": 2, "Brooklyn Nets": 17,
    "Charlotte Hornets": 30, "Chicago Bulls": 4, "Cleveland Cavaliers": 5,
    "Dallas Mavericks": 6, "Denver Nuggets": 7, "Detroit Pistons": 8,
    "Golden State Warriors": 9, "Houston Rockets": 10, "Indiana Pacers": 11,
    "Los Angeles Clippers": 12, "Los Angeles Lakers": 13, "Memphis Grizzlies": 29,
    "Miami Heat": 14, "Milwaukee Bucks": 15, "Minnesota Timberwolves": 16,
    "New Orleans Pelicans": 3, "New York Knicks": 18, "Oklahoma City Thunder": 25,
    "Orlando Magic": 19, "Philadelphia 76ers": 20, "Phoenix Suns": 21,
    "Portland Trail Blazers": 22, "Sacramento Kings": 23, "San Antonio Spurs": 24,
    "Toronto Raptors": 28, "Utah Jazz": 26, "Washington Wizards": 27,
}

NHL_ESPN_IDS = {
    "Anaheim Ducks": 25, "Boston Bruins": 6, "Buffalo Sabres": 7,
    "Calgary Flames": 20, "Carolina Hurricanes": 12, "Chicago Blackhawks": 16,
    "Colorado Avalanche": 21, "Columbus Blue Jackets": 29, "Dallas Stars": 25,
    "Detroit Red Wings": 17, "Edmonton Oilers": 22, "Florida Panthers": 13,
    "Los Angeles Kings": 26, "Minnesota Wild": 30, "Montreal Canadiens": 8,
    "Nashville Predators": 18, "New Jersey Devils": 1, "New York Islanders": 2,
    "New York Rangers": 3, "Ottawa Senators": 9, "Philadelphia Flyers": 4,
    "Pittsburgh Penguins": 5, "San Jose Sharks": 28, "Seattle Kraken": 55,
    "St. Louis Blues": 19, "Tampa Bay Lightning": 14, "Toronto Maple Leafs": 10,
    "Vancouver Canucks": 23, "Vegas Golden Knights": 54, "Washington Capitals": 15,
    "Winnipeg Jets": 52,
}

SPORT_ESPN = {
    "NBA": {"league": "nba", "ids": NBA_ESPN_IDS},
    "NHL": {"league": "nhl", "ids": NHL_ESPN_IDS},
}


def fetch_team_schedule(sport: str, team_id: int, season: int) -> list[dict]:
    """Fetch team's game results from ESPN."""
    league = SPORT_ESPN[sport]["league"]
    url = f"https://site.api.espn.com/apis/site/v2/sports/{_sport_type(sport)}/{league}/teams/{team_id}/schedule"
    params = {"season": season}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[stats] ERROR fetching schedule for team {team_id}: {e}")
        return []

    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home_team = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away_team = next((c for c in competitors if c.get("homeAway") == "away"), {})

        is_home = str(home_team.get("id")) == str(team_id)
        our_score = _parse_score(home_team.get("score", 0)) if is_home else _parse_score(away_team.get("score", 0))
        opp_score = _parse_score(away_team.get("score", 0)) if is_home else _parse_score(home_team.get("score", 0))

        opp_data = away_team if is_home else home_team
        games.append({
            "date": event.get("date", ""),
            "is_home": is_home,
            "our_score": our_score,
            "opp_score": opp_score,
            "won": our_score > opp_score,
            "opp_name": opp_data.get("team", {}).get("displayName", ""),
            "opp_id": str(opp_data.get("team", {}).get("id", "")),
        })

    # Sort by date descending (most recent first)
    games.sort(key=lambda g: g["date"], reverse=True)
    return games


def _parse_score(val) -> int:
    """Parse ESPN score which can be int, str, or dict with 'value' key."""
    if isinstance(val, dict):
        return int(val.get("value", val.get("displayValue", 0)))
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _sport_type(sport: str) -> str:
    return "basketball" if sport in ("NBA", "NCAAB") else "icehockey"


def calculate_profile(games: list[dict], as_of_date: str | None = None, n_last: int = 10) -> dict:
    """Calculate team profile from game results with enriched metrics."""
    if not games:
        return {}

    if as_of_date:
        cutoff_date = datetime.strptime(as_of_date, "%Y%m%d")
    else:
        cutoff_date = datetime.now()

    total_w = sum(1 for g in games if g["won"])
    total_l = len(games) - total_w
    home_games = [g for g in games if g["is_home"]]
    away_games = [g for g in games if not g["is_home"]]
    home_w = sum(1 for g in home_games if g["won"])
    away_w = sum(1 for g in away_games if g["won"])

    last5 = games[:5]
    last10 = games[:10]
    l5_w = sum(1 for g in last5 if g["won"])
    l10_w = sum(1 for g in last10 if g["won"])

    # PPG calculations
    ppg_l5 = round(sum(g["our_score"] for g in last5) / max(len(last5), 1), 1) if last5 else 0
    opp_ppg_l5 = round(sum(g["opp_score"] for g in last5) / max(len(last5), 1), 1) if last5 else 0
    ppg_l10 = round(sum(g["our_score"] for g in last10) / max(len(last10), 1), 1) if last10 else 0
    opp_ppg_l10 = round(sum(g["opp_score"] for g in last10) / max(len(last10), 1), 1) if last10 else 0
    ppg_season = round(sum(g["our_score"] for g in games) / max(len(games), 1), 1)
    opp_ppg_season = round(sum(g["opp_score"] for g in games) / max(len(games), 1), 1)

    # Streak
    streak = 0
    streak_type = ""
    if games:
        streak_type = "W" if games[0]["won"] else "L"
        for g in games:
            if (streak_type == "W" and g["won"]) or (streak_type == "L" and not g["won"]):
                streak += 1
            else:
                break

    # ─── REST / B2B / SCHEDULE ───────────────────────────────────────────
    rest_days = None
    is_b2b = False
    if len(games) >= 1 and games[0].get("date"):
        try:
            last_game_date = datetime.fromisoformat(games[0]["date"].replace("Z", "+00:00"))
            last_game_naive = last_game_date.replace(tzinfo=None)
            rest_days = (cutoff_date - last_game_naive).days
            is_b2b = rest_days <= 1
        except (ValueError, TypeError):
            pass

    # Games in last 7 days (fatigue indicator)
    games_last_7d = 0
    try:
        for g in games:
            if g.get("date"):
                gd = datetime.fromisoformat(g["date"].replace("Z", "+00:00")).replace(tzinfo=None)
                if 0 <= (cutoff_date - gd).days <= 7:
                    games_last_7d += 1
                else:
                    break
    except (ValueError, TypeError):
        pass

    # ─── ROAD TRIP / HOME STAND ──────────────────────────────────────────
    road_trip_len = 0
    home_stand_len = 0
    if games:
        # Count consecutive away or home games from most recent
        first_loc = games[0].get("is_home")
        for g in games:
            if g.get("is_home") == first_loc:
                if first_loc:
                    home_stand_len += 1
                else:
                    road_trip_len += 1
            else:
                break

    # ─── NET RATING / MARGIN ─────────────────────────────────────────────
    margin_l5 = round(ppg_l5 - opp_ppg_l5, 1)
    margin_l10 = round(ppg_l10 - opp_ppg_l10, 1)
    margin_season = round(ppg_season - opp_ppg_season, 1)

    # ─── PACE (total points proxy) ───────────────────────────────────────
    pace_l5 = round(ppg_l5 + opp_ppg_l5, 1)  # Combined PPG = tempo proxy
    pace_l10 = round(ppg_l10 + opp_ppg_l10, 1)

    # ─── BLOWOUT / CLOSE GAME TENDENCIES ─────────────────────────────────
    close_games_l10 = sum(1 for g in last10 if abs(g["our_score"] - g["opp_score"]) <= 5)
    blowouts_l10 = sum(1 for g in last10 if abs(g["our_score"] - g["opp_score"]) >= 15)

    # ─── COVER MARGIN PROXY (avg margin = spread cover tendency) ─────────
    # Positive margin on favorites = likely covering
    avg_margin_l10 = round(
        sum(g["our_score"] - g["opp_score"] for g in last10) / max(len(last10), 1), 1
    ) if last10 else 0

    return {
        "record": f"{total_w}-{total_l}",
        "home_record": f"{home_w}-{len(home_games) - home_w}",
        "away_record": f"{away_w}-{len(away_games) - away_w}",
        "L5": f"{l5_w}-{len(last5) - l5_w}",
        "L10": f"{l10_w}-{len(last10) - l10_w}",
        "ppg_L5": ppg_l5,
        "opp_ppg_L5": opp_ppg_l5,
        "ppg_L10": ppg_l10,
        "opp_ppg_L10": opp_ppg_l10,
        "ppg_season": ppg_season,
        "opp_ppg_season": opp_ppg_season,
        "streak": f"{streak_type}{streak}",
        "games_played": len(games),
        # Schedule / rest
        "rest_days": rest_days,
        "is_b2b": is_b2b,
        "games_last_7d": games_last_7d,
        "road_trip_len": road_trip_len,
        "home_stand_len": home_stand_len,
        # Net rating / margin
        "margin_L5": margin_l5,
        "margin_L10": margin_l10,
        "margin_season": margin_season,
        "avg_margin_L10": avg_margin_l10,
        # Pace
        "pace_L5": pace_l5,
        "pace_L10": pace_l10,
        # Game type tendencies
        "close_games_L10": close_games_l10,
        "blowouts_L10": blowouts_l10,
        # ATS will be filled by collect_results.py
        "ATS_L5": None,
        "ATS_L10": None,
        "ATS_season": None,
    }


def collect(sports: list[str] | None = None, date: str | None = None) -> dict[str, int]:
    """Collect stats for all teams on today's slate."""
    if sports is None:
        sports = ["NBA", "NHL"]
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    season = int(date[:4])

    results = {}

    for sport in sports:
        if sport not in SPORT_ESPN:
            print(f"[stats] SKIP {sport} — no ESPN mapping yet")
            continue

        sport_key = sport.lower()
        data_file = DATA_DIR / f"games_{sport_key}_{date}.json"
        if not data_file.exists():
            print(f"[stats] No data file for {sport}: {data_file}")
            continue

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        ids = SPORT_ESPN[sport]["ids"]
        teams_done = set()
        team_profiles: dict[str, dict] = {}
        team_raw_games: dict[str, list] = {}  # Store raw games for H2H
        teams_needed = set()

        for game in data.get("games", []):
            teams_needed.add(game["away"])
            teams_needed.add(game["home"])

        for team_name in sorted(teams_needed):
            if team_name in teams_done:
                continue

            team_id = ids.get(team_name)
            if not team_id:
                print(f"[stats] No ESPN ID for: {team_name}")
                team_profiles[team_name] = {}
                team_raw_games[team_name] = []
                continue

            games = fetch_team_schedule(sport, team_id, season=season)
            profile = calculate_profile(games, as_of_date=date)
            team_profiles[team_name] = profile
            team_raw_games[team_name] = games
            teams_done.add(team_name)
            print(f"[stats] {team_name}: {profile.get('record', '?')} | L5 {profile.get('L5', '?')} | {profile.get('streak', '?')}")
            time.sleep(0.3)  # Be nice to ESPN

        # Calculate H2H and merge into game data
        for game in data.get("games", []):
            away_name = game["away"]
            home_name = game["home"]
            away_prof = team_profiles.get(away_name, {})
            home_prof = team_profiles.get(home_name, {})

            # H2H this season
            away_games = team_raw_games.get(away_name, [])
            h2h_games = [g for g in away_games if g.get("opp_name") == home_name]
            h2h_w = sum(1 for g in h2h_games if g["won"])
            h2h_l = len(h2h_games) - h2h_w
            h2h_record = f"{h2h_w}-{h2h_l}" if h2h_games else "0-0"

            away_prof["h2h_season"] = h2h_record
            home_prof["h2h_season"] = f"{h2h_l}-{h2h_w}" if h2h_games else "0-0"

            # Rest advantage (positive = our team more rested)
            away_rest = away_prof.get("rest_days")
            home_rest = home_prof.get("rest_days")
            if away_rest is not None and home_rest is not None:
                away_prof["rest_advantage"] = away_rest - home_rest
                home_prof["rest_advantage"] = home_rest - away_rest
            else:
                away_prof["rest_advantage"] = 0
                home_prof["rest_advantage"] = 0

            game["away_profile"] = away_prof
            game["home_profile"] = home_prof

        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"[stats] {sport}: {len(teams_done)} teams profiled -> {data_file.name}")
        results[sport] = len(teams_done)

    return results


if __name__ == "__main__":
    sports_arg = sys.argv[1:] if len(sys.argv) > 1 else None
    collect(sports_arg)

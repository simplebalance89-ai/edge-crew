"""
Pipeline Orchestrator — Edge Crew V3
Runs the full analysis sequence for any date/sport.

LAYERS:
  0. Fetch schedule (ESPN scoreboard)
  1. Collect odds (SportsGameOdds API)
  1.5. Collect stats (ESPN team stats)
  1.6. Collect injuries (Tank01)
  1.7. Collect props (BDL)
  2. Roster profiles (4 Azure models, all teams)
  2.5. Player profiles + chain scoring
  3. Grade engine (15+2 Sinton.ia vars)
  3.5. Swim lane H2H race
  4. Output final slate

Usage:
  python run_pipeline.py                          # Today, all sports
  python run_pipeline.py --date 20260319          # Specific date
  python run_pipeline.py --sport NBA              # One sport
  python run_pipeline.py --date 20260319 --sport NBA
  python run_pipeline.py --skip-profiles          # Skip AI model calls (fast regrade)
"""

import json
import os
import sys
import time
import argparse
import subprocess
import concurrent.futures
from datetime import datetime, timedelta

import requests
from app_config import remove_dead_local_proxy_env, require_env
from paths import BASE_DIR, DATA_DIR, GRADES_DIR

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DATA_DIR.mkdir(parents=True, exist_ok=True)
GRADES_DIR.mkdir(parents=True, exist_ok=True)

REMOVED_PROXY_VARS = remove_dead_local_proxy_env()
if REMOVED_PROXY_VARS:
    print(f"[network] Removed dead proxy vars: {', '.join(REMOVED_PROXY_VARS)}")

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
SGO_KEY = os.environ.get("SPORTSGAMEODDS_KEY")
BDL_KEY = os.environ.get("BALLDONTLIE_API_KEY")

SPORT_ESPN = {
    "NBA":   ("basketball", "nba"),
    "NHL":   ("icehockey", "nhl"),
    "NCAAB": ("basketball", "mens-college-basketball"),
}

SPORT_SGO = {
    "NBA":   "NBA",
    "NHL":   "NHL",
    "NCAAB": "NCAAB",
}


# ─── Layer 0: Fetch Schedule ─────────────────────────────────────────────────────

def fetch_schedule(sport: str, date: str) -> list[dict]:
    """Pull tomorrow's games from ESPN scoreboard API."""
    sport_type, league = SPORT_ESPN[sport]
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport_type}/{league}/scoreboard"
    try:
        r = requests.get(url, params={"dates": date}, timeout=15)
        r.raise_for_status()
        events = r.json().get("events", [])
    except Exception as e:
        print(f"  [schedule] ESPN error for {sport}: {e}")
        return []

    games = []
    for ev in events:
        comp = ev.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})
        game_time = comp.get("date", "")[:16]

        games.append({
            "game_id": ev.get("id", ""),
            "sport": sport,
            "home": home.get("team", {}).get("displayName", "?"),
            "home_abbrev": home.get("team", {}).get("abbreviation", "?"),
            "home_espn_id": home.get("team", {}).get("id", ""),
            "away": away.get("team", {}).get("displayName", "?"),
            "away_abbrev": away.get("team", {}).get("abbreviation", "?"),
            "away_espn_id": away.get("team", {}).get("id", ""),
            "time": game_time,
            "odds": {},
            "away_profile": {},
            "home_profile": {},
            "injuries": {"away": [], "home": []},
            "lineup": {"away": [], "home": []},
            "props": [],
        })

    return games


# ─── Layer 1: Collect Odds (SGO) ─────────────────────────────────────────────────

def fetch_odds_sgo(sport: str, date: str) -> dict[str, dict]:
    """Fetch odds from SportsGameOdds API. Returns game_id -> odds dict."""
    if not SGO_KEY:
        print("  [odds] SPORTSGAMEODDS_KEY missing - continuing without odds")
        return {}
    sgo_sport = SPORT_SGO.get(sport, sport)
    # Date format for SGO: YYYY-MM-DD
    date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    url = f"https://api.sportsgameodds.com/v2/events/"
    headers = {}
    params = {"apiKey": SGO_KEY, "leagueID": sgo_sport, "oddsAvailable": "true", "limit": 50}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        events = r.json().get("data", [])
    except Exception as e:
        print(f"  [odds] SGO error for {sport}: {e}")
        return {}

    odds_map = {}
    for ev in events:
        teams = ev.get("teams", {})
        home = teams.get("home", {}).get("names", {}).get("long", "")
        away = teams.get("away", {}).get("names", {}).get("long", "")
        odds = ev.get("odds", {})

        # Parse SGO v2 odds keys
        spread_home = odds.get("points-home-game-sp-home", {})
        spread_away = odds.get("points-away-game-sp-away", {})
        total_over = odds.get("points-all-game-ou-over", {})
        total_under = odds.get("points-all-game-ou-under", {})
        ml_home = odds.get("points-home-game-ml-home", {})
        ml_away = odds.get("points-away-game-ml-away", {})

        key = f"{away}@{home}"
        total_line = total_over.get("bookSpread")
        ml_home_current = ml_home.get("bookOdds")
        ml_away_current = ml_away.get("bookOdds")
        spread_home_current = spread_home.get("bookSpread")
        spread_away_current = spread_away.get("bookSpread")
        odds_map[key] = {
            "spread_home": spread_home_current,
            "spread_home_val": spread_home_current,
            "spread_home_odds": spread_home.get("bookOdds"),
            "spread_away": spread_away_current,
            "spread_away_val": spread_away_current,
            "spread_away_odds": spread_away.get("bookOdds"),
            "total": total_line,  # SGO uses spread field for O/U line
            "total_current": total_line,
            "over_odds": total_over.get("bookOdds"),
            "under_odds": total_under.get("bookOdds"),
            "ml_home": ml_home_current,
            "ml_away": ml_away_current,
            "home_ml_current": ml_home_current,
            "away_ml_current": ml_away_current,
            "provider": "SGO",
        }
        if home and away:
            print(f"    Odds: {away} @ {home} | Spread: {spread_home.get('bookSpread')} | O/U: {total_over.get('bookSpread')} | ML: {ml_away.get('bookOdds')}/{ml_home.get('bookOdds')}")

    return odds_map


# ─── Layer 1.5: Team Stats (ESPN) ────────────────────────────────────────────────

def fetch_team_stats_espn(sport: str, team_id: str, date: str) -> dict:
    """Fetch L5/L10 stats for a team from ESPN."""
    sport_type, league = SPORT_ESPN[sport]
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport_type}/{league}/teams/{team_id}/schedule"

    try:
        r = requests.get(url, params={"season": date[:4]}, timeout=15)
        r.raise_for_status()
        events = r.json().get("events", [])
    except Exception as e:
        return {}

    date_dt = datetime.strptime(date, "%Y%m%d")
    completed = []

    for ev in events:
        comp = ev.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue
        game_date_str = comp.get("date", "")[:10]
        try:
            gd = datetime.strptime(game_date_str, "%Y-%m-%d")
        except ValueError:
            continue
        if gd >= date_dt:
            continue

        competitors = comp.get("competitors", [])
        my_team = next((c for c in competitors if str(c.get("team", {}).get("id", "")) == str(team_id)), None)
        opp_team = next((c for c in competitors if str(c.get("team", {}).get("id", "")) != str(team_id)), None)

        if my_team and opp_team:
            my_score_raw = my_team.get("score", 0)
            opp_score_raw = opp_team.get("score", 0)
            # ESPN returns score as dict {"value": N} or direct int
            if isinstance(my_score_raw, dict):
                my_score_raw = my_score_raw.get("value", 0)
            if isinstance(opp_score_raw, dict):
                opp_score_raw = opp_score_raw.get("value", 0)
            my_score = int(my_score_raw or 0)
            opp_score = int(opp_score_raw or 0)
            won = my_score > opp_score
            completed.append({
                "date": game_date_str,
                "score": my_score,
                "opp_score": opp_score,
                "won": won,
                "home": my_team.get("homeAway") == "home",
                "margin": my_score - opp_score,
            })

    completed.sort(key=lambda x: x["date"], reverse=True)

    def stats_from_games(games):
        if not games:
            return {}
        scores = [g["score"] for g in games]
        opp_scores = [g["opp_score"] for g in games]
        margins = [g["margin"] for g in games]
        wins = sum(1 for g in games if g["won"])
        losses = len(games) - wins
        return {
            "record": f"{wins}-{losses}",
            "ppg": round(sum(scores) / len(scores), 1),
            "opp_ppg": round(sum(opp_scores) / len(opp_scores), 1),
            "margin": round(sum(margins) / len(margins), 1),
        }

    l5 = completed[:5]
    l10 = completed[:10]
    l5_stats = stats_from_games(l5)
    l10_stats = stats_from_games(l10)

    # Rest days
    if completed:
        last_date = datetime.strptime(completed[0]["date"], "%Y-%m-%d")
        rest_days = (date_dt - last_date).days
    else:
        rest_days = 2

    # Streak
    streak_count = 0
    streak_type = "W" if completed and completed[0]["won"] else "L"
    for g in completed:
        if (g["won"] and streak_type == "W") or (not g["won"] and streak_type == "L"):
            streak_count += 1
        else:
            break

    # Home/away records
    home_games = [g for g in completed if g["home"]]
    away_games = [g for g in completed if not g["home"]]
    home_w = sum(1 for g in home_games if g["won"])
    away_w = sum(1 for g in away_games if g["won"])

    return {
        "L5": l5_stats.get("record", "0-5"),
        "L10": l10_stats.get("record", "0-10"),
        "ppg_L5": l5_stats.get("ppg", 0),
        "opp_ppg_L5": l5_stats.get("opp_ppg", 0),
        "margin_L5": l5_stats.get("margin", 0),
        "avg_margin_L10": l10_stats.get("margin", 0),
        "ppg_L10": l10_stats.get("ppg", 0),
        "opp_ppg_L10": l10_stats.get("opp_ppg", 0),
        "rest_days": rest_days,
        "is_b2b": rest_days == 1,
        "streak": f"{streak_type}{streak_count}",
        "home_record": f"{home_w}-{len(home_games) - home_w}",
        "away_record": f"{away_w}-{len(away_games) - away_w}",
        "games_last_7d": sum(1 for g in completed if (date_dt - datetime.strptime(g["date"], "%Y-%m-%d")).days <= 7),
    }


# ─── Layer 1.6: Injuries (Tank01) ────────────────────────────────────────────────

NBA_TEAM_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GS", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NO", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SA", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

FRESHNESS_TIERS = [
    (3, "FRESH"), (14, "RECENT"), (30, "ESTABLISHED"), (9999, "SEASON"),
]


def get_freshness(injury_date_str: str | None, ref_date: str) -> tuple[str, int]:
    if not injury_date_str:
        return "UNKNOWN", -1
    try:
        ref = datetime.strptime(ref_date, "%Y%m%d")
        injury_date_str = str(injury_date_str)
        if "-" in injury_date_str:
            inj = datetime.strptime(injury_date_str[:10], "%Y-%m-%d")
        else:
            inj = datetime.strptime(injury_date_str[:8], "%Y%m%d")
        days_out = (ref - inj).days
        for threshold, tier in FRESHNESS_TIERS:
            if days_out <= threshold:
                return tier, days_out
        return "SEASON", days_out
    except Exception:
        return "UNKNOWN", -1


def fetch_injuries_tank01_nba(team_name: str, date: str) -> list[dict]:
    abbrev = NBA_TEAM_ABBREV.get(team_name, "")
    if not abbrev:
        return []

    host = "tank01-fantasy-stats.p.rapidapi.com"
    if not RAPIDAPI_KEY:
        print(f"    [injuries] RAPIDAPI_KEY missing - skipping injuries for {team_name}")
        return []
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": host,
    }
    url = f"https://{host}/getNBATeamRoster"

    try:
        r = requests.get(url, headers=headers,
                         params={"teamAbv": abbrev, "getStats": "true"},
                         timeout=10)
        r.raise_for_status()
        data = r.json()
        roster = data.get("body", {}).get("roster", [])
    except Exception as e:
        print(f"    [injuries] Tank01 error for {team_name}: {e}")
        return []

    injuries = []
    if isinstance(roster, dict):
        players = roster.values()
    elif isinstance(roster, list):
        players = roster
    else:
        players = []
    for player in players:
        inj_status = player.get("injury", {})
        if not isinstance(inj_status, dict):
            continue
        status = (inj_status.get("designation") or "").upper()
        if status not in ("OUT", "GTD", "DOUBTFUL", "DAY-TO-DAY", "D2D"):
            continue
        inj_date = inj_status.get("injDate")
        freshness, days_out = get_freshness(inj_date, date)

        stats = player.get("stats", {})
        ppg = None
        try:
            if isinstance(stats, dict):
                raw_pts = stats.get("pts", 0) or 0
            else:
                raw_pts = 0
            ppg = float(raw_pts)
            if ppg < 1:
                ppg = None
        except (ValueError, TypeError):
            ppg = None

        injuries.append({
            "player": player.get("longName", player.get("name", "")),
            "pos": player.get("pos", ""),
            "status": "OUT" if status in ("OUT", "DOUBTFUL") else "GTD",
            "injDate": inj_date,
            "freshness": freshness,
            "days_out": days_out,
            "ppg": ppg,
            "description": inj_status.get("description", ""),
        })

    return injuries


# ─── Layer 3.5: Swim Lane H2H Race ───────────────────────────────────────────────

GRADE_TO_SCORE = {
    "A+": 9.2, "A": 7.9, "A-": 7.2, "B+": 6.7, "B": 6.2,
    "B-": 5.7, "C+": 5.2, "C": 4.5, "D": 3.5, "F": 1.0,
}


def run_swim_lane(game: dict, team_grades: dict, player_chains: list[dict]) -> dict:
    """
    Head-to-head swim lane race.
    Each layer awards a point to the side that wins that heat.
    Final: which team leads, by how much, and what size bet.
    """
    home = game.get("home", "")
    away = game.get("away", "")

    home_profile = game.get("home_profile", {})
    away_profile = game.get("away_profile", {})

    lanes = []
    home_leads = 0
    away_leads = 0

    # ── Heat 1: Team Profile Grade ────────────────────────────────────────────────
    home_grade = team_grades.get(home, "C")
    away_grade = team_grades.get(away, "C")
    home_grade_score = GRADE_TO_SCORE.get(home_grade, 5.0)
    away_grade_score = GRADE_TO_SCORE.get(away_grade, 5.0)
    diff1 = home_grade_score - away_grade_score

    if diff1 >= 0.5:
        winner1 = "home"
        home_leads += 1
    elif diff1 <= -0.5:
        winner1 = "away"
        away_leads += 1
    else:
        winner1 = "push"

    lanes.append({
        "heat": "Team Profile",
        "home": {"grade": home_grade, "score": home_grade_score},
        "away": {"grade": away_grade, "score": away_grade_score},
        "winner": winner1,
        "margin": round(abs(diff1), 1),
    })

    # ── Heat 2: Player Chains ─────────────────────────────────────────────────────
    home_chains = [c for c in player_chains if c.get("team") == home and c.get("math_chain", {}).get("fired")]
    away_chains = [c for c in player_chains if c.get("team") == away and c.get("math_chain", {}).get("fired")]

    home_chain_score = len(home_chains) + sum(
        c.get("consensus", {}).get("score", 0) for c in home_chains
    ) / max(len(home_chains) * 10, 1)
    away_chain_score = len(away_chains) + sum(
        c.get("consensus", {}).get("score", 0) for c in away_chains
    ) / max(len(away_chains) * 10, 1)

    diff2 = home_chain_score - away_chain_score
    if diff2 >= 0.5:
        winner2 = "home"
        home_leads += 1
    elif diff2 <= -0.5:
        winner2 = "away"
        away_leads += 1
    else:
        winner2 = "push"

    lanes.append({
        "heat": "Player Chains",
        "home": {"chains_fired": len(home_chains), "score": round(home_chain_score, 2)},
        "away": {"chains_fired": len(away_chains), "score": round(away_chain_score, 2)},
        "winner": winner2,
        "margin": round(abs(diff2), 2),
    })

    # ── Heat 3: Matchup (H2H + Pace + Defense) ───────────────────────────────────
    # Use existing L5 stats to score the matchup
    home_off = home_profile.get("ppg_L5", 110)
    away_def = away_profile.get("opp_ppg_L5", 112)  # How much away allows
    away_off = away_profile.get("ppg_L5", 110)
    home_def = home_profile.get("opp_ppg_L5", 112)

    # If home offense > what away allows: home wins matchup
    home_matchup = (home_off or 110) - (away_def or 112)
    away_matchup = (away_off or 110) - (home_def or 112)
    diff3 = home_matchup - away_matchup

    if diff3 >= 2:
        winner3 = "home"
        home_leads += 1
    elif diff3 <= -2:
        winner3 = "away"
        away_leads += 1
    else:
        winner3 = "push"

    lanes.append({
        "heat": "Matchup (Off vs Def)",
        "home": {
            "off_l5": home_off, "vs_opp_def": away_def,
            "edge": round(home_matchup, 1),
        },
        "away": {
            "off_l5": away_off, "vs_opp_def": home_def,
            "edge": round(away_matchup, 1),
        },
        "winner": winner3,
        "margin": round(abs(diff3), 1),
    })

    # ── Race Result ───────────────────────────────────────────────────────────────
    margin = home_leads - away_leads

    if margin >= 3:
        race_winner = "home"
        bet_size = "2u"
        confidence = "Very High"
    elif margin == 2:
        race_winner = "home"
        bet_size = "1u"
        confidence = "High"
    elif margin == 1:
        race_winner = "home"
        bet_size = "0.5u"
        confidence = "Moderate"
    elif margin == -1:
        race_winner = "away"
        bet_size = "0.5u"
        confidence = "Moderate"
    elif margin == -2:
        race_winner = "away"
        bet_size = "1u"
        confidence = "High"
    elif margin <= -3:
        race_winner = "away"
        bet_size = "2u"
        confidence = "Very High"
    else:
        race_winner = "push"
        bet_size = "PASS"
        confidence = "Low"

    odds = game.get("odds", {})
    if race_winner == "home":
        pick_team = home
        pick_spread = odds.get("spread_home_val", odds.get("spread_home", "?"))
    elif race_winner == "away":
        pick_team = away
        pick_spread = odds.get("spread_away_val", odds.get("spread_away", "?"))
    else:
        pick_team = "PASS"
        pick_spread = None

    return {
        "matchup": f"{away} @ {home}",
        "home": home,
        "away": away,
        "lanes": lanes,
        "home_score": home_leads,
        "away_score": away_leads,
        "margin": abs(margin),
        "race_winner": race_winner,
        "pick_team": pick_team,
        "pick_spread": pick_spread,
        "bet_size": bet_size,
        "confidence": confidence,
    }


# ─── Pipeline Print / Save ───────────────────────────────────────────────────────

def _lane_detail(lane: dict, side: str) -> str:
    """Format one side of a lane as clean readable string."""
    d = lane.get(side, {})
    heat = lane["heat"]
    if heat == "Team Profile":
        return f"{d.get('grade','?')} ({d.get('score','?')})"
    elif heat == "Player Chains":
        fired = d.get("chains_fired", 0)
        score = d.get("score", 0)
        return f"{fired} chains ({score:.1f})"
    elif heat == "Matchup (Off vs Def)":
        off = d.get("off_l5", 0)
        edge = d.get("edge", 0)
        return f"Off {off} | edge {edge:+.1f}"
    return str(d)[:20]


def print_race(race: dict):
    home = race["home"]
    away = race["away"]
    sep = "-" * 58
    print(f"\n  {sep}")
    print(f"  {away:<25}  @  {home}")
    print(f"  {sep}")
    print(f"  {'HEAT':<22} {'LEADER':<8} {'HOME':<22} {'AWAY'}")
    print(f"  {'-'*22} {'-'*8} {'-'*22} {'-'*22}")
    for lane in race["lanes"]:
        w = lane["winner"]
        arrow = "HOME >" if w == "home" else "AWAY <" if w == "away" else "  TIE "
        home_str = _lane_detail(lane, "home")
        away_str = _lane_detail(lane, "away")
        print(f"  {lane['heat']:<22} {arrow:<8} {home_str:<22} {away_str}")
    print(f"  {sep}")
    score_line = f"  RACE  {home} {race['home_score']} - {race['away_score']} {away}"
    winner = race["race_winner"]
    if winner == "push":
        pick_line = f"  PICK  PASS"
    else:
        spread = race.get("pick_spread") or "TBD"
        pick_line = f"  PICK  {race['pick_team']}  {spread}  |  {race['bet_size']}  |  {race['confidence']}"
    print(score_line)
    print(pick_line)


# ─── Main Pipeline ───────────────────────────────────────────────────────────────

def run_pipeline(
    sports: list[str] | None = None,
    date: str | None = None,
    skip_profiles: bool = False,
    skip_players: bool = False,
):
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    if sports is None:
        sports = ["NBA", "NHL", "NCAAB"]

    date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    print(f"\n{'='*60}")
    print(f"  EDGE CREW V3 - FULL PIPELINE")
    print(f"  Date: {date_fmt} | Sports: {', '.join(sports)}")
    print(f"{'='*60}\n")

    all_races = {}

    for sport in sports:
        print(f"\n{'='*60}")
        print(f"  [{sport}] Starting pipeline...")
        print(f"{'='*60}")

        # ── Layer 0: Schedule ─────────────────────────────────────────────────────
        print(f"  Layer 0: Fetching schedule...")
        games = fetch_schedule(sport, date)
        if not games:
            # Fallback: load cached game data if ESPN fails
            cached_file = DATA_DIR / f"games_{sport.lower()}_{date}.json"
            if cached_file.exists():
                with open(cached_file, "r", encoding="utf-8") as _cf:
                    cached = json.load(_cf)
                games = cached.get("games", [])
                if games:
                    print(f"  ESPN failed - loaded {len(games)} cached games from {cached_file.name}")
            if not games:
                print(f"  No games found for {sport} on {date_fmt}")
                continue
        print(f"  Found {len(games)} games")

        # ── Layer 1: Odds ─────────────────────────────────────────────────────────
        print(f"  Layer 1: Fetching odds...")
        odds_map = fetch_odds_sgo(sport, date)
        for game in games:
            key = f"{game['away']}@{game['home']}"
            if key in odds_map:
                game["odds"] = odds_map[key]
                print(f"    Odds loaded: {game['away']} @ {game['home']}")

        # ── Layer 1.5: Team Stats (parallel) ─────────────────────────────────────
        print(f"  Layer 1.5: Fetching team stats (parallel)...")
        teams_to_fetch = []
        for game in games:
            teams_to_fetch.append((game["home"], game.get("home_espn_id", ""), "home", game))
            teams_to_fetch.append((game["away"], game.get("away_espn_id", ""), "away", game))

        def fetch_stats_for_team(args):
            team_name, espn_id, side, game = args
            if not espn_id:
                return
            stats = fetch_team_stats_espn(sport, espn_id, date)
            if stats:
                game[f"{side}_profile"] = stats

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(fetch_stats_for_team, teams_to_fetch))

        print(f"  Team stats loaded for {len(teams_to_fetch)} teams")

        # ── Layer 1.6: Injuries ───────────────────────────────────────────────────
        if sport == "NBA":
            print(f"  Layer 1.6: Fetching injuries (Tank01)...")
            for game in games:
                for side in ["home", "away"]:
                    team = game[side]
                    injuries = fetch_injuries_tank01_nba(team, date)
                    game["injuries"][side] = injuries
                    if injuries:
                        out_count = sum(1 for i in injuries if i.get("status") == "OUT")
                        print(f"    {team}: {len(injuries)} injuries ({out_count} OUT)")
                    time.sleep(0.2)

        elif sport == "NCAAB":
            print(f"  Layer 1.6: Fetching injuries (ESPN NCAAB)...")
            try:
                from collect_injuries_ncaab import collect as collect_injuries_ncaab
                collect_injuries_ncaab(date)
                # Reload games after injuries are merged
                data_file_tmp = DATA_DIR / f"games_ncaab_{date}.json"
                if data_file_tmp.exists():
                    with open(data_file_tmp, "r", encoding="utf-8") as _f:
                        _reloaded = json.load(_f)
                    games = _reloaded.get("games", games)
            except Exception as e:
                print(f"  [injuries-ncaab] Error: {e}")

        # NHL: no injury collector yet — skip silently
        elif sport == "NHL":
            print(f"  Layer 1.6: No NHL injury collector - skipping")

        # ── Save game data ────────────────────────────────────────────────────────
        sport_key = sport.lower()
        out_file = DATA_DIR / f"games_{sport_key}_{date}.json"
        output_data = {
            "sport": sport,
            "date": date_fmt,
            "fetched_at": datetime.now().isoformat(),
            "games": games,
            "count": len(games),
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Game data saved: {out_file.name}")

        # ── Layer 2: Roster Profiles ──────────────────────────────────────────────
        team_grades = {}
        if not skip_profiles:
            print(f"\n  Layer 2: Running roster profiles (all 4 Azure models)...")
            all_teams = list({g["home"] for g in games} | {g["away"] for g in games})

            # Run in batches of 4 (roster_profile.py does 4 teams at once)
            from roster_profile import profile_team
            from pathlib import Path as _Path

            profile_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                futures = []
                for team in all_teams:
                    fut = ex.submit(profile_team, team, games)
                    futures.append((team, fut))
                for team, fut in futures:
                    try:
                        result = fut.result(timeout=120)
                        if result:
                            profile_results.append(result)
                            grade = result.get("consensus", {}).get("grade", "C")
                            team_grades[team] = grade
                            print(f"    {team}: {grade}")
                    except Exception as e:
                        print(f"    {team}: ERROR - {e}")

            # Save
            profile_out = GRADES_DIR / f"{sport_key}_roster_profiles_{date}.json"
            with open(profile_out, "w", encoding="utf-8") as f:
                json.dump({
                    "sport": sport, "date": date, "teams": profile_results
                }, f, indent=2)
            print(f"  Roster profiles saved: {profile_out.name}")
        else:
            # Load existing profiles
            profile_file = GRADES_DIR / f"{sport_key}_roster_profiles_{date}.json"
            if profile_file.exists():
                with open(profile_file, encoding="utf-8") as f:
                    pdata = json.load(f)
                for t in pdata.get("teams", []):
                    team_grades[t.get("team", "")] = t.get("consensus", {}).get("grade", "C")
                print(f"  Roster profiles loaded: {len(team_grades)} teams")

        # ── Layer 2.5: Player Profiles + Chains ───────────────────────────────────
        player_chains = []
        if not skip_players and not skip_profiles:
            print(f"\n  Layer 2.5: Running player profiles...")
            try:
                from player_profile import profile_game as _profile_game
                for game in games:
                    chains = _profile_game(game, team_grades, date, sport)
                    player_chains.extend(chains)
                    time.sleep(0.5)

                chains_out = GRADES_DIR / f"player_chains_{sport_key}_{date}.json"
                with open(chains_out, "w", encoding="utf-8") as f:
                    json.dump({
                        "sport": sport, "date": date,
                        "chains_fired": sum(1 for c in player_chains if c.get("math_chain", {}).get("fired")),
                        "players": player_chains,
                    }, f, indent=2)
                print(f"  Player chains saved: {chains_out.name} ({len(player_chains)} players)")
            except Exception as e:
                print(f"  [player] Error: {e} - skipping player chains")
        else:
            # Try to load existing
            chains_file = GRADES_DIR / f"player_chains_{sport_key}_{date}.json"
            if chains_file.exists():
                with open(chains_file, encoding="utf-8") as f:
                    cdata = json.load(f)
                player_chains = cdata.get("players", [])

        # ── Layer 3: Grade Engine ─────────────────────────────────────────────────
        print(f"\n  Layer 3: Running grade engine...")
        try:
            result = subprocess.run(
                [sys.executable, str(BASE_DIR / "grade_engine.py"), sport_key, date],
                capture_output=True, text=True, cwd=str(BASE_DIR)
            )
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    print(f"    {line}")
        except Exception as e:
            print(f"  Grade engine error: {e}")

        # ── Layer 3.5: Swim Lane H2H Race ─────────────────────────────────────────
        print(f"\n  Layer 3.5: Swim lane H2H race...")
        races = []
        for game in games:
            race = run_swim_lane(game, team_grades, player_chains)
            races.append(race)
            print_race(race)

        all_races[sport] = races

        # Save races
        race_out = GRADES_DIR / f"race_{sport_key}_{date}.json"
        with open(race_out, "w", encoding="utf-8") as f:
            json.dump({"sport": sport, "date": date, "races": races}, f, indent=2)

    # ── Final Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL SLATE - {date_fmt}")
    print(f"{'='*60}")

    for sport, races in all_races.items():
        print(f"\n  {sport}")
        for race in races:
            if race["race_winner"] != "push":
                print(f"    > {race['pick_team']} {race['pick_spread']} | {race['bet_size']} | {race['confidence']}")
            else:
                print(f"    - {race['matchup']} | PASS")

    print(f"\n{'='*60}\n")
    return all_races


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Crew V3 Pipeline")
    parser.add_argument("--date", default=None, help="Date YYYYMMDD (default: today)")
    parser.add_argument("--sport", default=None, help="Sport: NBA, NHL, NCAAB (default: all)")
    parser.add_argument("--skip-profiles", action="store_true", help="Skip AI model calls, use cached profiles")
    parser.add_argument("--skip-players", action="store_true", help="Skip player chain analysis")
    args = parser.parse_args()

    sports = [args.sport.upper()] if args.sport else None
    run_pipeline(
        sports=sports,
        date=args.date,
        skip_profiles=args.skip_profiles,
        skip_players=args.skip_players,
    )

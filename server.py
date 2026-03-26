"""
Layer 5: Local Web Server
Serves the consensus engine dashboard at http://localhost:8080
Reads from data/ and grades/ directories.
"""

import json
import os
import sys
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
GRADES_DIR = BASE_DIR / "grades"
STATIC_DIR = BASE_DIR / "static"
DASHBOARD_DIR = BASE_DIR / "slate_dashboard"

app = FastAPI(title="Edge Consensus Engine", version="0.1.0")
SUPPORTED_SPORTS = ["nba", "nhl", "ncaab"]
AUTO_ANALYZE_ENABLED = os.environ.get("EDGE_AUTO_ANALYZE", "1").lower() not in {"0", "false", "no"}
AUTO_ANALYZE_LOCK = threading.Lock()
AUTO_ANALYZE_IN_PROGRESS = False
AUTO_ANALYZE_STAMP = DATA_DIR / ".auto_analyze_stamp"

# Serve static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main dashboard (slate dashboard)."""
    # Prefer new slate dashboard
    dash = DASHBOARD_DIR / "index.html"
    if dash.exists():
        return HTMLResponse(dash.read_text(encoding="utf-8"))
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Edge Crew V3</h1><p>No dashboard found.</p>")


# ── Slate API (for dashboard) ────────────────────────────────────────────────────

def _load_team_grades(sport: str, date: str) -> dict:
    """Load team grades from roster profiles."""
    f = GRADES_DIR / f"{sport.lower()}_roster_profiles_{date}.json"
    if not f.exists():
        return {}
    with open(f, encoding="utf-8") as fh:
        data = json.load(fh)
    return {t.get("team", ""): t.get("consensus", {}).get("grade", "C") for t in data.get("teams", [])}


def _run_process(cmd: list[str], timeout: int = 120) -> dict:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(BASE_DIR),
        )
        output = (proc.stdout or proc.stderr or "").strip()
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "output": output[-4000:],
        }
    except Exception as e:
        return {"status": "error", "output": str(e)}


def _sports_with_data(date: str) -> list[str]:
    return [
        sport_key
        for sport_key in SUPPORTED_SPORTS
        if (DATA_DIR / f"games_{sport_key}_{date}.json").exists()
    ]


def _should_auto_analyze(date: str) -> bool:
    if not AUTO_ANALYZE_ENABLED:
        return False
    if _sports_with_data(date):
        return False
    if AUTO_ANALYZE_STAMP.exists():
        stamped_for = AUTO_ANALYZE_STAMP.read_text(encoding="utf-8").strip()
        if stamped_for == date:
            return False
    return True


def _mark_auto_analyze(date: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUTO_ANALYZE_STAMP.write_text(date, encoding="utf-8")


def _run_auto_analyze_for_today():
    global AUTO_ANALYZE_IN_PROGRESS
    today = datetime.now().strftime("%Y%m%d")
    with AUTO_ANALYZE_LOCK:
        if AUTO_ANALYZE_IN_PROGRESS or not _should_auto_analyze(today):
            return
        AUTO_ANALYZE_IN_PROGRESS = True
    try:
        _mark_auto_analyze(today)
        for sport_key in SUPPORTED_SPORTS:
            _run_process(
                ["python", str(BASE_DIR / "run_pipeline.py"), "--sport", sport_key.upper(), "--date", today],
                timeout=600,
            )
    finally:
        with AUTO_ANALYZE_LOCK:
            AUTO_ANALYZE_IN_PROGRESS = False


@app.on_event("startup")
async def startup_event():
    if _should_auto_analyze(datetime.now().strftime("%Y%m%d")):
        threading.Thread(target=_run_auto_analyze_for_today, daemon=True).start()


@app.get("/flowchart", response_class=HTMLResponse)
async def flowchart():
    f = DASHBOARD_DIR / "flowchart.html"
    if f.exists():
        return HTMLResponse(f.read_text(encoding="utf-8"))
    raise HTTPException(404, "Flowchart not found")


@app.get("/api/slate/{sport}")
async def get_slate(sport: str, date: str | None = None):
    """Merged game + grades + injuries for dashboard."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    sport_key = sport.lower()

    data_file = DATA_DIR / f"games_{sport_key}_{date}.json"
    if not data_file.exists():
        return {"sport": sport.upper(), "date": date, "games": [], "count": 0}

    with open(data_file, encoding="utf-8") as f:
        game_data = json.load(f)

    # Load grades
    grade_file = GRADES_DIR / f"{sport_key}_{date}_grades.json"
    grades_by_game = {}
    if grade_file.exists():
        with open(grade_file, encoding="utf-8") as f:
            gdata = json.load(f)
        for g in gdata.get("games", []):
            game_id = g.get("game_id")
            if game_id:
                grades_by_game[game_id] = g

    # Load team profiles
    team_grades = _load_team_grades(sport, date)

    games = game_data.get("games", [])
    for game in games:
        gid = game.get("game_id", "")
        grade_entry = grades_by_game.get(gid, {})
        odds = game.get("odds", {})
        game["consensus_grade"] = grade_entry.get("consensus_grade", "—")
        game["sizing"] = grade_entry.get("profiles", {}).get("sintonia", {}).get("sizing", "PASS")
        game["pick"] = grade_entry.get("profiles", {}).get("sintonia", {}).get("pick", "")
        game["home_grade"] = team_grades.get(game.get("home", ""), "—")
        game["away_grade"] = team_grades.get(game.get("away", ""), "—")
        game["spread_home_val"] = odds.get("spread_home_val", odds.get("spread_home"))
        game["spread_away_val"] = odds.get("spread_away_val", odds.get("spread_away"))
        game["spread"] = game["spread_home_val"]
        game["total_current"] = odds.get("total_current", odds.get("total"))
        game["total"] = odds.get("total", game["total_current"])

    return {"sport": sport.upper(), "date": date, "games": games, "count": len(games)}


@app.get("/api/race/{sport}")
async def get_race(sport: str, date: str | None = None):
    """Swim lane H2H race results."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    race_file = GRADES_DIR / f"race_{sport.lower()}_{date}.json"
    if not race_file.exists():
        return {"sport": sport.upper(), "date": date, "races": []}
    with open(race_file, encoding="utf-8") as f:
        data = json.load(f)

    slate_file = DATA_DIR / f"games_{sport.lower()}_{date}.json"
    games_by_matchup = {}
    if slate_file.exists():
        with open(slate_file, encoding="utf-8") as fh:
            slate = json.load(fh)
        for game in slate.get("games", []):
            key = f"{game.get('away', '')} @ {game.get('home', '')}"
            games_by_matchup[key] = game

    for race in data.get("races", []):
        if race.get("pick_spread") not in (None, "", "?"):
            continue
        matchup = race.get("matchup", "")
        game = games_by_matchup.get(matchup, {})
        odds = game.get("odds", {})
        if race.get("race_winner") == "home":
            race["pick_spread"] = odds.get("spread_home_val", odds.get("spread_home", "?"))
        elif race.get("race_winner") == "away":
            race["pick_spread"] = odds.get("spread_away_val", odds.get("spread_away", "?"))
        else:
            race["pick_spread"] = ""

    return data


@app.get("/api/workbench/{sport}")
async def get_workbench(sport: str, date: str | None = None):
    """Full workbench data: profiles + race + props for all games."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    sport_key = sport.lower()

    # Load game schedule
    data_file = DATA_DIR / f"games_{sport_key}_{date}.json"
    games = []
    if data_file.exists():
        with open(data_file, encoding="utf-8") as f:
            games = json.load(f).get("games", [])

    # Load all profile files
    profiles = {}
    for pattern in [f"NBA_PROFILE_*_{date}.json", f"{sport_key}_roster_profiles_{date}.json"]:
        for pf in GRADES_DIR.glob(pattern):
            with open(pf, encoding="utf-8") as f:
                pd = json.load(f)
            teams_data = pd.get("teams", {})
            if isinstance(teams_data, dict):
                for tname, tdata in teams_data.items():
                    profiles[tname] = tdata
            elif isinstance(teams_data, list):
                for t in teams_data:
                    tname = t.get("team", "")
                    if tname:
                        profiles[tname] = t

    # Load race data
    race_file = GRADES_DIR / f"race_{sport_key}_{date}.json"
    races = {}
    if race_file.exists():
        with open(race_file, encoding="utf-8") as f:
            rd = json.load(f)
        for r in rd.get("races", []):
            key = f"{r.get('away', '')}@{r.get('home', '')}"
            races[key] = r

    # Load player chains
    chains_file = GRADES_DIR / f"player_chains_{sport_key}_{date}.json"
    player_chains = []
    if chains_file.exists():
        with open(chains_file, encoding="utf-8") as f:
            player_chains = json.load(f).get("players", [])

    # Build workbench entries
    result = []
    for game in games:
        home = game.get("home", "")
        away = game.get("away", "")

        # Match profiles (fuzzy)
        home_profile = next((profiles[t] for t in profiles if home.lower() in t.lower() or t.lower() in home.lower()), {})
        away_profile = next((profiles[t] for t in profiles if away.lower() in t.lower() or t.lower() in away.lower()), {})

        # Match race
        race_key = f"{away}@{home}"
        race = races.get(race_key, {})

        # Match player chains for this game
        home_players = [p for p in player_chains if p.get("team", "").lower() in home.lower() or home.lower() in p.get("team", "").lower()]
        away_players = [p for p in player_chains if p.get("team", "").lower() in away.lower() or away.lower() in p.get("team", "").lower()]

        result.append({
            "game_id": game.get("game_id", ""),
            "home": home,
            "away": away,
            "time": game.get("time", ""),
            "spread": game.get("spread", "?"),
            "total": game.get("total", "?"),
            "home_profile": home_profile,
            "away_profile": away_profile,
            "race": race,
            "home_players": home_players,
            "away_players": away_players,
        })

    return {"sport": sport.upper(), "date": date, "games": result, "count": len(result)}


@app.get("/workbench", response_class=HTMLResponse)
async def workbench():
    f = DASHBOARD_DIR / "workbench.html"
    if f.exists():
        return HTMLResponse(f.read_text(encoding="utf-8"))
    raise HTTPException(404, "Workbench not found")


@app.get("/api/props/{sport}")
async def get_props(sport: str, date: str | None = None):
    """Player chains / prop edges."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    chains_file = GRADES_DIR / f"player_chains_{sport.lower()}_{date}.json"
    if not chains_file.exists():
        return {"sport": sport.upper(), "date": date, "players": [], "chains_fired": 0}
    with open(chains_file, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/sports")
async def get_sports():
    """List available sports for today."""
    today = datetime.now().strftime("%Y%m%d")
    sports = []
    for sport_key in SUPPORTED_SPORTS:
        data_file = DATA_DIR / f"games_{sport_key}_{today}.json"
        grade_file = GRADES_DIR / f"{sport_key}_{today}_grades.json"
        race_file = GRADES_DIR / f"race_{sport_key}_{today}.json"
        props_file = GRADES_DIR / f"player_chains_{sport_key}_{today}.json"
        entry = {
            "key": sport_key,
            "name": sport_key.upper(),
            "games": 0,
            "has_data": data_file.exists(),
            "has_grades": grade_file.exists(),
            "has_race": race_file.exists(),
            "has_props": props_file.exists(),
            "fetched_at": "",
        }
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            entry["games"] = data.get("count", 0)
            entry["fetched_at"] = data.get("fetched_at", "")
        sports.append(entry)
    return sports


@app.get("/api/games/{sport}")
async def get_games(sport: str, date: str | None = None):
    """Get all games for a sport with their data."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    data_file = DATA_DIR / f"games_{sport.lower()}_{date}.json"
    if not data_file.exists():
        return {"sport": sport.upper(), "date": date, "games": [], "count": 0}

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


@app.get("/api/grades/{sport}")
async def get_grades(sport: str, date: str | None = None):
    """Get grades for a sport."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    grade_file = GRADES_DIR / f"{sport.lower()}_{date}_grades.json"
    if not grade_file.exists():
        return {"sport": sport.upper(), "date": date, "games": [], "count": 0}

    with open(grade_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


@app.get("/api/game/{game_id}")
async def get_game_detail(game_id: str, date: str | None = None):
    """Get detailed data + grades for a specific game."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    result = {"game": None, "grades": {}}

    # Search across sports
    for sport_key in ["nba", "nhl", "ncaab"]:
        data_file = DATA_DIR / f"games_{sport_key}_{date}.json"
        if not data_file.exists():
            continue

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for game in data.get("games", []):
            if game.get("game_id") == game_id:
                result["game"] = game

                # Look for grades
                grade_file = GRADES_DIR / f"{sport_key}_{date}_grades.json"
                if grade_file.exists():
                    with open(grade_file, "r", encoding="utf-8") as gf:
                        grades_data = json.load(gf)
                    for graded in grades_data.get("games", []):
                        if graded.get("game_id") == game_id:
                            result["grades"] = graded
                            break
                return result

    return {"game": None, "grades": {}, "status": "not_found", "game_id": game_id}


@app.get("/api/pipeline/{sport}")
async def get_pipeline(sport: str, date: str | None = None):
    """V3 Pipeline — combined data for full analysis flow view."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    sport_key = sport.lower()

    result = {"sport": sport.upper(), "date": date}

    # Load all data sources
    data_file = DATA_DIR / f"games_{sport_key}_{date}.json"
    if data_file.exists():
        with open(data_file, encoding="utf-8") as f:
            result["games"] = json.load(f)
    else:
        result["games"] = {"games": [], "count": 0}

    grade_file = GRADES_DIR / f"{sport_key}_{date}_grades.json"
    if grade_file.exists():
        with open(grade_file, encoding="utf-8") as f:
            result["grades"] = json.load(f)
    else:
        result["grades"] = {"games": []}

    race_file = GRADES_DIR / f"race_{sport_key}_{date}.json"
    if race_file.exists():
        with open(race_file, encoding="utf-8") as f:
            result["race"] = json.load(f)
    else:
        result["race"] = {"races": []}

    chains_file = GRADES_DIR / f"player_chains_{sport_key}_{date}.json"
    if chains_file.exists():
        with open(chains_file, encoding="utf-8") as f:
            result["chains"] = json.load(f)
    else:
        result["chains"] = {"players": []}

    profile_file = GRADES_DIR / f"{sport_key}_roster_profiles_{date}.json"
    if profile_file.exists():
        with open(profile_file, encoding="utf-8") as f:
            result["profiles"] = json.load(f)
    else:
        result["profiles"] = {"teams": []}

    return result


@app.post("/api/model/{game_id}")
async def trigger_model(game_id: str, model: str = "grok"):
    """Push a single game to Grok (or another model) for grading."""
    from model_caller import grade_game_with_model
    try:
        result = grade_game_with_model(game_id, model_key=model)
        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/collect/{sport}")
async def trigger_collect(sport: str):
    """Trigger data collection for a sport (runs collectors)."""
    date = datetime.now().strftime("%Y%m%d")
    sport_upper = sport.upper()

    results = {}

    # Run collectors sequentially
    collectors = [
        ("odds", ["python", str(BASE_DIR / "collect_odds.py"), sport_upper]),
        ("injuries", ["python", str(BASE_DIR / "collect_injuries.py"), sport_upper]),
        ("stats", ["python", str(BASE_DIR / "collect_stats.py"), sport_upper]),
    ]
    if sport_upper == "NBA":
        collectors.append(("props", ["python", str(BASE_DIR / "collect_props.py"), date]))

    for name, cmd in collectors:
        results[name] = _run_process(cmd, timeout=120)

    return results


@app.post("/api/grade/{sport}")
async def trigger_grade(sport: str):
    """Trigger grading for a sport."""
    return _run_process(["python", str(BASE_DIR / "grade_engine.py"), sport.upper()], timeout=60)


@app.post("/api/analyze/{sport}")
async def analyze_sport(sport: str, date: str | None = None):
    """Run the full pipeline for one sport."""
    sport_key = sport.lower()
    if sport_key not in SUPPORTED_SPORTS:
        raise HTTPException(400, f"Unsupported sport: {sport}")
    target_date = date or datetime.now().strftime("%Y%m%d")
    result = _run_process(
        ["python", str(BASE_DIR / "run_pipeline.py"), "--sport", sport_key.upper(), "--date", target_date],
        timeout=600,
    )
    return {"sport": sport_key.upper(), "date": target_date, **result}


@app.post("/api/analyze")
async def analyze_all(date: str | None = None):
    """Run the full pipeline for all supported sports."""
    target_date = date or datetime.now().strftime("%Y%m%d")
    results = {}
    overall = "ok"
    for sport_key in SUPPORTED_SPORTS:
        result = _run_process(
            ["python", str(BASE_DIR / "run_pipeline.py"), "--sport", sport_key.upper(), "--date", target_date],
            timeout=600,
        )
        results[sport_key] = result
        if result["status"] != "ok":
            overall = "error"
    return {"status": overall, "date": target_date, "sports": results}


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"\n  Edge Consensus Engine — http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)

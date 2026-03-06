import os
import time
import json
import uuid
import hashlib
import secrets
import logging
import httpx
import re
import asyncio
from openai import AzureOpenAI
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger("edge-crew")

PST = ZoneInfo("America/Los_Angeles")
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response


import html as _html

def _sanitize(s):
    """Strip HTML/script tags from user input."""
    if not isinstance(s, str):
        return s
    return _html.escape(s.strip())

app = FastAPI()


async def _autograde_loop():
    """Background loop: auto-grade picks every 15 min between 10PM-6AM PST."""
    await asyncio.sleep(60)  # Wait 60s after startup before first check
    while True:
        try:
            now = datetime.now(PST)
            if now.hour >= 22 or now.hour < 6:
                # Fetch ungraded picks from Supabase
                if sb:
                    res = sb.table("picks").select("*").is_("result", "null").execute()
                    ungraded = res.data or []
                else:
                    data = _read_picks()
                    ungraded = [p for p in data.get("picks", []) if not p.get("result")]

                if ungraded:
                    ungraded = _deduplicate_picks(ungraded)
                    sports_needed = set()
                    for p in ungraded:
                        sport = p.get("sport", "").lower()
                        if sport in SPORT_KEYS:
                            sports_needed.add(sport)
                    if not sports_needed:
                        sports_needed = {"nba", "wnba", "nhl"}

                    fetch_tasks = []
                    for sport in sports_needed:
                        for key in SPORT_KEYS.get(sport, []):
                            fetch_tasks.append(_fetch_scores(key))

                    all_scores = []
                    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                    for result in results:
                        if not isinstance(result, Exception) and result:
                            all_scores.extend(result)

                    completed = [g for g in all_scores if g.get("completed")]
                    graded = 0
                    for pick in ungraded:
                        pick_type = pick.get("type", "").lower()
                        if pick_type == "parlay":
                            grade = _grade_parlay(pick, completed)
                        else:
                            grade = None
                            for game in completed:
                                if _teams_match(pick.get("matchup", ""), game.get("home_team", ""), game.get("away_team", "")):
                                    grade = _grade_pick_against_score(pick, game)
                                    if grade:
                                        break
                        if grade and sb:
                            try:
                                sb.table("picks").update({
                                    "result": grade,
                                    "graded_at": datetime.now(PST).isoformat(),
                                }).eq("id", pick.get("id", "")).execute()
                                graded += 1
                            except Exception:
                                pass
                    if graded:
                        print(f"[AUTOGRADE] Graded {graded}/{len(ungraded)} picks at {now.strftime('%I:%M %p PST')}")
        except Exception as e:
            print(f"[AUTOGRADE] Error: {e}")
        await asyncio.sleep(900)  # 15 minutes


@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(_autograde_loop())


@app.get("/health")
async def health_check():
    """Health check for Render auto-restart."""
    return JSONResponse({"status": "ok", "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ")})


# ===== CREW AUTH =====
CREW_PIN_SALT = os.environ.get("CREW_PIN_SALT", "edge-crew-default-salt-change-me")
PROFILES_FILE = os.path.join(os.path.dirname(__file__), "data", "crew_profiles.json")  # stays in app dir (not persistent — profiles are small + seeded)
_sessions = {}  # token -> {id, display_name, color, is_admin}

DEFAULT_CREW = [
    {"id": "peter", "display_name": "Peter", "color": "#D4A017", "is_admin": True},
    {"id": "chinny", "display_name": "Chinny", "color": "#10B981", "is_admin": False},
    {"id": "jimmy", "display_name": "Jimmy", "color": "#60A5FA", "is_admin": False},
    {"id": "alyssa", "display_name": "Alyssa", "color": "#E879F9", "is_admin": False},
    {"id": "sintonia", "display_name": "Sinton.ia", "color": "#A78BFA", "is_admin": False},
]


def _hash_pin(pin: str) -> str:
    return hashlib.sha256((CREW_PIN_SALT + pin).encode()).hexdigest()


def _read_profiles():
    try:
        with open(PROFILES_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"profiles": []}


def _write_profiles(data):
    os.makedirs(os.path.dirname(PROFILES_FILE), exist_ok=True)
    with open(PROFILES_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _seed_profiles():
    """Pre-seed default crew if profiles file is empty. Default PIN: 0000."""
    data = _read_profiles()
    if data.get("profiles"):
        return
    default_pin_hash = _hash_pin("0000")
    for member in DEFAULT_CREW:
        data.setdefault("profiles", []).append({
            "id": member["id"],
            "display_name": member["display_name"],
            "pin_hash": default_pin_hash,
            "color": member["color"],
            "is_admin": member["is_admin"],
            "created_at": datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S"),
            "last_login": None,
        })
    _write_profiles(data)


_seed_profiles()


def _get_crew(request: Request):
    """Read X-Crew-Token header, return profile dict or None."""
    token = request.headers.get("x-crew-token", "")
    return _sessions.get(token)


@app.post("/api/auth/register")
async def auth_register(request: Request):
    body = await request.json()
    name = (body.get("name") or "").strip()
    pin = (body.get("pin") or "").strip()
    color = (body.get("color") or "#D4A017").strip()

    if not name or len(name) < 2:
        return JSONResponse({"error": "Name must be at least 2 characters"}, status_code=400)
    if not pin or len(pin) != 4 or not pin.isdigit():
        return JSONResponse({"error": "PIN must be exactly 4 digits"}, status_code=400)

    data = _read_profiles()
    for p in data.get("profiles", []):
        if p["display_name"].lower() == name.lower():
            return JSONResponse({"error": "Name already taken"}, status_code=409)

    profile = {
        "id": str(uuid.uuid4())[:8],
        "display_name": name,
        "pin_hash": _hash_pin(pin),
        "color": color,
        "is_admin": False,
        "created_at": datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S"),
        "last_login": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    data.setdefault("profiles", []).append(profile)
    _write_profiles(data)

    token = secrets.token_urlsafe(32)
    _sessions[token] = {
        "id": profile["id"],
        "display_name": profile["display_name"],
        "color": profile["color"],
        "is_admin": profile["is_admin"],
    }
    return JSONResponse({"status": "ok", "token": token, "profile": _sessions[token]})


@app.post("/api/auth/login")
async def auth_login(request: Request):
    body = await request.json()
    name = (body.get("name") or "").strip()
    pin = (body.get("pin") or "").strip()

    if not name or not pin:
        return JSONResponse({"error": "Name and PIN required"}, status_code=400)

    data = _read_profiles()
    pin_hash = _hash_pin(pin)
    for p in data.get("profiles", []):
        if p["display_name"].lower() == name.lower() and p["pin_hash"] == pin_hash:
            p["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _write_profiles(data)

            token = secrets.token_urlsafe(32)
            _sessions[token] = {
                "id": p["id"],
                "display_name": p["display_name"],
                "color": p["color"],
                "is_admin": p.get("is_admin", False),
            }
            return JSONResponse({"status": "ok", "token": token, "profile": _sessions[token]})

    return JSONResponse({"error": "Invalid name or PIN"}, status_code=401)


@app.get("/api/auth/me")
async def auth_me(request: Request):
    crew = _get_crew(request)
    if not crew:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    return JSONResponse({"status": "ok", "profile": crew})


@app.post("/api/auth/logout")
async def auth_logout(request: Request):
    token = request.headers.get("x-crew-token", "")
    _sessions.pop(token, None)
    return JSONResponse({"status": "ok"})


@app.get("/api/auth/profiles")
async def auth_profiles():
    """Return list of all crew names + colors (no PINs). For login dropdown."""
    data = _read_profiles()
    return JSONResponse({
        "profiles": [
            {"display_name": p["display_name"], "color": p["color"]}
            for p in data.get("profiles", [])
        ]
    })



@app.middleware("http")
async def no_cache_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Surrogate-Control"] = "no-store"
    return response


ODDS_API_KEY = os.environ.get("ODDS_API_KEY_PAID", "") or os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
PREFERRED_BOOK = "hardrockbet"
FALLBACK_BOOKS = ["draftkings", "fanduel", "betmgm", "bovada"]
REGIONS = "us,us2"
# Persistent data directory — uses /data mount on Render, falls back to local ./data
DATA_DIR = "/data" if os.path.isdir("/data") else os.path.join(os.path.dirname(__file__), "data")
UPSETS_FILE = os.path.join(DATA_DIR, "upsets.json")
PICKS_FILE = os.path.join(DATA_DIR, "picks.json")

# SharpAPI config (primary odds source)
SHARPAPI_KEY = os.environ.get("SHARPAPI_KEY", "")
SHARPAPI_BASE = "https://api.sharpapi.io/api/v1"
SHARPAPI_LEAGUES = {
    "nba": "nba",
    "nhl": "nhl",
    "soccer": "soccer",
    "mma": "ufc",
    "boxing": "boxing",
}
# SharpAPI uses different totals market names per sport
SHARPAPI_MARKETS = {
    "nba": "moneyline,point_spread,total_points",
    "nhl": "moneyline,point_spread,total_goals",
    "soccer": "moneyline,point_spread,total_goals",
    "mma": "moneyline",
    "boxing": "moneyline,total_rounds",
}

# API-Sports config (lineups + injuries across all sports)
API_SPORTS_KEY = os.environ.get("API_SPORTS_KEY", "")
API_SPORTS_HOSTS = {
    "nba": "v2.nba.api-sports.io",
    "nhl": "v1.hockey.api-sports.io",
    "soccer": "v3.football.api-sports.io",
    "mlb": "v1.baseball.api-sports.io",
}
API_SPORTS_LEAGUES = {
    "nhl": 57,
    "mlb": 1,
    "soccer": [39, 140, 253, 2, 262],  # EPL, La Liga, MLS, UCL, Liga MX
}

# Azure OpenAI config
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_MODEL = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
REALTIME_DEPLOYMENT = "gpt-4o-realtime"
AZURE_BASE = AZURE_ENDPOINT.rstrip("/")

# EV Agent character config for voice
EV_AGENT_VOICE = {
    "voice": "ash",
    "prompt": """You are EV, the Expected Value agent for Edge Crew — a sports betting analytics platform. You are 100% math, no gut, pure edge. You analyze odds, find value, grade picks, and break down matchups.

Your personality: sharp, confident, data-driven. You speak in short punchy sentences. You know every sport — NBA, NFL, NHL, MLB, NCAAB, NCAAF, Tennis, Soccer, MMA, Boxing, WNBA. You reference real odds, spreads, and over/unders.

When asked about picks: give the edge percentage, the why, and the confidence level. When grading: be honest, show the math. When a user is on a streak, hype them up. When they're cold, analyze what went wrong.

Keep responses under 3 sentences for voice. Be the sharpest analyst in the room."""
}

# Supabase config (persistent storage — replaces file-based picks/upsets)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
sb = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        sb = None

# Simple in-memory cache to save API credits
_cache = {}
CACHE_TTL = 300  # 5 minutes
ANALYSIS_CACHE_TTL = 900  # 15 minutes for analysis

# Opening lines tracker — stores first-seen odds per game per day
_opening_lines = {}  # key: "YYYY-MM-DD:{game_id}" -> {spread, total, away_ml, home_ml}


def _now_ts():
    """Return current timestamp string for API responses (PST)."""
    return datetime.now(PST).strftime("%I:%M %p PST — %b %d, %Y")


def _get_cached(key, ttl=None):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < (ttl or CACHE_TTL):
            return data
    return None


def _set_cache(key, data):
    _cache[key] = (data, time.time())


def _track_opening_lines(game):
    """Store opening lines for a game (first time seen today). Returns shift data."""
    today = datetime.now(PST).strftime("%Y-%m-%d")
    key = f"{today}:{game.get('id', '')}"
    spread = game.get("home_spread")
    total = game.get("total")
    away_ml = game.get("away_ml")
    home_ml = game.get("home_ml")

    if key not in _opening_lines:
        _opening_lines[key] = {
            "spread": spread, "total": total,
            "away_ml": away_ml, "home_ml": home_ml,
            "ts": time.time(),
        }
        return None  # First fetch — no shift yet

    opening = _opening_lines[key]
    shifts = {}
    if spread is not None and opening["spread"] is not None and spread != opening["spread"]:
        shifts["spread"] = {"open": opening["spread"], "now": spread, "delta": round(spread - opening["spread"], 1)}
    if total is not None and opening["total"] is not None and total != opening["total"]:
        shifts["total"] = {"open": opening["total"], "now": total, "delta": round(total - opening["total"], 1)}
    if away_ml is not None and opening["away_ml"] is not None and away_ml != opening["away_ml"]:
        shifts["away_ml"] = {"open": opening["away_ml"], "now": away_ml}
    if home_ml is not None and opening["home_ml"] is not None and home_ml != opening["home_ml"]:
        shifts["home_ml"] = {"open": opening["home_ml"], "now": home_ml}
    return shifts if shifts else None


def _american_to_implied(odds):
    """Convert American odds to implied probability (0-1)."""
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def _detect_arbitrage(event, sport_label):
    """Scan all bookmakers for arbitrage opportunities on an event.
    Returns list of arb opps with book names and profit %."""
    bookmakers = event.get("bookmakers", [])
    if len(bookmakers) < 2:
        return []

    arbs = []

    # --- ML Arbitrage ---
    away_team = event["away_team"]
    home_team = event["home_team"]
    best_away_ml = {"odds": None, "implied": 1.0, "book": None}
    best_home_ml = {"odds": None, "implied": 1.0, "book": None}

    for bk in bookmakers:
        for market in bk.get("markets", []):
            if market["key"] == "h2h":
                for outcome in market["outcomes"]:
                    price = outcome.get("price", 0)
                    imp = _american_to_implied(price)
                    if imp is None:
                        continue
                    if outcome["name"] == away_team and imp < best_away_ml["implied"]:
                        best_away_ml = {"odds": price, "implied": imp, "book": bk["key"]}
                    elif outcome["name"] == home_team and imp < best_home_ml["implied"]:
                        best_home_ml = {"odds": price, "implied": imp, "book": bk["key"]}

    if best_away_ml["book"] and best_home_ml["book"]:
        total_implied = best_away_ml["implied"] + best_home_ml["implied"]
        if total_implied < 1.0:
            profit_pct = round((1.0 / total_implied - 1.0) * 100, 2)
            arbs.append({
                "type": "ML",
                "profit_pct": profit_pct,
                "legs": [
                    {"side": away_team, "odds": best_away_ml["odds"], "book": best_away_ml["book"]},
                    {"side": home_team, "odds": best_home_ml["odds"], "book": best_home_ml["book"]},
                ],
            })

    # --- Totals Arbitrage ---
    best_over = {"odds": None, "implied": 1.0, "book": None, "point": None}
    best_under = {"odds": None, "implied": 1.0, "book": None, "point": None}

    for bk in bookmakers:
        for market in bk.get("markets", []):
            if market["key"] == "totals":
                for outcome in market["outcomes"]:
                    price = outcome.get("price", 0)
                    point = outcome.get("point")
                    imp = _american_to_implied(price)
                    if imp is None or point is None:
                        continue
                    if outcome["name"] == "Over" and imp < best_over["implied"]:
                        best_over = {"odds": price, "implied": imp, "book": bk["key"], "point": point}
                    elif outcome["name"] == "Under" and imp < best_under["implied"]:
                        best_under = {"odds": price, "implied": imp, "book": bk["key"], "point": point}

    if best_over["book"] and best_under["book"] and best_over["point"] == best_under["point"]:
        total_implied = best_over["implied"] + best_under["implied"]
        if total_implied < 1.0:
            profit_pct = round((1.0 / total_implied - 1.0) * 100, 2)
            arbs.append({
                "type": f"O/U {best_over['point']}",
                "profit_pct": profit_pct,
                "legs": [
                    {"side": f"Over {best_over['point']}", "odds": best_over["odds"], "book": best_over["book"]},
                    {"side": f"Under {best_under['point']}", "odds": best_under["odds"], "book": best_under["book"]},
                ],
            })

    return arbs


# ============================================================
# ANALYSIS ENGINE v2 - Lineup, Injury, REST, Weighted Matrix
# ============================================================

ROTOWIRE_URLS = {
    "nba": {
        "lineups": "https://www.rotowire.com/basketball/nba-lineups.php",
        "injuries": "https://www.rotowire.com/basketball/injury-report.php",
    },
    "nhl": {
        "lineups": "https://www.rotowire.com/hockey/nhl-lineups.php",
        "injuries": "https://www.rotowire.com/hockey/injury-report.php",
    },
    "soccer": {
        "lineups": "https://www.rotowire.com/soccer/lineups.php",
        "injuries": "https://www.rotowire.com/soccer/injury-report.php",
    },
}

NBA_MATRIX = [
    ("injuries_lineup", 10, "Injuries / lineup changes"),
    ("rest_advantage", 9, "Rest advantage (B2B, 3-in-5)"),
    ("sharp_vs_public", 8, "Where the money is (sharp vs public)"),
    ("def_ranking", 8, "Defensive ranking vs position"),
    ("pace_matchup", 7, "Pace matchup"),
    ("travel", 7, "Travel (road trip length)"),
    ("ats_trend", 6, "ATS trend (last 7/14 days)"),
    ("home_away", 5, "Home/away record"),
    ("coach", 4, "Coach tendencies"),
    ("revenge_rivalry", 3, "Revenge / rivalry"),
]

NHL_MATRIX = [
    ("goalie_confirmed", 10, "Goalie confirmed starter"),
    ("injuries_lineup", 9, "Injuries / lineup changes"),
    ("b2b_fatigue", 8, "Back-to-back / schedule fatigue"),
    ("home_away", 7, "Home/away record"),
    ("sharp_vs_public", 7, "Where the money is"),
    ("pp_pk", 7, "Power play / penalty kill rankings"),
    ("save_pct_trend", 6, "Save percentage trend (last 5)"),
    ("division_rivalry", 5, "Division / rivalry"),
    ("corsi_xg", 5, "Corsi / expected goals"),
    ("travel_tz", 4, "Travel / time zone"),
]

SOCCER_MATRIX = [
    ("btts_trend", 10, "Both Teams to Score trend"),
    ("injuries_lineup", 9, "Injuries / lineup changes"),
    ("home_away_form", 8, "Home/away form (last 5)"),
    ("goals_avg", 8, "Goals scored/conceded avg"),
    ("h2h", 7, "Head-to-head record"),
    ("league_position", 7, "League position / motivation"),
    ("xg_trend", 6, "xG (expected goals) trend"),
    ("sharp_vs_public", 6, "Where the money is"),
    ("clean_sheet", 5, "Clean sheet pct"),
    ("travel_congestion", 4, "Travel / fixture congestion"),
]

SPORT_MATRICES = {"nba": NBA_MATRIX, "nhl": NHL_MATRIX, "soccer": SOCCER_MATRIX}


async def _fetch_api_sports(sport_lower):
    """Fetch today's games + lineups from API-Sports. Returns structured text for AI prompt."""
    if not API_SPORTS_KEY:
        return ""
    host = API_SPORTS_HOSTS.get(sport_lower)
    if not host:
        return ""

    today = time.strftime("%Y-%m-%d")
    headers = {"x-apisports-key": API_SPORTS_KEY}
    cache_key = f"apisports:{sport_lower}:{today}"
    cached = _get_cached(cache_key, ttl=1800)
    if cached:
        return cached

    parts = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if sport_lower == "soccer":
                # Football: get fixtures for today across our leagues
                leagues = API_SPORTS_LEAGUES.get("soccer", [])
                all_fixtures = []
                for league_id in leagues:
                    resp = await client.get(
                        f"https://{host}/fixtures",
                        params={"date": today, "league": league_id, "season": 2025},
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        all_fixtures.extend(data.get("response", []))

                if all_fixtures:
                    parts.append(f"API-SPORTS SOCCER FIXTURES ({len(all_fixtures)} games today):")
                    for fix in all_fixtures[:20]:
                        fid = fix["fixture"]["id"]
                        home = fix["teams"]["home"]["name"]
                        away = fix["teams"]["away"]["name"]
                        status = fix["fixture"]["status"]["long"]
                        league_name = fix.get("league", {}).get("name", "?")
                        parts.append(f"  {away} @ {home} ({league_name}) | {status}")

                        # Fetch lineups for each fixture (only if started/finished or close to start)
                        if status in ("Not Started", "First Half", "Second Half", "Halftime"):
                            try:
                                lr = await client.get(
                                    f"https://{host}/fixtures/lineups",
                                    params={"fixture": fid},
                                    headers=headers,
                                )
                                if lr.status_code == 200:
                                    lineup_data = lr.json().get("response", [])
                                    for team_lu in lineup_data:
                                        tname = team_lu.get("team", {}).get("name", "?")
                                        formation = team_lu.get("formation", "?")
                                        starters = [p["player"]["name"] for p in team_lu.get("startXI", [])]
                                        if starters:
                                            parts.append(f"    {tname} ({formation}): {', '.join(starters)}")
                            except Exception:
                                pass

            elif sport_lower == "nba":
                resp = await client.get(
                    f"https://{host}/games",
                    params={"date": today},
                    headers=headers,
                )
                if resp.status_code == 200:
                    games = resp.json().get("response", [])
                    if games:
                        parts.append(f"API-SPORTS NBA GAMES ({len(games)} today):")
                        for g in games:
                            away = g["teams"]["visitors"]["name"]
                            home = g["teams"]["home"]["name"]
                            status = g["status"]["long"]
                            arena = g.get("arena", {}).get("name", "?")
                            parts.append(f"  {away} @ {home} | {status} | {arena}")

            elif sport_lower == "nhl":
                league_id = API_SPORTS_LEAGUES.get("nhl", 57)
                resp = await client.get(
                    f"https://{host}/games",
                    params={"date": today, "league": league_id, "season": 2025},
                    headers=headers,
                )
                if resp.status_code == 200:
                    games = resp.json().get("response", [])
                    if games:
                        parts.append(f"API-SPORTS NHL GAMES ({len(games)} today):")
                        for g in games:
                            away = g["teams"]["away"]["name"]
                            home = g["teams"]["home"]["name"]
                            status = g["status"]["long"]
                            parts.append(f"  {away} @ {home} | {status}")

            elif sport_lower == "mlb":
                league_id = API_SPORTS_LEAGUES.get("mlb", 1)
                resp = await client.get(
                    f"https://{host}/games",
                    params={"date": today, "league": league_id, "season": 2026},
                    headers=headers,
                )
                if resp.status_code == 200:
                    games = resp.json().get("response", [])
                    if games:
                        parts.append(f"API-SPORTS MLB GAMES ({len(games)} today):")
                        for g in games:
                            away = g["teams"]["away"]["name"]
                            home = g["teams"]["home"]["name"]
                            status = g["status"]["long"]
                            parts.append(f"  {away} @ {home} | {status}")

    except Exception as e:
        print(f"API-Sports fetch error ({sport_lower}): {e}")
        parts.append(f"API-Sports fetch failed: {e}")

    result = "\n".join(parts)
    if result:
        _set_cache(cache_key, result)
    return result


async def _fetch_api_sports_lineups(sport_lower):
    """Fetch per-game lineups from API-Sports. Returns dict of matchup -> lineup data for frontend."""
    if not API_SPORTS_KEY:
        return {}
    host = API_SPORTS_HOSTS.get(sport_lower)
    if not host:
        return {}

    today = time.strftime("%Y-%m-%d")
    headers = {"x-apisports-key": API_SPORTS_KEY}
    cache_key = f"apisports_lineups:{sport_lower}:{today}"
    cached = _get_cached(cache_key, ttl=1800)
    if cached:
        return cached

    lineups = {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if sport_lower == "soccer":
                leagues = API_SPORTS_LEAGUES.get("soccer", [])
                for league_id in leagues:
                    resp = await client.get(
                        f"https://{host}/fixtures",
                        params={"date": today, "league": league_id, "season": 2025},
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        for fix in resp.json().get("response", []):
                            fid = fix["fixture"]["id"]
                            home = fix["teams"]["home"]["name"]
                            away = fix["teams"]["away"]["name"]
                            key = f"{away} @ {home}"
                            try:
                                lr = await client.get(
                                    f"https://{host}/fixtures/lineups",
                                    params={"fixture": fid},
                                    headers=headers,
                                )
                                if lr.status_code == 200:
                                    lu_data = lr.json().get("response", [])
                                    if lu_data:
                                        match_lineups = {}
                                        for tlu in lu_data:
                                            tname = tlu.get("team", {}).get("name", "?")
                                            formation = tlu.get("formation", "?")
                                            starters = [p["player"]["name"] for p in tlu.get("startXI", [])]
                                            subs = [p["player"]["name"] for p in tlu.get("substitutes", [])[:5]]
                                            match_lineups[tname] = {
                                                "formation": formation,
                                                "starters": starters,
                                                "subs": subs,
                                            }
                                        lineups[key] = match_lineups
                            except Exception:
                                pass
    except Exception as e:
        print(f"API-Sports lineups fetch error ({sport_lower}): {e}")

    if lineups:
        _set_cache(cache_key, lineups)
    return lineups


async def _fetch_rotowire_page(url, cache_key, ttl=1800):
    """Fetch a RotoWire page, return raw HTML. Cached 30 min."""
    cached = _get_cached(cache_key, ttl=ttl)
    if cached:
        return cached
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml",
            })
            if resp.status_code == 200:
                _set_cache(cache_key, resp.text)
                return resp.text
    except Exception as e:
        print(f"RotoWire fetch failed ({url}): {e}")
    return ""


async def _get_lineup_and_injury_context(sport):
    """Fetch injury data from CBS Sports (server-rendered) + RotoWire lineups."""
    import re
    sport_lower = sport.lower()
    parts = []

    CBS_INJURY_URLS = {
        "nba": "https://www.cbssports.com/nba/injuries/",
        "wnba": "https://www.cbssports.com/wnba/injuries/",
        "nhl": "https://www.cbssports.com/nhl/injuries/",
        "soccer": None,
    }

    # --- CBS SPORTS INJURY REPORT (primary - server-rendered, reliable) ---
    cbs_url = CBS_INJURY_URLS.get(sport_lower)
    if cbs_url:
        cache_key = f"cbs_injuries:{sport_lower}"
        cached = _get_cached(cache_key, ttl=1800)
        if cached:
            parts.append(cached)
        else:
            try:
                async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                    resp = await client.get(cbs_url, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                    })
                    print(f"CBS Sports {sport} response: {resp.status_code}, size: {len(resp.text)}")
                    if resp.status_code == 200:
                        html = resp.text
                        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
                        player_rows = [r for r in rows if 'CellPlayerName' in r]
                        print(f"CBS Sports {sport}: {len(rows)} rows, {len(player_rows)} player rows")

                        if player_rows:
                            injury_lines = []
                            injury_lines.append(
                                f"INJURY REPORT ({sport.upper()} via CBS Sports"
                                f" - {len(player_rows)} players):"
                            )

                            for row in player_rows:
                                name_match = re.findall(r'>([^<]+)</a>', row)
                                tds = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
                                cells = []
                                for td in tds:
                                    text = re.sub(r'<[^>]+>', '', td).strip()
                                    if text:
                                        cells.append(text)

                                if name_match and len(cells) >= 4:
                                    full_name = name_match[-1] if len(name_match) > 1 else name_match[0]
                                    pos = cells[1] if len(cells) > 1 else "?"
                                    injury_type = cells[3] if len(cells) > 3 else "?"
                                    status = cells[4] if len(cells) > 4 else cells[-1]
                                    injury_lines.append(
                                        f"  - {full_name.strip()} ({pos})"
                                        f" | {injury_type} | {status}"
                                    )

                            injury_text = "\n".join(injury_lines)
                            _set_cache(cache_key, injury_text)
                            parts.append(injury_text)
                        else:
                            # Fallback: try broader pattern for any injury table
                            all_names = re.findall(r'<a[^>]*href="/nba/players/[^"]*"[^>]*>([^<]+)</a>', html)
                            statuses = re.findall(r'>(Out|Day-To-Day|Game Time Decision|Expected to be out[^<]*)<', html)
                            print(f"CBS fallback: {len(all_names)} names, {len(statuses)} statuses")
                            if all_names and statuses:
                                injury_lines = [f"INJURY REPORT ({sport.upper()} via CBS Sports - fallback parser):"]
                                for name, status in zip(all_names[:60], statuses[:60]):
                                    injury_lines.append(f"  - {name.strip()} | {status.strip()}")
                                injury_text = "\n".join(injury_lines)
                                _set_cache(cache_key, injury_text)
                                parts.append(injury_text)
                            else:
                                parts.append(f"CBS Sports returned {len(html)} bytes but no parseable injury data.")
                    else:
                        parts.append(f"CBS Sports returned HTTP {resp.status_code}.")
            except Exception as e:
                print(f"CBS injury fetch failed for {sport}: {e}")
                parts.append(f"CBS Sports injury fetch failed: {e}")

    # --- ROTOWIRE LINEUPS (secondary - best effort for lineup confirmations) ---
    urls = ROTOWIRE_URLS.get(sport_lower, {})
    lineup_url = urls.get("lineups")
    if lineup_url:
        html = await _fetch_rotowire_page(lineup_url, f"rw_lineups:{sport_lower}")
        if html:
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text)
            inj_flags = re.findall(
                r'(\w[\w\s\.\'-]+(?:OUT|GTD|QUESTIONABLE|DOUBTFUL|PROBABLE|DNP))',
                text
            )
            if inj_flags:
                parts.append("\nLINEUP PAGE FLAGS (RotoWire):")
                for item in inj_flags[:20]:
                    parts.append(f"  - {item.strip()}")

    # --- API-SPORTS (game data + lineups where available) ---
    try:
        api_sports_data = await _fetch_api_sports(sport_lower)
        if api_sports_data:
            parts.append(f"\n{api_sports_data}")
    except Exception as e:
        print(f"API-Sports context fetch error: {e}")

    if not parts:
        parts.append("INJURY/LINEUP: No data sources returned results. Grade conservatively.")

    return "\n".join(parts)


def _build_matrix_section(sport):
    """Return weighted variable matrix instructions for AI prompt."""
    matrix = SPORT_MATRICES.get(sport.lower(), [])
    if not matrix:
        return ""
    total_weight = sum(w for _, w, _ in matrix)
    max_score = total_weight * 10
    lines = []
    lines.append(f"\n=== {sport.upper()} SCORING MATRIX (10 Variables, Weighted 1-10) ===")
    lines.append(f"Score each variable 1-10 for EACH game. Weight x Score = Weighted.")
    lines.append(f"Max possible: {max_score}. Composite = sum / {max_score} x 10.\n")
    for i, (key, weight, label) in enumerate(matrix, 1):
        lines.append(f"  {i}. {label} (weight: {weight})")
    lines.append("\nCOMPOSITE THRESHOLDS:")
    lines.append("  9.0-10.0 = BEST BET (load up, full unit)")
    lines.append("  7.5-8.9  = STRONG PLAY (A-/B+)")
    lines.append("  6.0-7.4  = MODERATE EDGE (B/B-)")
    lines.append("  4.5-5.9  = LEAN (C, small or pass)")
    lines.append("  Below 4.5 = NO EDGE (D/F, pass)")
    return "\n".join(lines)



PREFERRED_SHARP_BOOK = "draftkings"
FALLBACK_SHARP_BOOKS = ["fanduel", "betmgm", "caesars", "pointsbet"]


def _parse_sharpapi_events(rows, sport_label):
    """Aggregate SharpAPI flat rows into game dicts matching frontend format.

    SharpAPI returns one row per selection per book per event. We fetch from
    ALL sportsbooks, prefer DraftKings lines, and fill in missing markets
    from FanDuel/others. This handles cases like NHL where DraftKings only
    has totals but FanDuel has moneylines.
    """
    # Group rows by event_id AND sportsbook
    events = {}
    for r in rows:
        sel = r.get("selection_type", "")
        if sel not in ("home", "away", "over", "under"):
            continue
        eid = r["event_id"]
        book = r.get("sportsbook", "unknown")
        if eid not in events:
            events[eid] = {
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "event_start_time": r["event_start_time"],
                "books": {},
            }
        if book not in events[eid]["books"]:
            events[eid]["books"][book] = []
        events[eid]["books"][book].append(r)

    games = []
    for eid, ev in events.items():
        game = {
            "id": eid,
            "sport": sport_label,
            "away": ev["away_team"],
            "home": ev["home_team"],
            "time": ev["event_start_time"],
            "bookmaker": None,
            "markets": {},
            "lines_available": False,
            "fetched_at": _now_ts(),
        }

        # Track which book provided each market type
        ml_book = None
        spread_book = None
        total_book = None
        books_used = set()

        # Process books in priority order: preferred first, then fallbacks
        book_order = [PREFERRED_SHARP_BOOK] + FALLBACK_SHARP_BOOKS
        for b in ev["books"]:
            if b not in book_order:
                book_order.append(b)

        for book in book_order:
            if book not in ev["books"]:
                continue
            for r in ev["books"][book]:
                mkt = r["market_type"]
                sel = r["selection_type"]
                odds = r.get("odds_american", 0)
                line = r.get("line")

                if mkt == "moneyline":
                    # Only take ML from first book that has it
                    if ml_book is not None and ml_book != book:
                        continue
                    ml_book = book
                    books_used.add(book)
                    if sel == "away":
                        game["away_ml"] = odds
                    elif sel == "home":
                        game["home_ml"] = odds

                elif mkt == "point_spread":
                    if spread_book is not None and spread_book != book:
                        continue
                    spread_book = book
                    books_used.add(book)
                    if sel == "away":
                        game["away_spread"] = line or 0
                        game["away_spread_odds"] = odds
                    elif sel == "home":
                        game["home_spread"] = line or 0
                        game["home_spread_odds"] = odds

                elif mkt in ("total_points", "total_goals", "total_rounds"):
                    if total_book is not None and total_book != book:
                        continue
                    total_book = book
                    books_used.add(book)
                    if line is not None:
                        game["total"] = line
                    if sel == "over":
                        game["over_odds"] = odds
                    elif sel == "under":
                        game["under_odds"] = odds

        has_ml = ml_book is not None
        has_spread = spread_book is not None
        has_total = total_book is not None

        # Show which books contributed
        if len(books_used) > 1:
            game["bookmaker"] = " + ".join(sorted(books_used))
        elif books_used:
            game["bookmaker"] = next(iter(books_used))
        else:
            game["bookmaker"] = "unknown"

        game["lines_available"] = has_spread or has_total or has_ml
        game["lines_complete"] = has_spread and has_total and has_ml
        # NHL: ML + totals is complete (no spreads expected)
        if sport_label == "NHL" and has_ml and has_total:
            game["lines_complete"] = True
        # MMA/Boxing: ML-only is complete
        elif sport_label in ("MMA", "BOXING") and has_ml:
            game["lines_complete"] = True
        # Track line shifts for SharpAPI games too
        shifts = _track_opening_lines(game)
        if shifts:
            game["shifts"] = shifts
        games.append(game)

    return games


async def _fetch_sharpapi_odds(sport_lower, sport_label):
    """Fetch odds from SharpAPI for a given sport.

    Fetches from ALL sportsbooks so we get the best available data.
    The parser prefers DraftKings but will use FanDuel/others for
    markets DK doesn't carry (e.g. NHL moneylines).
    """
    league = SHARPAPI_LEAGUES.get(sport_lower)
    if not league:
        return []

    markets = SHARPAPI_MARKETS.get(sport_lower, "moneyline")
    url = f"{SHARPAPI_BASE}/odds"
    params = {
        "league": league,
        "market": markets,
        "limit": 500,
    }
    headers = {"X-API-Key": SHARPAPI_KEY}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code == 200:
                body = resp.json()
                rows = body.get("data", [])
                return _parse_sharpapi_events(rows, sport_label)
    except Exception:
        pass
    return []


def _parse_event(event, sport_label):
    """Parse a single event from The Odds API into clean format.

    Tries PREFERRED_BOOK first. If that book has no markets for this event,
    falls back through FALLBACK_BOOKS in order. Tracks which book was used.
    """
    game = {
        "id": event["id"],
        "sport": sport_label,
        "away": event["away_team"],
        "home": event["home_team"],
        "time": event["commence_time"],
        "bookmaker": None,
        "markets": {},
        "lines_available": False,
        "fetched_at": _now_ts(),
    }

    # Try preferred book first, then fallbacks
    book_order = [PREFERRED_BOOK] + FALLBACK_BOOKS
    bookmakers_data = {bk["key"]: bk for bk in event.get("bookmakers", [])}

    for book_key in book_order:
        bk = bookmakers_data.get(book_key)
        if not bk:
            continue
        markets = {}
        for market in bk.get("markets", []):
            markets[market["key"]] = market["outcomes"]
        if markets:
            game["bookmaker"] = book_key
            game["markets"] = markets
            break

    spreads = game["markets"].get("spreads", [])
    totals = game["markets"].get("totals", [])
    h2h = game["markets"].get("h2h", [])

    has_spread = False
    has_total = False
    has_ml = False

    if spreads:
        has_spread = True
        for s in spreads:
            if s["name"] == event["away_team"]:
                game["away_spread"] = s.get("point", 0)
                game["away_spread_odds"] = s.get("price", -110)
            elif s["name"] == event["home_team"]:
                game["home_spread"] = s.get("point", 0)
                game["home_spread_odds"] = s.get("price", -110)

    if totals:
        has_total = True
        game["total"] = totals[0].get("point", 0)
        for t in totals:
            if t["name"] == "Over":
                game["over_odds"] = t.get("price", -110)
            elif t["name"] == "Under":
                game["under_odds"] = t.get("price", -110)

    if h2h:
        has_ml = True
        for h in h2h:
            if h["name"] == event["away_team"]:
                game["away_ml"] = h.get("price", 0)
            elif h["name"] == event["home_team"]:
                game["home_ml"] = h.get("price", 0)

    game["lines_available"] = has_spread or has_total or has_ml
    # NHL/MMA/boxing: ML-only is considered "complete" for grading
    if sport_label in ("NHL", "MMA", "BOXING") and has_ml:
        game["lines_complete"] = True
    else:
        game["lines_complete"] = has_spread and has_total and has_ml

    # Track line shifts (opening vs current)
    shifts = _track_opening_lines(game)
    if shifts:
        game["shifts"] = shifts

    # Detect arbitrage across bookmakers
    arbs = _detect_arbitrage(event, sport_label)
    if arbs:
        game["arbs"] = arbs

    return game


SPORT_KEYS = {
    "nba": ["basketball_nba"],
    "wnba": ["basketball_wnba"],
    "ncaab": ["basketball_ncaab"],
    "ncaaf": ["americanfootball_ncaaf"],
    "nhl": ["icehockey_nhl"],
    "nfl": ["americanfootball_nfl"],
    "mlb": ["baseball_mlb"],
    "mma": ["mma_mixed_martial_arts"],
    "boxing": ["boxing_boxing"],
    "tennis": [
        "tennis_atp_aus_open_singles",
        "tennis_atp_french_open",
        "tennis_atp_wimbledon",
        "tennis_atp_us_open",
        "tennis_atp_indian_wells",
        "tennis_atp_miami_open",
        "tennis_atp_italian_open",
        "tennis_atp_madrid_open",
        "tennis_atp_cincinnati_open",
        "tennis_atp_shanghai_masters",
        "tennis_atp_canadian_open",
        "tennis_atp_monte_carlo_masters",
        "tennis_wta_aus_open_singles",
        "tennis_wta_french_open",
        "tennis_wta_wimbledon",
        "tennis_wta_us_open",
        "tennis_wta_indian_wells",
        "tennis_wta_miami_open",
        "tennis_wta_madrid_open",
        "tennis_wta_cincinnati_open",
        "tennis_wta_canadian_open",
    ],
    "soccer": [
        "soccer_usa_mls",
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_uefa_champs_league",
        "soccer_mexico_ligamx",
    ],
}


_active_tennis_cache = {"keys": [], "fetched_at": 0}


async def _get_active_tennis_keys():
    """Fetch currently active tennis tournament keys from The Odds API. Cached 1 hour."""
    now = time.time()
    if _active_tennis_cache["keys"] and now - _active_tennis_cache["fetched_at"] < 3600:
        return _active_tennis_cache["keys"]
    try:
        url = f"{ODDS_API_BASE}/"
        params = {"apiKey": ODDS_API_KEY}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            if r.status_code == 200:
                sports = r.json()
                keys = [s["key"] for s in sports if s["key"].startswith("tennis_") and s.get("active")]
                _active_tennis_cache["keys"] = keys
                _active_tennis_cache["fetched_at"] = now
                return keys
    except Exception:
        pass
    return SPORT_KEYS.get("tennis", [])  # fallback to hardcoded list


async def _fetch_sport_odds(sport_key, markets, sport_label):
    """Fetch odds for a single sport key from The Odds API.

    Fetches from ALL available bookmakers (no bookmaker filter in request)
    so we always get games even if preferred book hasn't posted yet.
    """
    url = f"{ODDS_API_BASE}/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": markets,
        "oddsFormat": "american",
    }
    games = []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                for event in resp.json():
                    games.append(_parse_event(event, sport_label))
    except Exception:
        pass
    return games


@app.get("/api/odds/{sport}")
async def get_odds(sport: str, markets: str = "h2h,spreads,totals"):
    """Fetch live odds — SharpAPI primary, The Odds API fallback."""
    sport_lower = sport.lower()
    cache_key = f"{sport_lower}:{markets}"
    cached = _get_cached(cache_key)
    if cached:
        return JSONResponse(cached)

    label = sport.upper()
    all_games = []
    source_name = ""

    # --- PRIMARY: The Odds API ---
    if ODDS_API_KEY:
        if sport_lower == "tennis":
            keys = await _get_active_tennis_keys()
        else:
            keys = SPORT_KEYS.get(sport_lower, [sport_lower])
        for key in keys:
            games = await _fetch_sport_odds(key, markets, label)
            all_games.extend(games)
        if all_games:
            source_name = "The Odds API"

    # --- FALLBACK: SharpAPI ---
    if not all_games and SHARPAPI_KEY and sport_lower in SHARPAPI_LEAGUES:
        all_games = await _fetch_sharpapi_odds(sport_lower, label)
        if all_games:
            source_name = "SharpAPI (DraftKings)"

    if not all_games and not SHARPAPI_KEY and not ODDS_API_KEY:
        return JSONResponse({"error": "No odds API configured (set SHARPAPI_KEY or ODDS_API_KEY)"}, status_code=500)

    # Count games with complete vs incomplete lines
    complete = sum(1 for g in all_games if g.get("lines_complete"))
    incomplete = sum(1 for g in all_games if g.get("lines_available") and not g.get("lines_complete"))
    no_lines = sum(1 for g in all_games if not g.get("lines_available"))

    # Track which books are sourcing data
    books_used = list(set(g.get("bookmaker") for g in all_games if g.get("bookmaker")))

    result = {
        "sport": label,
        "games": all_games,
        "count": len(all_games),
        "lines_complete": complete,
        "lines_incomplete": incomplete,
        "no_lines": no_lines,
        "books_used": books_used,
        "source": f"{source_name} | Active: {', '.join(books_used) if books_used else 'none'}",
        "fetched_at": _now_ts(),
        "cached": False,
    }

    _set_cache(cache_key, result)
    return JSONResponse(result)


@app.get("/api/slate")
async def get_slate():
    """Fetch full slate — all sports combined."""
    if not ODDS_API_KEY and not SHARPAPI_KEY:
        return JSONResponse({"error": "No odds API configured"}, status_code=500)

    all_games = []
    for sport in ["nba", "wnba", "ncaab", "ncaaf", "nhl", "mlb", "tennis", "soccer", "mma", "boxing"]:
        resp = await get_odds(sport)
        if hasattr(resp, 'body'):
            data = json.loads(resp.body)
            if "games" in data:
                all_games.extend(data["games"])

    return JSONResponse({
        "games": all_games,
        "count": len(all_games),
        "source": "SharpAPI + The Odds API (multi-source)",
        "fetched_at": _now_ts(),
    })


# ===== PLAYER PROPS — Real Lines from The Odds API =====
PROP_MARKETS = {
    "nba": "player_points,player_rebounds,player_assists,player_threes",
    "nhl": "player_points,player_assists,player_goals",
    "nfl": "player_pass_yds,player_rush_yds,player_reception_yds,player_pass_tds",
    "mlb": "pitcher_strikeouts,batter_total_bases,batter_hits",
    "soccer": "player_goals,player_assists,player_shots",
    "mma": "fighter_wins_by_ko,fighter_wins_by_submission",
    "boxing": "fighter_wins_by_ko,fighter_wins_by_decision",
}
PROPS_CACHE_TTL = 300  # 5 min


def _calc_implied_prob(american_odds):
    """Convert American odds to implied probability %."""
    try:
        odds = float(american_odds)
        if odds < 0:
            return round(abs(odds) / (abs(odds) + 100) * 100, 1)
        else:
            return round(100 / (odds + 100) * 100, 1)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def _calc_edge(best_prob, consensus_prob):
    """Edge = consensus implied prob - best book implied prob.
    Positive = you're getting a better price than the market average."""
    if best_prob and consensus_prob:
        return round(consensus_prob - best_prob, 1)
    return 0


@app.get("/api/props/{sport}")
async def get_player_props(sport: str):
    """Fetch real player prop lines from The Odds API with edge analysis."""
    sport_lower = sport.lower()
    if sport_lower not in PROP_MARKETS:
        return JSONResponse({"sport": sport.upper(), "props": [], "count": 0, "fetched_at": _now_ts(), "message": f"No prop markets configured for {sport}"})

    cache_key = f"props:{sport_lower}"
    cached = _get_cached(cache_key, ttl=PROPS_CACHE_TTL)
    if cached:
        return JSONResponse(cached)

    if not ODDS_API_KEY:
        return JSONResponse({"error": "No ODDS_API_KEY configured"}, status_code=500)

    sport_keys = SPORT_KEYS.get(sport_lower, [sport_lower])
    markets = PROP_MARKETS[sport_lower]
    all_props = []

    for sport_key in sport_keys:
        # First get events list
        events_url = f"{ODDS_API_BASE}/{sport_key}/events/"
        params = {"apiKey": ODDS_API_KEY}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                events_resp = await client.get(events_url, params=params)
                if events_resp.status_code != 200:
                    continue
                events = events_resp.json()

                # Fetch props for each event (limit to 6 games to conserve credits)
                for event in events[:6]:
                    event_id = event.get("id")
                    home = event.get("home_team", "")
                    away = event.get("away_team", "")
                    matchup = f"{away} @ {home}"
                    commence = event.get("commence_time", "")

                    props_url = f"{ODDS_API_BASE}/{sport_key}/events/{event_id}/odds"
                    props_params = {
                        "apiKey": ODDS_API_KEY,
                        "regions": REGIONS,
                        "markets": markets,
                        "oddsFormat": "american",
                    }
                    try:
                        props_resp = await client.get(props_url, params=props_params)
                        if props_resp.status_code != 200:
                            continue
                        event_data = props_resp.json()
                    except Exception:
                        continue

                    # Parse bookmaker props into unified format
                    props_by_player = {}  # key: "player|stat|over/under" -> list of {book, line, odds}
                    for bookmaker in event_data.get("bookmakers", []):
                        book_name = bookmaker.get("title", bookmaker.get("key", "Unknown"))
                        for market in bookmaker.get("markets", []):
                            market_key = market.get("key", "")
                            stat_label = market_key.replace("player_", "").replace("pitcher_", "").replace("batter_", "").replace("_", " ").title()
                            for outcome in market.get("outcomes", []):
                                player = outcome.get("description", "Unknown")
                                side = outcome.get("name", "Over")  # Over/Under
                                line = outcome.get("point", 0)
                                price = outcome.get("price", 0)
                                pk = f"{player}|{stat_label}|{side}|{line}"
                                if pk not in props_by_player:
                                    props_by_player[pk] = {
                                        "player": player,
                                        "stat": stat_label,
                                        "side": side,
                                        "line": line,
                                        "matchup": matchup,
                                        "commence": commence,
                                        "books": [],
                                    }
                                props_by_player[pk]["books"].append({
                                    "book": book_name,
                                    "odds": price,
                                    "implied_prob": _calc_implied_prob(price),
                                })

                    # Calculate edge for each prop
                    for pk, prop in props_by_player.items():
                        books = prop["books"]
                        if not books:
                            continue
                        # Best odds = highest for Over (least negative), lowest abs for Under
                        best = max(books, key=lambda b: b["odds"])
                        prop["best_book"] = best["book"]
                        prop["best_odds"] = best["odds"]
                        prop["best_prob"] = best["implied_prob"]
                        # Consensus = average implied prob across books
                        probs = [b["implied_prob"] for b in books if b["implied_prob"]]
                        prop["consensus_prob"] = round(sum(probs) / len(probs), 1) if probs else None
                        prop["edge"] = _calc_edge(prop["best_prob"], prop["consensus_prob"])
                        prop["book_count"] = len(books)
                        # Edge verdict
                        edge = prop["edge"]
                        if edge >= 3:
                            prop["verdict"] = "SHARP VALUE"
                        elif edge >= 1:
                            prop["verdict"] = "SLIGHT EDGE"
                        elif edge >= -1:
                            prop["verdict"] = "FAIR LINE"
                        else:
                            prop["verdict"] = "BAD NUMBER"

                        all_props.append(prop)

        except Exception as e:
            logger.error(f"Props fetch error for {sport_key}: {e}")
            continue

    # Sort: SHARP VALUE first, then by edge descending
    verdict_order = {"SHARP VALUE": 0, "SLIGHT EDGE": 1, "FAIR LINE": 2, "BAD NUMBER": 3}
    all_props.sort(key=lambda p: (verdict_order.get(p.get("verdict", ""), 9), -(p.get("edge", 0))))

    result = {
        "sport": sport.upper(),
        "props": all_props,
        "count": len(all_props),
        "games_scanned": min(6, len(all_props)),
        "fetched_at": _now_ts(),
        "cached": False,
    }

    _set_cache(cache_key, result)
    return JSONResponse(result)


@app.get("/api/credits")
async def get_credits():
    """Check remaining API credits for both odds sources."""
    result = {"checked_at": _now_ts()}

    # SharpAPI status
    if SHARPAPI_KEY:
        result["sharpapi"] = {"status": "configured", "tier": "free", "rate_limit": "12 req/min"}
    else:
        result["sharpapi"] = {"status": "not configured"}

    # The Odds API credits
    if ODDS_API_KEY:
        url = f"{ODDS_API_BASE}/"
        params = {"apiKey": ODDS_API_KEY}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, params=params)
                result["odds_api"] = {
                    "status": "configured",
                    "remaining": resp.headers.get("x-requests-remaining", "?"),
                    "used": resp.headers.get("x-requests-used", "?"),
                }
        except Exception as e:
            result["odds_api"] = {"status": "error", "error": str(e)}
    else:
        result["odds_api"] = {"status": "not configured"}

    result["primary"] = "The Odds API" if ODDS_API_KEY else "SharpAPI"
    return JSONResponse(result)


@app.get("/api/edge/{sport}")
async def get_edge(sport: str):
    """Alyssa's Edge — 100% Math. No AI. Pure Edge.

    Contrarian value analysis calculated from odds data:
    - Public fade: flag heavy favorites where ML implies >70% implied prob
    - Line value: identify spreads that look off
    - Upset score: underdog value rating 1-10
    - Sharp signals: when line movement doesn't match public direction
    """
    sport_lower = sport.lower()
    odds_cache_key = f"{sport_lower}:h2h,spreads,totals"
    odds_data = _get_cached(odds_cache_key)
    if not odds_data:
        odds_resp = await get_odds(sport)
        if hasattr(odds_resp, 'body'):
            odds_data = json.loads(odds_resp.body)
        else:
            odds_data = {"games": [], "count": 0}

    games = odds_data.get("games", [])
    if not games:
        return JSONResponse({"sport": sport.upper(), "games": [], "generated_at": _now_ts()})

    edge_games = []
    for g in games:
        away = g.get("away", "?")
        home = g.get("home", "?")
        away_ml = g.get("away_ml")
        home_ml = g.get("home_ml")
        home_spread = g.get("home_spread")
        away_spread = g.get("away_spread")
        total = g.get("total")
        game_time = g.get("time", "")

        if not away_ml or not home_ml:
            continue

        # Convert ML to implied probability
        def ml_to_prob(ml):
            ml = float(ml)
            if ml > 0:
                return 100 / (ml + 100)
            else:
                return abs(ml) / (abs(ml) + 100)

        away_prob = ml_to_prob(away_ml)
        home_prob = ml_to_prob(home_ml)

        # Normalize so they sum to ~1 (remove vig)
        total_prob = away_prob + home_prob
        away_true = away_prob / total_prob
        home_true = home_prob / total_prob
        vig = (total_prob - 1) * 100  # vig percentage

        # Determine favorite/underdog
        if home_true > away_true:
            fav, dog = home, away
            fav_prob, dog_prob = home_true, away_true
            fav_ml, dog_ml = home_ml, away_ml
            fav_spread = home_spread
        else:
            fav, dog = away, home
            fav_prob, dog_prob = away_true, home_true
            fav_ml, dog_ml = away_ml, home_ml
            fav_spread = away_spread

        # --- UPSET SCORE (1-10) ---
        # Based on dog implied probability. Higher prob = higher upset score
        upset_score = min(10, max(1, round(dog_prob * 20)))

        # --- PUBLIC FADE FLAG ---
        # If favorite has >72% implied prob, the public is heavy on them
        public_fade = fav_prob > 0.72

        # --- VALUE RATING ---
        # Dogs with >30% true prob are value plays
        # Dogs with >40% are strong value
        value_tag = ""
        if dog_prob >= 0.42:
            value_tag = "STRONG VALUE"
        elif dog_prob >= 0.35:
            value_tag = "VALUE"
        elif dog_prob >= 0.28:
            value_tag = "SLIGHT VALUE"

        # --- SPREAD vs ML DISCREPANCY ---
        # If spread is tight but ML is wide, there's a signal
        spread_ml_flag = ""
        if fav_spread is not None and fav_ml is not None:
            try:
                spread_val = abs(float(fav_spread))
                ml_val = abs(float(fav_ml))
                # Tight spread (<4) but heavy ML (>200) = trap game potential
                if spread_val < 4 and ml_val > 180:
                    spread_ml_flag = "TRAP ALERT: Tight spread but heavy ML"
                # Big spread (>8) but soft ML (<250) = blowout not priced in
                elif spread_val > 8 and ml_val < 250:
                    spread_ml_flag = "BLOWOUT SIGNAL: Big spread, soft ML"
            except (ValueError, TypeError):
                pass

        # --- TOTAL LEAN ---
        total_lean = ""
        if total:
            try:
                t = float(total)
                over_odds = g.get("over_odds")
                under_odds = g.get("under_odds")
                if over_odds and under_odds:
                    over_prob = ml_to_prob(over_odds)
                    under_prob = ml_to_prob(under_odds)
                    total_p = over_prob + under_prob
                    if over_prob / total_p > 0.54:
                        total_lean = f"LEAN OVER {t}"
                    elif under_prob / total_p > 0.54:
                        total_lean = f"LEAN UNDER {t}"
            except (ValueError, TypeError):
                pass

        edge_game = {
            "away": away,
            "home": home,
            "time": game_time,
            "away_ml": away_ml,
            "home_ml": home_ml,
            "home_spread": home_spread,
            "total": total,
            "favorite": fav,
            "underdog": dog,
            "fav_prob": round(fav_prob * 100, 1),
            "dog_prob": round(dog_prob * 100, 1),
            "vig": round(vig, 1),
            "upset_score": upset_score,
            "public_fade": public_fade,
            "value_tag": value_tag,
            "spread_ml_flag": spread_ml_flag,
            "total_lean": total_lean,
            "dog_ml": dog_ml,
        }
        edge_games.append(edge_game)

    # Sort by upset score descending (best upset value first)
    edge_games.sort(key=lambda x: x["upset_score"], reverse=True)

    return JSONResponse({
        "sport": sport.upper(),
        "games": edge_games,
        "count": len(edge_games),
        "generated_at": _now_ts(),
        "tag": "100% Math. No AI. Pure Edge.",
    })


@app.get("/api/analysis/{sport}")
async def get_analysis(sport: str):
    """Generate AI analysis for a sport based on current live odds.

    RULE: Do NOT grade any game without complete line data.
    Games missing spread, total, or ML get grade "INCOMPLETE".
    """
    if not AZURE_ENDPOINT or not AZURE_KEY:
        return JSONResponse({"error": "Azure OpenAI not configured"}, status_code=500)

    sport_lower = sport.lower()
    cache_key = f"analysis:{sport_lower}"
    cached = _get_cached(cache_key, ttl=ANALYSIS_CACHE_TTL)
    if cached:
        return JSONResponse(cached)

    # Get current odds for context
    odds_cache_key = f"{sport_lower}:h2h,spreads,totals"
    odds_data = _get_cached(odds_cache_key)
    if not odds_data:
        # Fetch fresh odds
        odds_resp = await get_odds(sport)
        if hasattr(odds_resp, 'body'):
            odds_data = json.loads(odds_resp.body)
        else:
            odds_data = {"games": [], "count": 0}

    if not odds_data.get("games"):
        return JSONResponse({
            "sport": sport.upper(),
            "gotcha": "No games on the slate right now.",
            "games": [],
            "generated_at": _now_ts(),
        })

    # Separate complete vs incomplete games
    complete_games = []
    incomplete_games = []
    for g in odds_data["games"]:
        away = g.get("away", "?")
        home = g.get("home", "?")
        spread = g.get("home_spread", "?")
        total = g.get("total", "?")
        away_ml = g.get("away_ml", "?")
        home_ml = g.get("home_ml", "?")
        book = g.get("bookmaker", "unknown")
        game_time_raw = g.get("time", "?")
        # Convert UTC ISO time to PST for display
        try:
            gt = datetime.fromisoformat(game_time_raw.replace("Z", "+00:00")).astimezone(PST)
            game_time = gt.strftime("%I:%M %p PST")
        except Exception:
            game_time = game_time_raw

        line_str = f"{away} @ {home} | Spread: {home} {spread} | Total: {total} | ML: {away} ({away_ml}) / {home} ({home_ml}) | Book: {book} | Time: {game_time}"

        if g.get("lines_complete"):
            complete_games.append(line_str)
        else:
            missing = []
            if spread == "?" or not g.get("home_spread"):
                missing.append("spread")
            if total == "?" or not g.get("total"):
                missing.append("total")
            if away_ml == "?" or not g.get("away_ml"):
                missing.append("ML")
            incomplete_games.append({
                "line": line_str,
                "missing": missing,
                "matchup": f"{away} @ {home}",
            })

    games_text = "\n".join(complete_games)
    today = datetime.now(PST).strftime("%B %d, %Y")
    now_time = datetime.now(PST).strftime("%I:%M %p PST")

    # Build incomplete games note for prompt
    incomplete_note = ""
    if incomplete_games:
        incomplete_note = "\n\nINCOMPLETE GAMES (DO NOT GRADE — missing data):\n"
        for ig in incomplete_games:
            incomplete_note += f"- {ig['matchup']} — MISSING: {', '.join(ig['missing'])}\n"

    # ===== FETCH REAL DATA: Lineups + Injuries from RotoWire =====
    injury_context = ""
    try:
        injury_context = await _get_lineup_and_injury_context(sport_lower)
    except Exception as e:
        injury_context = f"INJURY DATA FETCH FAILED: {e}. Grade conservatively."
        print(f"Injury context fetch error for {sport_lower}: {e}")

    # ===== FETCH ROSTER CONTEXT (verify who's on which team) =====
    roster_context = ""
    try:
        roster_cache = _get_cached(f"rosters:{sport_lower}", ttl=86400)
        if roster_cache:
            roster_lines = [f"\n=== CURRENT ROSTERS (ESPN - verified) ==="]
            rosters = roster_cache.get("rosters", {})
            # Only include teams playing today
            today_teams = set()
            for g in odds_data.get("games", []):
                today_teams.add(g.get("away", ""))
                today_teams.add(g.get("home", ""))
            for abbr, info in rosters.items():
                team_full = info.get("team", "")
                if team_full in today_teams or abbr in [_normalize_team(t).split()[0].upper() for t in today_teams]:
                    top_players = [p["name"] for p in info.get("players", [])[:8]]
                    roster_lines.append(f"{abbr} ({team_full}): {', '.join(top_players)}")
            if len(roster_lines) > 1:
                roster_context = "\n".join(roster_lines)
    except Exception as e:
        print(f"Roster context error: {e}")

    # ===== BUILD WEIGHTED MATRIX SECTION =====
    matrix_section = _build_matrix_section(sport_lower)

    # ===== BATCH ANALYSIS — split games into groups of 4 for parallel Azure calls =====
    BATCH_SIZE = 4
    MAX_BATCHES = 6  # Cap at 6 batches (24 games) to avoid excessive Azure calls
    MAX_GAMES = BATCH_SIZE * MAX_BATCHES

    def _build_batch_prompt(batch_games_text, batch_incomplete_note, is_first_batch):
        """Build the Azure prompt for a batch of games."""
        gotcha_instruction = ""
        if is_first_batch:
            gotcha_instruction = f'"gotcha": "HTML unordered list (<ul><li>...</li></ul>). 4-8 bullet points covering: KEY INJURIES affecting tonight\'s lines (star players OUT/GTD), B2B/rest flags, line movement alerts, traps to avoid, weather (outdoor), sharp money indicators. Bold critical items with <strong>. End with: <li><em>Analysis generated {now_time} | Data: The Odds API + CBS Sports + RotoWire</em></li>",'
        else:
            gotcha_instruction = '"gotcha": "",'

        return f"""You are Edge Finder v2, a sharp sports betting analyst. You have access to REAL injury data and a weighted scoring matrix.

CREW: Peter (heavy/value/sharp, sizes up on conviction), Chinny (props/NHL/soccer master), Jimmy (new, learning), Sinton.ia (card builder/grader).
RULES: "Why is the market wrong?" = required for every grade. No answer = NO BET (grade D/F). Valid edges: news not priced in, public overreaction, rest/schedule, matchup-specific, sharp vs public, situational. Invalid: "better team", "should win", "volume play".

Today's {sport.upper()} slate - {today} (pulled at {now_time}):

=== ODDS DATA (THIS BATCH) ===
{batch_games_text}
{batch_incomplete_note}

=== INJURY & LINEUP INTELLIGENCE (CBS Sports + RotoWire + API-Sports) ===
{injury_context}

{roster_context}

{matrix_section}

=== EDGE QUESTION (MANDATORY for every game) ===
Before grading: "Why is the market wrong here?" If you cannot answer, grade D or F.

Generate analysis in this EXACT JSON format:
{{
  {gotcha_instruction}
  "games": [
    {{
      "matchup": "AWAY @ HOME",
      "composite_score": 7.2,
      "grade": "A+/A/A-/B+/B/B-/C+/C/D/F/INCOMPLETE",
      "edge_question": "Why is the market wrong? 1-2 sentences. If no answer: 'No clear edge identified.'",
      "tags": ["B2B", "SHARP", "TRAP", "UPSET", "PASS", "BEST BET", "REST-EDGE", "INJURY-IMPACT", "INCOMPLETE"],
      "matrix_scores": {{
        "var1_name": {{"score": 7, "weight": 10, "weighted": 70, "note": "why this score"}},
        "var2_name": {{"score": 5, "weight": 9, "weighted": 45, "note": "why this score"}}
      }},
      "injury_impact": "Which key players are OUT/GTD and how it changes the line. Be specific.",
      "rest_schedule": "B2B? Days rest for each team? Travel? 3-in-5?",
      "edge_summary": "One line: the edge in plain English",
      "peter_zone": "2-3 sentences. Peter's play: conviction level, sizing suggestion (full unit / half / small / pass), line value assessment.",
      "trends": ["ATS trend", "O/U trend", "H2H trend"],
      "flags": ["injury flag 1", "schedule flag", "sharp money flag"],
      "chinny_props": [
        {{"player": "Player Name", "prop": "Over 24.5 Points", "line": "-115", "grade": "A/B+/B/C", "edge": "why this prop hits"}}
      ],
      "data_status": "COMPLETE or INCOMPLETE - state exactly what is missing",
      "book_source": "which sportsbook"
    }}
  ]
}}

GRADING RULES:
- Use the SCORING MATRIX above. Fill in matrix_scores for each game with ALL variables scored 1-10.
- composite_score = (sum of all weighted scores) / max_possible * 10. Round to 1 decimal.
- Map composite to grade using the thresholds above.
- For NHL: totals-only or ML-only ARE gradeable. Only INCOMPLETE if NO data at all.
- For MMA/Boxing: ML-only IS gradeable.
- For NBA/Soccer: missing spread, total, OR ML = INCOMPLETE.
- injury_impact is MANDATORY. Check the RotoWire data above. Name specific players.
- If RotoWire data is unavailable, flag it: "Injury data not confirmed - grade with caution."
- rest_schedule is MANDATORY. Check game times for B2B detection.
- Chinny props: top 3-5 per game, B+ grade minimum. Skip for INCOMPLETE. Each prop MUST have player name, prop line, odds estimate, individual grade (A/B+/B/C), and 1-sentence edge explanation.
- PASS games get grade D or F with explicit reason.
- Be brutally honest. C means marginal. D means no edge. F means trap.

Return ONLY valid JSON. No markdown. No explanation."""

    def _call_azure_batch(prompt_text):
        """Synchronous Azure call for one batch. Runs in thread via asyncio."""
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_KEY,
            api_version="2024-10-21",
        )
        response = client.chat.completions.create(
            model=AZURE_MODEL,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7,
            max_tokens=6000,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
        return json.loads(raw)

    try:
        # Split complete games into batches of BATCH_SIZE, capped at MAX_BATCHES
        games_to_analyze = complete_games[:MAX_GAMES]  # cap total games
        game_batches = []
        for i in range(0, len(games_to_analyze), BATCH_SIZE):
            game_batches.append(games_to_analyze[i:i + BATCH_SIZE])

        if not game_batches:
            game_batches = [[]]  # still need gotcha even with 0 complete games

        skipped = len(complete_games) - len(games_to_analyze)
        logger.info(f"Analysis {sport}: {len(complete_games)} games -> {len(game_batches)} batches of {BATCH_SIZE} (skipped {skipped})")

        # Build prompts for each batch
        batch_prompts = []
        for idx, batch in enumerate(game_batches):
            batch_text = "\n".join(batch) if batch else "(no complete games in this batch)"
            # Only include incomplete note in first batch
            batch_inc = incomplete_note if idx == 0 else ""
            batch_prompts.append(_build_batch_prompt(batch_text, batch_inc, is_first_batch=(idx == 0)))

        # Run all batches in parallel via asyncio threads
        tasks = [asyncio.to_thread(_call_azure_batch, p) for p in batch_prompts]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results: first batch provides gotcha, all provide games
        all_analyzed_games = []
        gotcha_html = ""
        for idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Analysis batch {idx} failed: {result}")
                continue
            if idx == 0 and result.get("gotcha"):
                gotcha_html = result["gotcha"]
            if result.get("games"):
                all_analyzed_games.extend(result["games"])

        analysis = {
            "gotcha": gotcha_html or "Analysis partially generated. Some batches may have failed.",
            "games": all_analyzed_games,
            "sport": sport.upper(),
            "generated_at": _now_ts(),
            "source": "Azure OpenAI (Edge Finder)",
            "books_used": odds_data.get("books_used", []),
            "games_complete": len(complete_games),
            "games_incomplete": len(incomplete_games),
            "games_analyzed": len(all_analyzed_games),
            "batches": len(game_batches),
            "injury_source": "CBS Sports + RotoWire + API-Sports" if injury_context else "none",
            "injury_data_length": len(injury_context),
            "matrix": sport_lower in SPORT_MATRICES,
        }

        _set_cache(cache_key, analysis)
        return JSONResponse(analysis)

    except json.JSONDecodeError:
        return JSONResponse({
            "sport": sport.upper(),
            "gotcha": "Analysis generation returned invalid format. Refresh to retry.",
            "games": [],
            "generated_at": _now_ts(),
        })
    except Exception as e:
        return JSONResponse(
            {"error": f"Analysis generation failed: {str(e)}"},
            status_code=500,
        )


def _read_picks():
    """Read picks from data/picks.json."""
    try:
        with open(PICKS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"picks": [], "updated_at": None}


def _write_picks(data):
    """Write picks to data/picks.json."""
    os.makedirs(os.path.dirname(PICKS_FILE), exist_ok=True)
    with open(PICKS_FILE, "w") as f:
        json.dump(data, f, indent=2)


@app.get("/api/picks")
async def get_picks(date: str = ""):
    """Return picks, optionally filtered by date (YYYY-MM-DD or 'today')."""
    if date == "today":
        date = datetime.now(PST).strftime("%Y-%m-%d")

    if sb:
        query = sb.table("picks").select("*").order("created_at", desc=True)
        if date:
            query = query.eq("date", date)
        res = query.execute()
        return JSONResponse({"picks": res.data, "count": len(res.data)})

    data = _read_picks()
    picks = data.get("picks", [])
    if date:
        picks = [p for p in picks if p.get("date", "").startswith(date)]
    return JSONResponse({"picks": picks, "count": len(picks)})


@app.get("/api/picks/unique")
async def get_unique_picks():
    """Return scored unique/contrarian picks from graded history."""
    if sb:
        res = sb.table("picks").select("*").eq("result", "W").execute()
        wins = res.data or []
    else:
        data = _read_picks()
        wins = [p for p in data.get("picks", []) if p.get("result") == "W"]

    graded_results = [{"pick_id": p.get("id"), "result": "W"} for p in wins]
    unique = _score_unique_picks(graded_results, wins)
    return JSONResponse({"unique_picks": unique, "count": len(unique)})


@app.post("/api/picks")
async def save_pick(request: Request):
    """Save a new pick from the bet slip popup."""
    body = await request.json()
    crew = _get_crew(request)
    name = crew["display_name"] if crew else body.get("name", "")
    matchup = body.get("matchup", "")
    selection = body.get("selection", "")

    if not name or not matchup or not selection:
        return JSONResponse({"error": "name, matchup, and selection are required"}, status_code=400)

    pick = {
        "id": str(uuid.uuid4())[:8],
        "name": _sanitize(name),
        "sport": _sanitize(body.get("sport", "")),
        "type": _sanitize(body.get("type", "Spread")),
        "matchup": _sanitize(matchup),
        "selection": _sanitize(selection),
        "odds": _sanitize(body.get("odds", "-110")),
        "units": _sanitize(body.get("units", "1")),
        "confidence": _sanitize(body.get("confidence", "Lean")),
        "notes": _sanitize(body.get("notes", "")),
        "date": datetime.now(PST).strftime("%Y-%m-%d"),
        "time": datetime.now(PST).strftime("%I:%M %p"),
        "placed": False,
        "placed_at": None,
        "result": None,
        "graded_at": None,
        "created_at": datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S"),
    }

    if sb:
        sb.table("picks").insert(pick).execute()
    else:
        data = _read_picks()
        data["picks"].insert(0, pick)
        data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _write_picks(data)

    return JSONResponse({"status": "ok", "pick": pick})


@app.post("/api/picks/place")
async def place_pick(request: Request):
    """Mark a pick as placed on Hard Rock Bet."""
    body = await request.json()
    pick_id = body.get("id", "")
    placed = body.get("placed", True)

    if not pick_id:
        return JSONResponse({"error": "id is required"}, status_code=400)

    if sb:
        update_data = {
            "placed": placed,
            "placed_at": datetime.now().isoformat() if placed else None,
        }
        res = sb.table("picks").update(update_data).eq("id", pick_id).execute()
        if not res.data:
            return JSONResponse({"error": "Pick not found"}, status_code=404)
        return JSONResponse({"status": "ok", "pick": res.data[0]})

    data = _read_picks()
    for pick in data["picks"]:
        if pick["id"] == pick_id:
            pick["placed"] = placed
            pick["placed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if placed else None
            data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _write_picks(data)
            return JSONResponse({"status": "ok", "pick": pick})

    return JSONResponse({"error": "Pick not found"}, status_code=404)


@app.post("/api/picks/grade")
async def grade_pick(request: Request):
    """Grade a pick W/L/P."""
    body = await request.json()
    pick_id = body.get("id", "")
    result_val = body.get("result", "")

    if not pick_id or result_val not in ("W", "L", "P"):
        return JSONResponse({"error": "id and result (W/L/P) required"}, status_code=400)

    if sb:
        res = sb.table("picks").update({
            "result": result_val,
            "graded_at": datetime.now().isoformat(),
        }).eq("id", pick_id).execute()
        if not res.data:
            return JSONResponse({"error": "Pick not found"}, status_code=404)
        return JSONResponse({"status": "ok", "pick": res.data[0]})

    data = _read_picks()
    for pick in data["picks"]:
        if pick["id"] == pick_id:
            pick["result"] = result_val
            pick["graded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _write_picks(data)
            return JSONResponse({"status": "ok", "pick": pick})

    return JSONResponse({"error": "Pick not found"}, status_code=404)


async def _fetch_scores(sport_key: str) -> list:
    """Fetch completed game scores from The Odds API. Cached 5 min."""
    if not ODDS_API_KEY:
        logger.warning("No ODDS_API_KEY configured for scores fetch")
        return []
    cache_key = f"scores:{sport_key}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached
    url = f"{ODDS_API_BASE}/{sport_key}/scores/"
    params = {"apiKey": ODDS_API_KEY, "daysFrom": 3}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params=params)
            logger.info(f"Scores fetch {sport_key}: status={r.status_code}")
            if r.status_code == 200:
                data = r.json()
                logger.info(f"Scores fetch {sport_key}: got {len(data)} games")
                _set_cache(cache_key, data)
                return data
            else:
                logger.warning(f"Scores fetch {sport_key}: HTTP {r.status_code}")
    except Exception as e:
        logger.warning(f"Scores fetch failed for {sport_key}: {type(e).__name__}: {e}")
    return []


def _normalize_team(name: str) -> str:
    """Normalize team name for fuzzy matching."""
    return name.lower().strip().replace(".", "").replace("-", " ")


# Common abbreviations → full team names for matching
_NBA_ABBREVS = {
    "lal": "lakers", "lac": "clippers", "clip": "clippers",
    "gsw": "warriors", "gs": "warriors", "bos": "celtics", "cel": "celtics",
    "nyk": "knicks", "ny": "knicks", "bkn": "nets", "phi": "76ers",
    "mil": "bucks", "chi": "bulls", "cle": "cavaliers", "det": "pistons",
    "ind": "pacers", "atl": "hawks", "mia": "heat", "orl": "magic",
    "was": "wizards", "tor": "raptors", "cha": "hornets", "clt": "hornets",
    "mem": "grizzlies", "no": "pelicans", "nop": "pelicans", "dal": "mavericks",
    "hou": "rockets", "sa": "spurs", "sas": "spurs", "okc": "thunder",
    "den": "nuggets", "min": "timberwolves", "por": "trail blazers",
    "uta": "jazz", "sac": "kings", "phx": "suns",
}
_WNBA_ABBREVS = {
    "nym": "liberty", "ny": "liberty", "lva": "aces", "lv": "aces",
    "sea": "storm", "chi": "sky", "min": "lynx", "con": "sun", "ct": "sun",
    "ind": "fever", "atl": "dream", "dal": "wings", "phx": "mercury",
    "was": "mystics", "la": "sparks", "las": "sparks",
}

_NHL_ABBREVS = {
    "bos": "bruins", "bru": "bruins", "buf": "sabres", "car": "hurricanes",
    "cbj": "blue jackets", "col": "avalanche", "dal": "stars",
    "edm": "oilers", "fla": "panthers", "lak": "kings",
    "min": "wild", "mtl": "canadiens", "njd": "devils",
    "nsh": "predators", "nyi": "islanders", "nyr": "rangers", "ott": "senators",
    "pit": "penguins", "sea": "kraken", "stl": "blues", "tb": "lightning",
    "tbl": "lightning", "van": "canucks", "vgk": "golden knights",
    "wpg": "jets", "wsh": "capitals", "chi": "blackhawks", "det": "red wings",
    "phi": "flyers", "tor": "maple leafs",
}

def _get_abbrevs_for_sport(sport: str) -> dict:
    """Return the correct abbreviation dict for the sport."""
    s = (sport or "").lower()
    if s in ("nhl", "hockey"):
        return _NHL_ABBREVS
    if s in ("wnba",):
        return _WNBA_ABBREVS
    return _NBA_ABBREVS

# Combined fallback for cases where sport isn't available
_TEAM_ABBREVS = {**_NBA_ABBREVS, **_WNBA_ABBREVS, **_NHL_ABBREVS}


def _expand_abbrevs(text: str, sport: str = "") -> str:
    """Expand common team abbreviations in text."""
    abbrevs = _get_abbrevs_for_sport(sport) if sport else _TEAM_ABBREVS
    words = text.lower().split()
    expanded = []
    for w in words:
        clean = w.strip("@().,-")
        if clean in abbrevs:
            expanded.append(abbrevs[clean])
        expanded.append(clean)
    return " ".join(expanded)


def _teams_match(pick_text: str, home: str, away: str) -> bool:
    """Check if a pick's matchup references this game."""
    pt = _expand_abbrevs(_normalize_team(pick_text))
    h = _normalize_team(home)
    a = _normalize_team(away)
    # Check if both team names (or significant parts) appear in the pick matchup
    h_parts = h.split()
    a_parts = a.split()
    h_match = any(p in pt for p in h_parts if len(p) > 3) or h in pt
    a_match = any(p in pt for p in a_parts if len(p) > 3) or a in pt
    return h_match and a_match


_PROP_STATS = {"points", "assists", "rebounds", "steals", "blocks", "threes",
               "turnovers", "pts", "ast", "reb", "stl", "blk", "3pm", "pra",
               "goals", "saves", "shots", "hits", "faceoffs", "toi",
               "strikeouts", "runs", "rbi", "hrs", "walks",
               "yards", "touchdowns", "completions", "receptions"}


def _is_prop(pick: dict) -> bool:
    """Detect if a pick is a player prop (not a game-level bet)."""
    if pick.get("type", "").lower() == "prop":
        return True
    sel = pick.get("selection", "").lower()
    return any(stat in sel for stat in _PROP_STATS)


def _grade_pick_against_score(pick: dict, game: dict) -> str | None:
    """Determine W/L/P for a pick given final scores. Returns 'W', 'L', 'P', or None."""
    # NEVER grade player props against game scores - needs a stats API
    if _is_prop(pick):
        return None

    scores = game.get("scores")
    if not scores or not game.get("completed"):
        return None

    # Parse scores
    score_map = {}
    for s in scores:
        score_map[_normalize_team(s["name"])] = int(s.get("score", 0) or 0)

    home = _normalize_team(game.get("home_team", ""))
    away = _normalize_team(game.get("away_team", ""))
    home_score = score_map.get(home, 0)
    away_score = score_map.get(away, 0)

    # Skip games where both scores are 0 — likely not actually played or data missing
    if home_score == 0 and away_score == 0:
        return None

    selection = pick.get("selection", "")
    pick_type = pick.get("type", "").lower()
    sel_lower = selection.lower().strip()
    # Expand abbreviations in selection for matching (e.g. "OKC ML" → "thunder okc ml")
    sel_expanded = _expand_abbrevs(sel_lower)

    # --- MONEYLINE ---
    if pick_type in ("moneyline", "ml", "money line") or "ml" in sel_lower:
        # Figure out which team was picked
        if any(p in sel_expanded for p in _normalize_team(game["home_team"]).split() if len(p) > 3):
            return "W" if home_score > away_score else ("P" if home_score == away_score else "L")
        elif any(p in sel_expanded for p in _normalize_team(game["away_team"]).split() if len(p) > 3):
            return "W" if away_score > home_score else ("P" if home_score == away_score else "L")

    # --- SPREAD ---
    if pick_type == "spread" or any(c in selection for c in ["+", "-"]):
        # Strip odds in parens like (-105) or (+110) before parsing spread
        sel_no_odds = re.sub(r'\([+-]?\d+\)', '', selection)
        spread_match = re.search(r'([+-]?\d+\.?\d*)', sel_no_odds)
        if spread_match:
            spread = float(spread_match.group(1))
            # Determine which team the spread applies to
            sel_before_num = _expand_abbrevs(sel_lower[:sel_lower.find(spread_match.group(1))].strip())
            picked_home = any(p in sel_before_num for p in home.split() if len(p) > 3)
            picked_away = any(p in sel_before_num for p in away.split() if len(p) > 3)

            if picked_home:
                adj = home_score + spread
                if adj > away_score: return "W"
                elif adj == away_score: return "P"
                else: return "L"
            elif picked_away:
                adj = away_score + spread
                if adj > home_score: return "W"
                elif adj == home_score: return "P"
                else: return "L"

    # --- OVER/UNDER ---
    if pick_type in ("over/under", "total", "over", "under") or "over" in sel_lower or "under" in sel_lower:
        total_match = re.search(r'(\d+\.?\d*)', selection)
        if total_match:
            line = float(total_match.group(1))
            actual_total = home_score + away_score
            is_over = "over" in sel_lower
            if actual_total == line: return "P"
            if is_over:
                return "W" if actual_total > line else "L"
            else:
                return "W" if actual_total < line else "L"

    return None


def _parse_parlay_legs(pick: dict) -> list:
    """Parse parlay legs from selection/notes text.

    Looks for patterns like:
      "Hornets ML + Leafs ML + OKC ML"
      "CHA +7.5, TOR ML, Over 215.5"
      "Leg 1: Hornets ML | Leg 2: Leafs ML"
    Returns list of dicts with {selection, type} per leg.
    """
    text = pick.get("selection", "")
    # Split on " / " (space-slash-space) or " | " or comma-space or "leg N:"
    # Do NOT split on "+" as it appears in spreads like "LAL +5"
    parts = re.split(r'\s*/\s*|\s*\|\s*|\s*,\s*|\s+and\s+|\s*leg\s*\d+\s*:\s*', text, flags=re.IGNORECASE)
    legs = []
    for part in parts:
        part = part.strip()
        if len(part) < 3:
            continue
        # Determine type from text
        p_lower = part.lower()
        if "over" in p_lower or "under" in p_lower:
            leg_type = "Over/Under"
        elif re.search(r'[+-]\d', part):
            leg_type = "Spread"
        elif "ml" in p_lower or "money" in p_lower:
            leg_type = "Moneyline"
        else:
            leg_type = "Moneyline"  # default for team-name-only legs
        legs.append({"selection": part, "type": leg_type})
    return legs


def _grade_parlay(pick: dict, completed_games: list) -> str | None:
    """Grade a parlay by grading each leg individually.

    All legs W = Parlay W
    Any leg L = Parlay L
    Mix of W and P (no L) = reduced parlay W
    Not all legs gradeable = None (wait)
    """
    legs = _parse_parlay_legs(pick)
    if not legs:
        return None

    leg_results = []
    for leg in legs:
        leg_pick = {
            "selection": leg["selection"],
            "type": leg["type"],
            "matchup": pick.get("matchup", ""),
        }
        graded = False
        for game in completed_games:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            # Try matching this leg to a game — only match on the leg's own selection,
            # NOT the full parlay matchup string (which contains all teams and causes false matches)
            if _teams_match(leg["selection"], home, away):
                result = _grade_pick_against_score(leg_pick, game)
                if result:
                    leg_results.append(result)
                    graded = True
                    break
        if not graded:
            leg_results.append(None)

    # If any leg ungraded, can't determine parlay result yet
    if None in leg_results:
        # But if any leg is already L, parlay is dead
        if "L" in leg_results:
            return "L"
        return None

    # All legs graded
    if "L" in leg_results:
        return "L"
    if all(r == "W" or r == "P" for r in leg_results):
        if all(r == "P" for r in leg_results):
            return "P"
        return "W"  # Mix of W and P = reduced parlay wins
    return None


def _pick_key(p):
    """Generate a dedup key from pick fields."""
    return f"{p.get('name','').lower().strip()}|{p.get('date','')}|{p.get('selection','').lower().strip()}|{p.get('matchup','').lower().strip()}"


def _deduplicate_picks(picks):
    """Remove duplicate picks, keeping the earliest created_at."""
    seen = set()
    unique = []
    for p in sorted(picks, key=lambda x: x.get("created_at", "")):
        key = _pick_key(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _score_unique_picks(graded_results: list, all_picks: list) -> list:
    """Score unique/contrarian picks that deserve special recognition.

    A pick is 'unique' if it:
    - Won on underdog odds (positive odds like +150, +200)
    - Won on a spread going against the favorite (taking points)
    - Had high confidence and won
    - Was flagged as high-edge by the model

    Returns list of unique picks with scores.
    """
    unique = []
    pick_map = {p.get("id", ""): p for p in all_picks if p.get("id")}

    for r in graded_results:
        if r.get("result") != "W":
            continue

        pick_id = r.get("pick_id", "")
        pick = pick_map.get(pick_id, {})
        if not pick:
            continue

        score = 0
        tags = []
        odds_str = pick.get("odds", "-110")
        confidence = pick.get("confidence", "").lower()

        # Underdog win — positive odds
        try:
            odds_val = int(str(odds_str).replace("+", ""))
            if odds_val > 0:
                score += min(odds_val // 50, 5)  # +150=3, +200=4, +300=5 (cap at 5)
                tags.append(f"dog +{odds_val}")
        except (ValueError, TypeError):
            pass

        # High confidence win
        if confidence in ("hammer", "max", "5u", "strong"):
            score += 3
            tags.append("high-conviction")
        elif confidence in ("confident", "3u", "4u"):
            score += 2
            tags.append("confident")

        # Spread underdog (taking + points and winning)
        selection = pick.get("selection", "")
        pick_type = pick.get("type", "").lower()
        if pick_type == "spread" and "+" in selection:
            spread_match = re.search(r'\+(\d+\.?\d*)', re.sub(r'\([+-]?\d+\)', '', selection))
            if spread_match:
                pts = float(spread_match.group(1))
                if pts >= 5:
                    score += 2
                    tags.append(f"spread dog +{pts}")
                elif pts >= 2:
                    score += 1
                    tags.append(f"spread dog +{pts}")

        if score > 0:
            unique.append({
                "pick_id": pick_id,
                "selection": pick.get("selection", ""),
                "matchup": pick.get("matchup", ""),
                "score": score,
                "tags": tags,
                "name": pick.get("name", ""),
            })

    unique.sort(key=lambda x: x["score"], reverse=True)
    return unique


@app.post("/api/picks/autograde")
async def autograde_picks(request: Request):
    """Auto-grade ungraded picks by fetching final scores from The Odds API.

    Accepts optional JSON body with {picks: [...]} from client localStorage.
    Falls back to server-side picks if none sent.
    """
    # Accept picks from client (localStorage) or fall back to server storage
    body = {}
    try:
        raw = await request.body()
        if raw and raw.strip():
            body = json.loads(raw)
    except Exception as e:
        logger.warning(f"Autograde body parse: {e}")

    client_picks = body.get("picks", []) if isinstance(body, dict) else []

    if client_picks:
        ungraded = [p for p in client_picks if not p.get("result")]
    elif sb:
        res = sb.table("picks").select("*").is_("result", "null").execute()
        ungraded = res.data or []
    else:
        data = _read_picks()
        ungraded = [p for p in data.get("picks", []) if not p.get("result")]

    if not ungraded:
        return JSONResponse({"status": "ok", "graded": 0, "message": "No ungraded picks"})

    # Deduplicate picks by key fields (name + date + selection + matchup)
    ungraded = _deduplicate_picks(ungraded)

    # Determine which sports need scores
    sports_needed = set()
    for p in ungraded:
        sport = p.get("sport", "").lower()
        if sport in SPORT_KEYS:
            sports_needed.add(sport)
        # Skip unknown sports instead of fetching everything

    if not sports_needed:
        sports_needed = {"nba", "wnba", "nhl"}  # fallback only if zero sports detected

    # Fetch scores for all sports IN PARALLEL (was sequential — very slow)
    fetch_tasks = []
    fetch_labels = []
    for sport in sports_needed:
        for key in SPORT_KEYS.get(sport, []):
            fetch_tasks.append(_fetch_scores(key))
            fetch_labels.append(f"{sport}/{key}")

    all_scores = []
    fetch_errors = []
    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    for label, result in zip(fetch_labels, results):
        if isinstance(result, Exception):
            fetch_errors.append(f"{label}: {result}")
        elif result:
            all_scores.extend(result)

    completed = [g for g in all_scores if g.get("completed")]

    # Debug info
    debug_parts = [
        f"{len(ungraded)} ungraded",
        f"{len(sports_needed)} sports ({', '.join(sports_needed)})",
        f"{len(all_scores)} games fetched",
        f"{len(completed)} completed",
    ]
    if fetch_errors:
        debug_parts.append(f"fetch errors: {', '.join(fetch_errors)}")

    if not completed:
        return JSONResponse({
            "status": "ok", "graded": 0,
            "message": f"No completed games found",
            "debug": " | ".join(debug_parts),
        })

    # Match and grade
    graded_count = 0
    results = []
    no_match = []
    for pick in ungraded:
        matchup = pick.get("matchup", "")
        selection = pick.get("selection", "")
        pick_type = pick.get("type", "").lower()
        matched = False

        # --- PARLAY GRADING ---
        if pick_type == "parlay" or "+" in selection and selection.count("+") >= 1 and "+" not in selection[:2]:
            result = _grade_parlay(pick, completed)
            if result:
                pick_id = pick.get("id", "")
                if sb and pick_id:
                    try:
                        sb.table("picks").update({
                            "result": result,
                            "graded_at": datetime.now(PST).isoformat(),
                        }).eq("id", pick_id).execute()
                    except Exception:
                        pass
                graded_count += 1
                legs = _parse_parlay_legs(pick)
                results.append({
                    "pick_id": pick_id,
                    "selection": selection,
                    "matchup": matchup,
                    "result": result,
                    "final_score": f"Parlay ({len(legs)} legs)",
                })
                matched = True
            if not matched:
                no_match.append(f"PARLAY: {selection[:60]}")
            continue

        # --- SINGLE PICK GRADING ---
        for game in completed:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            if not _teams_match(matchup, home, away):
                # Also try matching on selection text
                if not _teams_match(selection, home, away):
                    continue

            result = _grade_pick_against_score(pick, game)
            if result:
                pick_id = pick.get("id", "")
                # Update server-side storage if pick has ID
                if sb and pick_id:
                    try:
                        sb.table("picks").update({
                            "result": result,
                            "graded_at": datetime.now(PST).isoformat(),
                        }).eq("id", pick_id).execute()
                    except Exception:
                        pass

                graded_count += 1
                scores_info = game.get("scores", [])
                score_str = " - ".join([f"{s['name']} {s.get('score', '?')}" for s in scores_info])
                results.append({
                    "pick_id": pick_id,
                    "selection": selection,
                    "matchup": matchup,
                    "result": result,
                    "final_score": score_str,
                })
                matched = True
                break

        if not matched:
            no_match.append(f"{matchup} ({selection})")

    if no_match:
        debug_parts.append(f"no match: {', '.join(no_match[:5])}")

    # Also show sample completed games for debugging
    sample_games = [f"{g['away_team']} @ {g['home_team']}" for g in completed[:5]]
    debug_parts.append(f"sample games: {', '.join(sample_games)}")

    # Score unique picks — contrarian/underdog wins get bonus grades
    unique_picks = _score_unique_picks(results, ungraded)

    return JSONResponse({
        "status": "ok",
        "graded": graded_count,
        "total_ungraded": len(ungraded),
        "completed_games": len(completed),
        "results": results,
        "unique_picks": unique_picks,
        "debug": " | ".join(debug_parts),
    })


# ===== EDGE CREW AI AGENT (Chat) =====
AGENT_SYSTEM_PROMPT = """You are EV (Expected Value) — the Edge Crew's sharp AI betting analyst. You live for +EV plays.

Your personality: Direct, confident, data-driven. You talk like the sharpest guy at the window — no fluff, just edges. Sign off key calls with "That's +EV." when it fits.

You can help with:
1. **Lock picks** — When someone says "lock [team] [spread/ML/over/under] [odds] [units]", extract the pick details and return a JSON action.
2. **Show today's slate** — Summarize what games are available.
3. **Bankroll check** — Summarize the crew's record and stats.
4. **Grade picks** — Trigger auto-grading.
5. **Analysis** — Give quick takes on matchups, edges, and value.
6. **Pregame** — Prep for a specific game or customer.

IMPORTANT — When the user wants to lock a pick, respond with a JSON block:
```json
{"action":"lock_pick","sport":"nba","matchup":"Team A @ Team B","selection":"Team A -3.5","odds":"-110","units":"1","confidence":"Lean"}
```

When the user asks to grade picks, respond with:
```json
{"action":"autograde"}
```

For everything else, just respond conversationally. Keep responses under 150 words. Be the sharpest guy in the room."""


@app.post("/api/chat")
async def agent_chat(request: Request):
    """Edge Crew AI Agent — conversational pick locking and analysis."""
    if not AZURE_ENDPOINT or not AZURE_KEY:
        return JSONResponse({"error": "Azure OpenAI not configured"}, status_code=500)

    body = await request.json()
    user_msg = body.get("message", "")
    history = body.get("history", [])

    if not user_msg:
        return JSONResponse({"error": "message is required"}, status_code=400)

    # Build context: include today's picks summary and current slate info
    context_parts = []

    # Add picks context
    if sb:
        today = datetime.now(PST).strftime("%Y-%m-%d")
        res = sb.table("picks").select("*").order("created_at", desc=True).limit(30).execute()
        picks_data = res.data or []
    else:
        data = _read_picks()
        picks_data = data.get("picks", [])[:30]

    if picks_data:
        graded = [p for p in picks_data if p.get("result")]
        ungraded = [p for p in picks_data if not p.get("result")]
        w = sum(1 for p in graded if p["result"] == "W")
        l = sum(1 for p in graded if p["result"] == "L")
        push = sum(1 for p in graded if p["result"] == "P")
        context_parts.append(f"CREW STATUS: {len(picks_data)} total picks, {w}-{l}-{push} record, {len(ungraded)} ungraded.")
        recent = picks_data[:5]
        picks_str = "\n".join([f"- {p.get('name','?')}: {p.get('selection','')} ({p.get('matchup','')}) {p.get('odds','')} {p.get('units','1')}u [{p.get('result','PENDING')}]" for p in recent])
        context_parts.append(f"RECENT PICKS:\n{picks_str}")

    # Add cached slate info if available — include odds so agent can share real lines
    for sport in ["nba", "wnba", "ncaab", "ncaaf", "nhl", "mlb", "tennis", "soccer", "mma", "boxing"]:
        cache_key = f"{sport}:h2h,spreads,totals"
        cached = _get_cached(cache_key)
        if cached and cached.get("games"):
            games = cached["games"]
            game_count = len(games)
            lines = []
            for g in games[:8]:
                away = g.get("away", "?")
                home = g.get("home", "?")
                away_spread = g.get("away_spread", "")
                home_spread = g.get("home_spread", "")
                total = g.get("total", "")
                away_ml = g.get("away_ml", "")
                home_ml = g.get("home_ml", "")
                time_str = g.get("fetched_at", g.get("time", ""))
                line = f"{away} @ {home}"
                details = []
                if home_spread:
                    details.append(f"{home} {home_spread}")
                if total:
                    details.append(f"O/U {total}")
                if away_ml and home_ml:
                    details.append(f"ML: {away} {away_ml} / {home} {home_ml}")
                if details:
                    line += f" | {' | '.join(details)}"
                lines.append(line)
            context_parts.append(f"{sport.upper()} SLATE ({game_count} games):\n" + "\n".join(lines))

    context = "\n\n".join(context_parts) if context_parts else "No slate data loaded yet. Tell user to click a sport tab first to load odds."

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT + "\n\nCURRENT CONTEXT:\n" + context},
    ]
    # Add conversation history (last 10)
    for msg in history[-10:]:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": user_msg})

    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_MODEL}/chat/completions?api-version=2024-08-01-preview"
    headers = {"api-key": AZURE_KEY, "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json={
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.7,
            }, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            reply = data["choices"][0]["message"]["content"].strip()
            return JSONResponse({"response": reply})
    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        return JSONResponse({"response": "Connection dropped. Try again."}, status_code=200)


def _read_upsets():
    """Read upsets from data/upsets.json. Auto-migrates old format."""
    try:
        with open(UPSETS_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"picks": []}

    # Detect old format (has sport keys like "NBA": "text") and migrate
    if "picks" not in data:
        data = _migrate_upsets(data)
        _write_upsets(data)

    return data


def _write_upsets(data):
    """Write upsets to data/upsets.json."""
    os.makedirs(os.path.dirname(UPSETS_FILE), exist_ok=True)
    with open(UPSETS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _migrate_upsets(old_data):
    """Convert old {NBA: 'text', NHL: 'text'} → {picks: [...]}."""
    picks = []
    for sport in ["NBA", "NHL"]:
        text = old_data.get(sport, "")
        if not text:
            continue
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            team = parts[0] if len(parts) > 0 else ""
            odds = parts[1] if len(parts) > 1 else ""
            thesis = parts[2] if len(parts) > 2 else ""
            picks.append({
                "id": str(uuid.uuid4())[:8],
                "name": "Peter",
                "sport": sport,
                "team": team,
                "odds": odds,
                "thesis": thesis,
                "date": old_data.get("updated_at", datetime.now().strftime("%Y-%m-%d"))[:10],
                "created_at": old_data.get("updated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            })
    return {"picks": picks}


@app.put("/api/picks/{pick_id}")
async def update_pick(pick_id: str, request: Request):
    """Update a pick's fields (selection, odds, units, etc.)."""
    body = await request.json()
    allowed = {"selection", "odds", "units", "matchup", "notes", "confidence", "sport", "type"}
    update_data = {k: _sanitize(str(v)) for k, v in body.items() if k in allowed and v is not None}
    if not update_data:
        return JSONResponse({"error": "No valid fields to update"}, status_code=400)
    if sb:
        res = sb.table("picks").update(update_data).eq("id", pick_id).execute()
        if not res.data:
            return JSONResponse({"error": "Pick not found"}, status_code=404)
        return JSONResponse({"status": "ok", "pick": res.data[0]})
    else:
        data = _read_picks()
        for pick in data["picks"]:
            if pick["id"] == pick_id:
                pick.update(update_data)
                _write_picks(data)
                return JSONResponse({"status": "ok", "pick": pick})
        return JSONResponse({"error": "Pick not found"}, status_code=404)


@app.delete("/api/picks/{pick_id}")
async def delete_pick(pick_id: str):
    """Delete a pick by ID."""
    if sb:
        sb.table("picks").delete().eq("id", pick_id).execute()
    else:
        data = _read_picks()
        data["picks"] = [p for p in data["picks"] if p["id"] != pick_id]
        _write_picks(data)
    return JSONResponse({"status": "deleted", "id": pick_id})


@app.get("/api/upsets")
async def get_upsets(sport: str = "", date: str = ""):
    """Return upset picks, optionally filtered by sport and date."""
    if sb:
        filter_date = date if date and date != "today" else datetime.now().strftime("%Y-%m-%d")
        query = sb.table("upsets").select("*").eq("date", filter_date).order("created_at", desc=True)
        if sport:
            query = query.ilike("sport", sport)
        res = query.execute()
        return JSONResponse({"picks": res.data, "count": len(res.data)})

    data = _read_upsets()
    picks = data.get("picks", [])
    filter_date = date if date and date != "today" else datetime.now().strftime("%Y-%m-%d")
    picks = [p for p in picks if p.get("date", "") == filter_date]
    if sport:
        picks = [p for p in picks if p.get("sport", "").upper() == sport.upper()]
    return JSONResponse({"picks": picks, "count": len(picks)})


@app.post("/api/upsets")
async def save_upset(request: Request):
    """Save a new upset pick from any crew member."""
    body = await request.json()
    crew = _get_crew(request)
    name = crew["display_name"] if crew else body.get("name", "")
    team = body.get("team", "")
    odds = body.get("odds", "")

    if not name or not team or not odds:
        return JSONResponse({"error": "name, team, and odds are required"}, status_code=400)

    pick = {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "sport": body.get("sport", "").upper(),
        "team": team,
        "odds": odds,
        "thesis": body.get("thesis", ""),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "created_at": datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S"),
    }

    if sb:
        sb.table("upsets").insert(pick).execute()
    else:
        data = _read_upsets()
        data["picks"].insert(0, pick)
        _write_upsets(data)

    return JSONResponse({"status": "ok", "pick": pick})


@app.delete("/api/upsets/{pick_id}")
async def delete_upset(pick_id: str):
    """Delete an upset pick by ID."""
    if sb:
        res = sb.table("upsets").delete().eq("id", pick_id).execute()
        if not res.data:
            return JSONResponse({"error": "Pick not found"}, status_code=404)
        return JSONResponse({"status": "ok"})

    data = _read_upsets()
    original_len = len(data["picks"])
    data["picks"] = [p for p in data["picks"] if p.get("id") != pick_id]
    if len(data["picks"]) == original_len:
        return JSONResponse({"error": "Pick not found"}, status_code=404)
    _write_upsets(data)
    return JSONResponse({"status": "ok"})


@app.get("/api/bankroll")
async def get_bankroll():
    """Get shared bankroll settings."""
    if sb:
        res = sb.table("bankroll_settings").select("*").limit(1).execute()
        if res.data:
            row = res.data[0]
            return JSONResponse({
                "starting_balance": float(row.get("starting_balance", 1000)),
                "unit_size": float(row.get("unit_size", 25)),
                "updated_by": row.get("updated_by", ""),
            })
    return JSONResponse({"starting_balance": 1000, "unit_size": 25})


@app.post("/api/bankroll")
async def save_bankroll(request: Request):
    """Save shared bankroll settings."""
    body = await request.json()
    if sb:
        data = {
            "id": 1,
            "starting_balance": body.get("starting_balance", 1000),
            "unit_size": body.get("unit_size", 25),
            "updated_at": datetime.now().isoformat(),
            "updated_by": body.get("updated_by", ""),
        }
        sb.table("bankroll_settings").upsert(data).execute()
        return JSONResponse({"status": "ok", **data})
    return JSONResponse({"status": "ok", "note": "no database configured"})


@app.get("/api/gotcha")
async def get_gotcha():
    """Get gotcha notes (per sport)."""
    if sb:
        res = sb.table("gotcha_notes").select("*").execute()
        notes = {}
        for row in res.data:
            notes[row["sport"]] = {
                "notes": row.get("notes", ""),
                "updated_by": row.get("updated_by", ""),
            }
        return JSONResponse({"notes": notes})
    return JSONResponse({"notes": {}})


@app.post("/api/gotcha")
async def save_gotcha_notes(request: Request):
    """Save gotcha notes for a sport."""
    body = await request.json()
    sport = body.get("sport", "NBA")
    notes_text = body.get("notes", "")
    if sb:
        data = {
            "sport": sport,
            "notes": notes_text,
            "updated_at": datetime.now().isoformat(),
            "updated_by": body.get("updated_by", ""),
        }
        sb.table("gotcha_notes").upsert(data, on_conflict="sport").execute()
        return JSONResponse({"status": "ok", **data})
    return JSONResponse({"status": "ok", "note": "no database configured"})


@app.get("/sw.js")
async def serve_sw():
    return FileResponse("static/sw.js", media_type="application/javascript",
                        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"})

@app.get("/manifest.json")
async def serve_manifest():
    return FileResponse("static/manifest.json", media_type="application/json", headers={"Cache-Control": "no-cache"})

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/api/lineups/{sport}")
async def get_lineups(sport: str):
    """Fetch injury/lineup data from CBS Sports + RotoWire for frontend badges."""
    import re
    sport_lower = sport.lower()

    cache_key = f"frontend_injuries:{sport_lower}"
    cached = _get_cached(cache_key, ttl=1800)
    if cached:
        return JSONResponse(cached)

    injuries_by_team = {}  # team_name -> [{"player": ..., "status": ..., "injury": ..., "pos": ...}]

    # --- CBS SPORTS (primary - structured injury report) ---
    CBS_URLS = {
        "nba": "https://www.cbssports.com/nba/injuries/",
        "wnba": "https://www.cbssports.com/wnba/injuries/",
        "nhl": "https://www.cbssports.com/nhl/injuries/",
    }
    cbs_url = CBS_URLS.get(sport_lower)
    if cbs_url:
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                resp = await client.get(cbs_url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                })
                if resp.status_code == 200:
                    html = resp.text
                    # CBS uses team links like <a href="/nba/teams/CHI/">Chicago</a>
                    # Split by team position to get per-team sections
                    sport_path = sport_lower
                    team_pattern = rf'href="/{sport_path}/teams/([^"]+)/?\"[^>]*>([^<]+)</a>'
                    team_positions = [(m.start(), m.group(2).strip()) for m in re.finditer(team_pattern, html)]

                    for i, (pos, team_name) in enumerate(team_positions):
                        end_pos = team_positions[i + 1][0] if i + 1 < len(team_positions) else len(html)
                        section = html[pos:end_pos]

                        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', section, re.DOTALL)
                        for row in rows:
                            if 'CellPlayerName' not in row:
                                continue
                            name_match = re.findall(r'>([^<]+)</a>', row)
                            tds = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
                            cells = [re.sub(r'<[^>]+>', '', td).strip() for td in tds]
                            cells = [c for c in cells if c]

                            if name_match and len(cells) >= 4:
                                full_name = name_match[-1].strip() if len(name_match) > 1 else name_match[0].strip()
                                pos_str = cells[1] if len(cells) > 1 else "?"
                                injury_type = cells[3] if len(cells) > 3 else "?"
                                status = cells[4] if len(cells) > 4 else cells[-1]
                                # Normalize status
                                status_upper = status.strip().upper()
                                if "OUT" in status_upper or "DOUBTFUL" in status_upper:
                                    norm_status = "OUT"
                                elif any(k in status_upper for k in ("DAY", "GTD", "GAME TIME", "QUESTIONABLE", "PROBABLE")):
                                    norm_status = "GTD"
                                else:
                                    norm_status = status.strip()

                                if team_name not in injuries_by_team:
                                    injuries_by_team[team_name] = []
                                injuries_by_team[team_name].append({
                                    "player": full_name,
                                    "pos": pos_str,
                                    "injury": injury_type,
                                    "status": norm_status,
                                    "raw_status": status.strip(),
                                })
        except Exception as e:
            print(f"CBS lineups/injuries fetch error ({sport_lower}): {e}")

    result = {
        "sport": sport.upper(),
        "injuries_by_team": injuries_by_team,
        "team_count": len(injuries_by_team),
        "total_players": sum(len(v) for v in injuries_by_team.values()),
        "source": "CBS Sports" if injuries_by_team else "none",
        "fetched_at": _now_ts(),
    }
    if injuries_by_team:
        _set_cache(cache_key, result)
    return JSONResponse(result)


_ESPN_TEAM_IDS = {
    "nba": ["atl","bos","bkn","cha","chi","cle","dal","den","det","gs","hou","ind",
            "lac","lal","mem","mia","mil","min","no","ny","okc","orl","phi","phx",
            "por","sac","sa","tor","utah","wsh"],
    "wnba": ["atl","chi","con","dal","ind","lva","la","min","ny","phx","sea","wsh"],
    "nhl": ["ana","ari","bos","buf","car","cgy","chi","col","cbj","dal","det","edm",
            "fla","la","min","mtl","nsh","njd","nyi","nyr","ott","phi","pit","sjs",
            "sea","stl","tb","tor","utah","van","vgk","wpg","wsh"],
}

@app.get("/api/rosters/{sport}")
async def get_rosters(sport: str):
    """Fetch current team rosters from ESPN. Cached 24 hours."""
    sport_lower = sport.lower()
    cache_key = f"rosters:{sport_lower}"
    cached = _get_cached(cache_key, ttl=86400)
    if cached:
        return JSONResponse(cached)

    espn_sport = {"nba": "basketball/nba", "wnba": "basketball/wnba", "nhl": "hockey/nhl"}.get(sport_lower)
    team_ids = _ESPN_TEAM_IDS.get(sport_lower, [])
    if not espn_sport or not team_ids:
        return JSONResponse({"error": f"No roster source for {sport}"}, status_code=400)

    rosters = {}
    async with httpx.AsyncClient(timeout=15.0) as client:
        for tid in team_ids:
            try:
                resp = await client.get(
                    f"https://site.api.espn.com/apis/site/v2/sports/{espn_sport}/teams/{tid}/roster",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    team_name = data.get("team", {}).get("displayName", "?")
                    abbr = data.get("team", {}).get("abbreviation", tid.upper())
                    players = []
                    for a in data.get("athletes", []):
                        players.append({
                            "name": a.get("displayName", "?"),
                            "pos": a.get("position", {}).get("abbreviation", "?"),
                            "number": a.get("jersey", "?"),
                        })
                    rosters[abbr] = {"team": team_name, "players": players}
            except Exception:
                pass

    result = {
        "sport": sport.upper(),
        "rosters": rosters,
        "team_count": len(rosters),
        "source": "ESPN",
        "fetched_at": _now_ts(),
    }
    if rosters:
        _set_cache(cache_key, result)
    return JSONResponse(result)


@app.post("/api/cache/clear")
async def clear_server_cache():
    """Clear all server-side cached odds and analysis data."""
    _cache.clear()
    return JSONResponse({"status": "cleared", "message": "Server cache flushed"})


@app.get("/api/debug/injuries/{sport}")
async def debug_injuries(sport: str):
    """Debug endpoint: test CBS Sports injury fetch from this server."""
    import re
    sport_lower = sport.lower()
    CBS_URLS = {"nba": "https://www.cbssports.com/nba/injuries/", "wnba": "https://www.cbssports.com/wnba/injuries/", "nhl": "https://www.cbssports.com/nhl/injuries/"}
    url = CBS_URLS.get(sport_lower)
    if not url:
        return JSONResponse({"error": f"No CBS URL for {sport}"})
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            })
            html = resp.text
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
            player_rows = [r for r in rows if 'CellPlayerName' in r]
            sample = []
            for row in player_rows[:5]:
                names = re.findall(r'>([^<]+)</a>', row)
                sample.append(names[-1] if names else "?")
            return JSONResponse({
                "status_code": resp.status_code,
                "html_length": len(html),
                "total_rows": len(rows),
                "player_rows": len(player_rows),
                "sample_players": sample,
                "has_CellPlayerName": "CellPlayerName" in html,
                "snippet": html[max(0, html.find("CellPlayerName")-100):html.find("CellPlayerName")+200][:300] if "CellPlayerName" in html else html[:500],
            })
    except Exception as e:
        return JSONResponse({"error": str(e)})


# ── WebRTC Voice (Azure Realtime API) ──────────────────────────────

@app.post("/api/voice/token")
async def voice_token():
    """Get ephemeral WebRTC token from Azure Realtime API."""
    if not AZURE_BASE or not AZURE_KEY:
        return JSONResponse({"error": "Azure not configured"}, status_code=500)
    url = f"{AZURE_BASE}/openai/v1/realtime/client_secrets"
    headers = {"api-key": AZURE_KEY, "Content-Type": "application/json"}
    payload = {
        "session": {
            "type": "realtime",
            "model": REALTIME_DEPLOYMENT,
            "instructions": EV_AGENT_VOICE["prompt"],
            "audio": {"output": {"voice": EV_AGENT_VOICE["voice"]}}
        }
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return {"token": data["value"], "expires_at": data.get("expires_at"), "voice": EV_AGENT_VOICE["voice"]}


@app.post("/api/voice/sdp")
async def voice_sdp(request: Request):
    """Proxy SDP exchange so Azure URL stays server-side."""
    body = await request.json()
    url = f"{AZURE_BASE}/openai/v1/realtime/calls?webrtcfilter=on"
    headers = {"Authorization": f"Bearer {body['token']}", "Content-Type": "application/sdp"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, content=body["sdp"], headers=headers)
        resp.raise_for_status()
        return Response(content=resp.content, media_type="application/sdp")


NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
    "Surrogate-Control": "no-store",
}


@app.get("/bets")
async def bets_page():
    return FileResponse("static/bets.html", headers=NO_CACHE_HEADERS)


@app.get("/ev")
async def ev_page():
    return FileResponse("static/ev.html", headers=NO_CACHE_HEADERS)


@app.get("/picks")
async def picks_page():
    return FileResponse("static/picks.html", headers=NO_CACHE_HEADERS)


@app.get("/")
async def root():
    return FileResponse("static/index.html", headers=NO_CACHE_HEADERS)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

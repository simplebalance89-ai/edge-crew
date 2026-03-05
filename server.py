import os
import time
import json
import uuid
import hashlib
import secrets
import httpx
import re
from openai import AzureOpenAI
from datetime import datetime
from zoneinfo import ZoneInfo

PST = ZoneInfo("America/Los_Angeles")
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse


import html as _html

def _sanitize(s):
    """Strip HTML/script tags from user input."""
    if not isinstance(s, str):
        return s
    return _html.escape(s.strip())

app = FastAPI()


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

    return game


SPORT_KEYS = {
    "nba": ["basketball_nba"],
    "nhl": ["icehockey_nhl"],
    "nfl": ["americanfootball_nfl"],
    "mlb": ["baseball_mlb"],
    "mma": ["mma_mixed_martial_arts"],
    "boxing": ["boxing_boxing"],
    "soccer": [
        "soccer_usa_mls",
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_uefa_champs_league",
        "soccer_mexico_ligamx",
    ],
}


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
    for sport in ["nba", "nhl", "soccer", "mma", "boxing"]:
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
        game_time = g.get("time", "?")

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
    today = time.strftime("%B %d, %Y")
    now_time = time.strftime("%I:%M %p %Z")

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

    # ===== BUILD WEIGHTED MATRIX SECTION =====
    matrix_section = _build_matrix_section(sport_lower)


    prompt = f"""You are Edge Finder v2, a sharp sports betting analyst. You have access to REAL injury data and a weighted scoring matrix.

CREW: Peter (heavy/value/sharp, sizes up on conviction), Chinny (props/NHL/soccer master), Jimmy (new, learning), Sinton.ia (card builder/grader).
RULES: "Why is the market wrong?" = required for every grade. No answer = NO BET (grade D/F). Valid edges: news not priced in, public overreaction, rest/schedule, matchup-specific, sharp vs public, situational. Invalid: "better team", "should win", "volume play".

Today's {sport.upper()} slate - {today} (pulled at {now_time}):

=== ODDS DATA ===
{games_text}
{incomplete_note}

=== INJURY & LINEUP INTELLIGENCE (CBS Sports + RotoWire + API-Sports) ===
{injury_context}

{matrix_section}

=== EDGE QUESTION (MANDATORY for every game) ===
Before grading: "Why is the market wrong here?" If you cannot answer, grade D or F.

Generate analysis in this EXACT JSON format:
{{
  "gotcha": "HTML unordered list (<ul><li>...</li></ul>). 4-8 bullet points covering: KEY INJURIES affecting tonight's lines (star players OUT/GTD), B2B/rest flags, line movement alerts, traps to avoid, weather (outdoor), sharp money indicators. Bold critical items with <strong>. End with: <li><em>Analysis generated {now_time} | Data: API-Sports + RotoWire + SharpAPI</em></li>",
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
      "chinny_props": ["player PROP over/under LINE - matchup reason", "..."],
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
- Chinny props: top 3-5 per game, B+ grade minimum. Skip for INCOMPLETE.
- PASS games get grade D or F with explicit reason.
- Be brutally honest. C means marginal. D means no edge. F means trap.

Return ONLY valid JSON. No markdown. No explanation."""

    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_KEY,
            api_version="2024-10-21",
        )
        response = client.chat.completions.create(
            model=AZURE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=6000,
        )
        raw = response.choices[0].message.content.strip()
        # Clean markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

        analysis = json.loads(raw)
        analysis["sport"] = sport.upper()
        analysis["generated_at"] = _now_ts()
        analysis["source"] = "Azure OpenAI (Edge Finder)"
        analysis["books_used"] = odds_data.get("books_used", [])
        analysis["games_complete"] = len(complete_games)
        analysis["games_incomplete"] = len(incomplete_games)
        analysis["injury_source"] = "CBS Sports + RotoWire + API-Sports" if injury_context else "none"
        analysis["injury_data_length"] = len(injury_context)
        analysis["matrix"] = sport_lower in SPORT_MATRICES

        _set_cache(cache_key, analysis)
        return JSONResponse(analysis)

    except json.JSONDecodeError:
        return JSONResponse({
            "sport": sport.upper(),
            "gotcha": "Analysis generation returned invalid format. Refresh to retry.",
            "games": [],
            "generated_at": _now_ts(),
            "raw": raw[:500] if 'raw' in dir() else "",
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
    """Fetch completed game scores from The Odds API."""
    if not ODDS_API_KEY:
        return []
    url = f"{ODDS_API_BASE}/{sport_key}/scores/"
    params = {"apiKey": ODDS_API_KEY, "daysFrom": 3}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params=params)
            if r.status_code == 200:
                return r.json()
    except Exception as e:
        logger.warning(f"Scores fetch failed for {sport_key}: {e}")
    return []


def _normalize_team(name: str) -> str:
    """Normalize team name for fuzzy matching."""
    return name.lower().strip().replace(".", "").replace("-", " ")


def _teams_match(pick_text: str, home: str, away: str) -> bool:
    """Check if a pick's matchup references this game."""
    pt = _normalize_team(pick_text)
    h = _normalize_team(home)
    a = _normalize_team(away)
    # Check if both team names (or significant parts) appear in the pick matchup
    h_parts = h.split()
    a_parts = a.split()
    h_match = any(p in pt for p in h_parts if len(p) > 3) or h in pt
    a_match = any(p in pt for p in a_parts if len(p) > 3) or a in pt
    return h_match and a_match


def _grade_pick_against_score(pick: dict, game: dict) -> str | None:
    """Determine W/L/P for a pick given final scores. Returns 'W', 'L', 'P', or None."""
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

    selection = pick.get("selection", "")
    pick_type = pick.get("type", "").lower()
    sel_lower = selection.lower().strip()

    # --- MONEYLINE ---
    if pick_type in ("moneyline", "ml", "money line") or "ml" in sel_lower:
        # Figure out which team was picked
        if any(p in sel_lower for p in _normalize_team(game["home_team"]).split() if len(p) > 3):
            return "W" if home_score > away_score else ("P" if home_score == away_score else "L")
        elif any(p in sel_lower for p in _normalize_team(game["away_team"]).split() if len(p) > 3):
            return "W" if away_score > home_score else ("P" if home_score == away_score else "L")

    # --- SPREAD ---
    if pick_type == "spread" or any(c in selection for c in ["+", "-"]):
        spread_match = re.search(r'([+-]?\d+\.?\d*)', selection)
        if spread_match:
            spread = float(spread_match.group(1))
            # Determine which team the spread applies to
            sel_before_num = sel_lower[:sel_lower.find(spread_match.group(1))].strip()
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


@app.post("/api/picks/autograde")
async def autograde_picks():
    """Auto-grade ungraded picks by fetching final scores from The Odds API."""
    # Get all ungraded picks
    if sb:
        res = sb.table("picks").select("*").is_("result", "null").execute()
        ungraded = res.data or []
    else:
        data = _read_picks()
        ungraded = [p for p in data.get("picks", []) if not p.get("result")]

    if not ungraded:
        return JSONResponse({"status": "ok", "graded": 0, "message": "No ungraded picks"})

    # Determine which sports need scores
    sports_needed = set()
    for p in ungraded:
        sport = p.get("sport", "").lower()
        if sport in SPORT_KEYS:
            sports_needed.add(sport)

    # Fetch scores for each sport
    all_scores = []
    for sport in sports_needed:
        for key in SPORT_KEYS[sport]:
            scores = await _fetch_scores(key)
            all_scores.extend(scores)

    completed = [g for g in all_scores if g.get("completed")]
    if not completed:
        return JSONResponse({"status": "ok", "graded": 0, "message": f"No completed games found for {', '.join(sports_needed)}"})

    # Match and grade
    graded_count = 0
    results = []
    for pick in ungraded:
        matchup = pick.get("matchup", "")
        for game in completed:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            if not _teams_match(matchup, home, away):
                continue

            result = _grade_pick_against_score(pick, game)
            if result:
                pick_id = pick.get("id", "")
                if sb and pick_id:
                    sb.table("picks").update({
                        "result": result,
                        "graded_at": datetime.now(PST).isoformat(),
                    }).eq("id", pick_id).execute()
                elif not sb:
                    file_data = _read_picks()
                    for fp in file_data["picks"]:
                        if fp["id"] == pick_id:
                            fp["result"] = result
                            fp["graded_at"] = datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S")
                            break
                    _write_picks(file_data)

                graded_count += 1
                scores_info = game.get("scores", [])
                score_str = " - ".join([f"{s['name']} {s.get('score', '?')}" for s in scores_info])
                results.append({
                    "pick_id": pick_id,
                    "selection": pick.get("selection", ""),
                    "matchup": matchup,
                    "result": result,
                    "final_score": score_str,
                })
                break

    return JSONResponse({
        "status": "ok",
        "graded": graded_count,
        "total_ungraded": len(ungraded),
        "completed_games": len(completed),
        "results": results,
    })


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
    """Fetch per-game lineups from API-Sports for the frontend."""
    sport_lower = sport.lower()
    lineups = await _fetch_api_sports_lineups(sport_lower)
    return JSONResponse({
        "sport": sport.upper(),
        "lineups": lineups,
        "source": "API-Sports" if lineups else "none",
        "fetched_at": _now_ts(),
    })


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
    CBS_URLS = {"nba": "https://www.cbssports.com/nba/injuries/", "nhl": "https://www.cbssports.com/nhl/injuries/"}
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


NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
    "Surrogate-Control": "no-store",
}


@app.get("/bets")
async def bets_page():
    return FileResponse("static/bets.html", headers=NO_CACHE_HEADERS)


@app.get("/")
async def root():
    return FileResponse("static/index.html", headers=NO_CACHE_HEADERS)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

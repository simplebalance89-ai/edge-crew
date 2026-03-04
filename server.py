import os
import time
import json
import uuid
import httpx
from openai import AzureOpenAI
from datetime import datetime
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



@app.middleware("http")
async def no_cache_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Surrogate-Control"] = "no-store"
    return response


ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
PREFERRED_BOOK = "hardrockbet"
FALLBACK_BOOKS = ["draftkings", "fanduel", "betmgm", "bovada"]
REGIONS = "us,us2"
UPSETS_FILE = os.path.join(os.path.dirname(__file__), "data", "upsets.json")
PICKS_FILE = os.path.join(os.path.dirname(__file__), "data", "picks.json")

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
    """Return current timestamp string for API responses."""
    return time.strftime("%I:%M %p %Z — %b %d, %Y")


def _get_cached(key, ttl=None):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < (ttl or CACHE_TTL):
            return data
    return None


def _set_cache(key, data):
    _cache[key] = (data, time.time())


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

    # --- PRIMARY: SharpAPI ---
    if SHARPAPI_KEY and sport_lower in SHARPAPI_LEAGUES:
        all_games = await _fetch_sharpapi_odds(sport_lower, label)
        if all_games:
            source_name = "SharpAPI (DraftKings)"

    # --- FALLBACK: The Odds API ---
    if not all_games and ODDS_API_KEY:
        keys = SPORT_KEYS.get(sport_lower, [sport_lower])
        for key in keys:
            games = await _fetch_sport_odds(key, markets, label)
            all_games.extend(games)
        if all_games:
            source_name = "The Odds API"

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

    result["primary"] = "SharpAPI" if SHARPAPI_KEY else "The Odds API"
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

    prompt = f"""You are Edge Finder, a sharp sports betting analyst for a crew of 3 bettors: Peter (heavy/value/sharp), Chinny (props/NHL/soccer), and Jimmy (new, learning).

Today's {sport.upper()} slate — {today} (pulled at {now_time}):
{games_text}
{incomplete_note}
Generate analysis in this exact JSON format:
{{
  "gotcha": "3-6 bullet points as an HTML unordered list (<ul><li>...</li></ul>). Key injuries, situational edges, traps, B2B flags, line moves, weather (outdoor sports). Bold the important parts with <strong>. Include a timestamp note at the end: <li><em>Analysis generated {now_time}</em></li>",
  "games": [
    {{
      "matchup": "AWAY @ HOME",
      "grade": "A/A-/B+/B/B-/C+/C/INCOMPLETE",
      "tags": ["B2B", "SHARP", "TRAP", "UPSET", "PASS", "BEST BET", "INCOMPLETE"],
      "edge_summary": "One line edge summary",
      "peter_zone": "2-3 sentences. Peter's perspective — value, sharps, line moves, conviction level.",
      "trends": ["trend 1", "trend 2", "trend 3"],
      "flags": ["injury/flag 1", "flag 2"],
      "chinny_props": ["player PROP over/under LINE — reason", "..."],
      "data_status": "COMPLETE or INCOMPLETE — state what's missing if incomplete",
      "book_source": "which bookmaker provided these lines"
    }}
  ]
}}

CRITICAL RULES:
- For NHL: games with totals-only (over/under) OR moneyline-only ARE gradeable. SharpAPI/DraftKings often only provides totals for NHL. Grade based on available data — total value, matchup quality, trends, situational factors. Only mark INCOMPLETE if NO data at all.
- For MMA/Boxing: moneyline-only games ARE gradeable. Grade based on ML value and matchup.
- For NBA/NFL/MLB/Soccer: if a game is missing spread, total, OR moneyline — grade it "INCOMPLETE" with tag "INCOMPLETE".
- In data_status, state exactly what's missing: "MISSING: spread" or "MISSING: ML, total" etc.
- Do NOT assign a real grade (A through C) to any NBA/NFL/MLB game with incomplete lines.
- INCOMPLETE games still get listed — show the matchup but make it clear you can't grade without full data.
- For complete games: grade honestly. C means skip. A means best bet.
- Flag PASS games explicitly.
- Chinny's props: top 3-5 per game. Player props only. Grade B or higher. Skip for INCOMPLETE games.
- Be specific about injury impacts on lines.
- If a line looks suspicious or moved significantly, flag it.
- Keep it sharp. No filler. Every word earns its spot.

Return ONLY valid JSON. No markdown fences. No explanation."""

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
            max_tokens=4000,
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
    if sb:
        query = sb.table("picks").select("*").order("created_at", desc=True)
        if date:
            if date == "today":
                date = datetime.now().strftime("%Y-%m-%d")
            query = query.eq("date", date)
        res = query.execute()
        return JSONResponse({"picks": res.data, "count": len(res.data)})

    data = _read_picks()
    picks = data.get("picks", [])
    if date:
        if date == "today":
            date = datetime.now().strftime("%Y-%m-%d")
        picks = [p for p in picks if p.get("date", "").startswith(date)]
    return JSONResponse({"picks": picks, "count": len(picks)})


@app.post("/api/picks")
async def save_pick(request: Request):
    """Save a new pick from the bet slip popup."""
    body = await request.json()
    name = body.get("name", "")
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
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "placed": False,
        "placed_at": None,
        "result": None,
        "graded_at": None,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    name = body.get("name", "")
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
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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


@app.post("/api/cache/clear")
async def clear_server_cache():
    """Clear all server-side cached odds and analysis data."""
    _cache.clear()
    return JSONResponse({"status": "cleared", "message": "Server cache flushed"})


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

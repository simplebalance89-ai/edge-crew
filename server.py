import os
import time
import json
import httpx
from openai import AzureOpenAI
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

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

# Azure OpenAI config
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_MODEL = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4.1-mini")

# Simple in-memory cache to save API credits
_cache = {}
CACHE_TTL = 300  # 5 minutes
ANALYSIS_CACHE_TTL = 900  # 15 minutes for analysis


def _now_ts():
    """Return current timestamp string for API responses."""
    return time.strftime("%I:%M %p %Z — %b %d, %Y")


def _get_cached(key):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
    return None


def _set_cache(key, data):
    _cache[key] = (data, time.time())


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
    """Fetch live odds from The Odds API for a given sport."""
    if not ODDS_API_KEY:
        return JSONResponse({"error": "ODDS_API_KEY not configured"}, status_code=500)

    sport_lower = sport.lower()
    cache_key = f"{sport_lower}:{markets}"
    cached = _get_cached(cache_key)
    if cached:
        return JSONResponse(cached)

    keys = SPORT_KEYS.get(sport_lower, [sport_lower])
    label = sport.upper()

    all_games = []
    for key in keys:
        games = await _fetch_sport_odds(key, markets, label)
        all_games.extend(games)

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
        "source": f"Preferred: Hard Rock Bet | Active: {', '.join(books_used) if books_used else 'none'}",
        "fetched_at": _now_ts(),
        "cached": False,
    }

    _set_cache(cache_key, result)
    return JSONResponse(result)


@app.get("/api/slate")
async def get_slate():
    """Fetch full slate — all sports combined."""
    if not ODDS_API_KEY:
        return JSONResponse({"error": "ODDS_API_KEY not configured"}, status_code=500)

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
        "source": "The Odds API (multi-book)",
        "fetched_at": _now_ts(),
    })


@app.get("/api/credits")
async def get_credits():
    """Check remaining API credits."""
    if not ODDS_API_KEY:
        return JSONResponse({"error": "ODDS_API_KEY not configured"})

    url = f"{ODDS_API_BASE}/"
    params = {"apiKey": ODDS_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            return JSONResponse({
                "remaining": resp.headers.get("x-requests-remaining", "?"),
                "used": resp.headers.get("x-requests-used", "?"),
                "checked_at": _now_ts(),
            })
    except Exception as e:
        return JSONResponse({"error": str(e)})


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
    cached = _get_cached(cache_key)
    if cached and (time.time() - _cache[cache_key][1]) < ANALYSIS_CACHE_TTL:
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
- If a game is missing spread, total, OR moneyline — grade it "INCOMPLETE" with tag "INCOMPLETE".
- In data_status, state exactly what's missing: "MISSING: spread" or "MISSING: ML, total" etc.
- Do NOT assign a real grade (A through C) to any game with incomplete lines.
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


@app.get("/api/upsets")
async def get_upsets():
    """Return current upset specials from data/upsets.json."""
    try:
        with open(UPSETS_FILE, "r") as f:
            data = json.load(f)
        return JSONResponse(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return JSONResponse({"NBA": "", "NHL": ""})


@app.post("/api/upsets")
async def save_upsets(request: Request):
    """Save an upset special for a sport to data/upsets.json."""
    body = await request.json()
    sport = body.get("sport", "").upper()
    text = body.get("text", "")

    if sport not in ("NBA", "NHL"):
        return JSONResponse({"error": "Invalid sport"}, status_code=400)

    # Read existing data
    try:
        with open(UPSETS_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"NBA": "", "NHL": ""}

    data[sport] = text
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(UPSETS_FILE), exist_ok=True)
    with open(UPSETS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return JSONResponse({"status": "ok", "sport": sport, "updated_at": data["updated_at"]})


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

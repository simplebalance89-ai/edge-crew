import os
import time
import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
BOOKMAKER = "hardrockbet"
REGIONS = "us,us2"

# Simple in-memory cache to save API credits
_cache = {}
CACHE_TTL = 300  # 5 minutes


def _get_cached(key):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
    return None


def _set_cache(key, data):
    _cache[key] = (data, time.time())


def _parse_event(event, sport_label):
    """Parse a single event from The Odds API into clean format."""
    game = {
        "id": event["id"],
        "sport": sport_label,
        "away": event["away_team"],
        "home": event["home_team"],
        "time": event["commence_time"],
        "bookmaker": BOOKMAKER,
        "markets": {},
    }

    for bk in event.get("bookmakers", []):
        if bk["key"] == BOOKMAKER:
            for market in bk.get("markets", []):
                game["markets"][market["key"]] = market["outcomes"]

    spreads = game["markets"].get("spreads", [])
    totals = game["markets"].get("totals", [])
    h2h = game["markets"].get("h2h", [])

    if spreads:
        for s in spreads:
            if s["name"] == event["away_team"]:
                game["away_spread"] = s.get("point", 0)
                game["away_spread_odds"] = s.get("price", -110)
            elif s["name"] == event["home_team"]:
                game["home_spread"] = s.get("point", 0)
                game["home_spread_odds"] = s.get("price", -110)

    if totals:
        game["total"] = totals[0].get("point", 0)
        for t in totals:
            if t["name"] == "Over":
                game["over_odds"] = t.get("price", -110)
            elif t["name"] == "Under":
                game["under_odds"] = t.get("price", -110)

    if h2h:
        for h in h2h:
            if h["name"] == event["away_team"]:
                game["away_ml"] = h.get("price", 0)
            elif h["name"] == event["home_team"]:
                game["home_ml"] = h.get("price", 0)

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
    """Fetch odds for a single sport key from The Odds API."""
    url = f"{ODDS_API_BASE}/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": markets,
        "oddsFormat": "american",
        "bookmakers": BOOKMAKER,
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

    result = {
        "sport": label,
        "games": all_games,
        "count": len(all_games),
        "source": "Hard Rock Bet via The Odds API",
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
            import json
            data = json.loads(resp.body)
            if "games" in data:
                all_games.extend(data["games"])

    return JSONResponse({
        "games": all_games,
        "count": len(all_games),
        "source": "Hard Rock Bet via The Odds API",
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
            })
    except Exception as e:
        return JSONResponse({"error": str(e)})


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

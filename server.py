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
import statistics
import db
from openai import AzureOpenAI, OpenAI
import anthropic as anthropic_sdk
from datetime import datetime, timedelta
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

# ── Error Log ─────────────────────────────────────────────────────────────────
from collections import deque
_error_log = deque(maxlen=100)

def log_error(source: str, message: str, detail: str = ""):
    _error_log.appendleft({
        "ts": datetime.now(PST).isoformat(),
        "source": source,
        "message": message,
        "detail": str(detail)[:500],
    })
    logger.error("[%s] %s — %s", source, message, str(detail)[:200])

@app.get("/api/errors")
async def get_error_log(limit: int = 50):
    return {"errors": list(_error_log)[:limit], "total": len(_error_log)}

@app.post("/api/errors")
async def post_error(request: Request):
    """Accept error reports from frontend JS."""
    try:
        body = await request.json()
        log_error(
            body.get("source", "frontend"),
            body.get("error", "unknown"),
            body.get("stack", ""),
        )
    except Exception:
        pass
    return {"status": "ok"}


async def _autograde_loop():
    """Background loop: auto-grade picks every 2 hours, by game.

    Architecture:
    1. Grade from local scores archive first (no API calls)
    2. Only fetch fresh scores for sports/dates with remaining ungraded picks
    3. Cache completed game results so they're never re-fetched
    """
    await asyncio.sleep(60)  # Wait 60s after startup before first check
    while True:
        try:
            now = datetime.now(PST)
            data = _read_picks()
            ungraded = [p for p in data.get("picks", []) if not p.get("result")]

            if ungraded:
                ungraded = _deduplicate_picks(ungraded)
                print(f"[AUTOGRADE] {len(ungraded)} ungraded picks at {now.strftime('%I:%M %p PST')}")

                # ── Pass 1: Grade from local archive (FREE — no API calls) ──
                archive = _read_scores_archive()
                archived_completed = [g for g in archive.values() if g.get("completed")]
                graded_from_cache = 0

                if archived_completed:
                    for game in archived_completed:
                        home = game.get("home_team", "")
                        away = game.get("away_team", "")
                        for pick in ungraded:
                            if pick.get("result") or pick.get("type", "").lower() == "parlay":
                                continue
                            pick_sport = pick.get("sport", "").lower()
                            if not _teams_match(pick.get("matchup", ""), home, away, pick_sport):
                                if not _teams_match(pick.get("selection", ""), home, away, pick_sport):
                                    continue
                            if not _game_date_matches_pick(pick, game):
                                continue
                            grade = _grade_pick_against_score(pick, game)
                            if grade:
                                await _update_pick_result(pick.get("id", ""), grade)
                                pick["result"] = grade
                                graded_from_cache += 1
                                print(f"[AUTOGRADE] Cache hit — graded {pick.get('selection', '?')} as {grade}")

                if graded_from_cache:
                    print(f"[AUTOGRADE] Graded {graded_from_cache} picks from cache (no API calls)")

                # ── Pass 2: Fetch fresh scores only for remaining ungraded picks ──
                still_ungraded = [p for p in ungraded if not p.get("result") and p.get("type", "").lower() != "parlay"]
                sports_needed = set()
                for p in still_ungraded:
                    sport = p.get("sport", "").lower()
                    if sport in SPORT_KEYS:
                        sports_needed.add(sport)

                graded_from_api = 0
                all_completed = list(archived_completed)  # Start with cached for parlay grading

                if sports_needed:
                    print(f"[AUTOGRADE] Fetching scores for: {', '.join(sorted(sports_needed))}")
                    days_needed = _days_from_oldest_pick(still_ungraded)
                    fetch_tasks = []
                    for sport in sports_needed:
                        for key in SPORT_KEYS.get(sport, []):
                            fetch_tasks.append(_fetch_scores(key, days_from=days_needed))

                    all_scores = []
                    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                    for result in results:
                        if not isinstance(result, Exception) and result:
                            all_scores.extend(result)

                    api_completed = [g for g in all_scores if g.get("completed")]
                    all_completed.extend(api_completed)

                    for game in api_completed:
                        home = game.get("home_team", "")
                        away = game.get("away_team", "")
                        for pick in still_ungraded:
                            if pick.get("result"):
                                continue
                            pick_sport = pick.get("sport", "").lower()
                            if not _teams_match(pick.get("matchup", ""), home, away, pick_sport):
                                if not _teams_match(pick.get("selection", ""), home, away, pick_sport):
                                    continue
                            if not _game_date_matches_pick(pick, game):
                                continue
                            grade = _grade_pick_against_score(pick, game)
                            if grade:
                                await _update_pick_result(pick.get("id", ""), grade)
                                pick["result"] = grade
                                graded_from_api += 1
                elif not sports_needed and not still_ungraded:
                    print(f"[AUTOGRADE] All singles graded from cache — no API calls needed")

                # ── Pass 3: Parlays (need all completed games from both sources) ──
                # Dedupe completed games by ID
                seen_ids = set()
                deduped_completed = []
                for g in all_completed:
                    gid = g.get("id")
                    if gid and gid not in seen_ids:
                        seen_ids.add(gid)
                        deduped_completed.append(g)

                graded_parlays = 0
                for pick in ungraded:
                    if pick.get("result"):
                        continue
                    if pick.get("type", "").lower() == "parlay":
                        grade = _grade_parlay(pick, deduped_completed)
                        if grade:
                            await _update_pick_result(pick.get("id", ""), grade)
                            pick["result"] = grade
                            graded_parlays += 1

                # ── Pass 4: AI prop grading fallback ──
                graded_ai_props = 0
                still_ungraded_props = [p for p in ungraded if not p.get("result") and _is_prop(p)]
                if still_ungraded_props and deduped_completed:
                    try:
                        ai_results = await _ai_grade_props(still_ungraded_props, deduped_completed)
                        for ai_grade in ai_results:
                            pick_id = ai_grade["pick_id"]
                            result = ai_grade["result"]
                            prop_data = {
                                "result": result,
                                "actual_value": ai_grade.get("actual_value", 0),
                                "line": ai_grade.get("line", 0),
                                "stat": ai_grade.get("stat", ""),
                                "player": ai_grade.get("player", ""),
                                "over_under": ai_grade.get("over_under", "over"),
                                "pct_over": ai_grade.get("pct_over", 0),
                            }
                            await _update_pick_result(pick_id, result, prop_data=prop_data)
                            graded_ai_props += 1
                            # Update in-memory pick
                            for p in ungraded:
                                if p.get("id") == pick_id:
                                    p["result"] = result
                                    break
                            # Prop board
                            if result == "W":
                                matching_pick = next((p for p in ungraded if p.get("id") == pick_id), {})
                                _maybe_add_to_prop_board(matching_pick, prop_data)
                        if graded_ai_props:
                            print(f"[AUTOGRADE] AI prop fallback graded {graded_ai_props} props")
                    except Exception as e:
                        logger.warning(f"[AUTOGRADE] AI prop grading failed: {e}")

                total_graded = graded_from_cache + graded_from_api + graded_parlays + graded_ai_props
                if total_graded:
                    print(f"[AUTOGRADE] Total graded: {total_graded}/{len(ungraded)} (cache:{graded_from_cache} api:{graded_from_api} parlays:{graded_parlays} ai_props:{graded_ai_props})")
                else:
                    print(f"[AUTOGRADE] No picks gradable yet ({len(ungraded)} still pending)")

                # Persist closing lines for any newly completed games
                try:
                    _persist_closing_lines()
                except Exception as e:
                    logger.warning(f"Closing lines persist failed: {e}")
        except Exception as e:
            print(f"[AUTOGRADE] Error: {e}")
        await asyncio.sleep(7200)  # Every 2 hours


async def _daily_slate_pull():
    """Background loop: pull daily slate at scheduled times."""
    await asyncio.sleep(30)
    last_slate_pull = None
    last_analysis_run = None
    last_ncaab_morning = None  # Track 7 AM NCAAB morning slate
    ncaab_morning_retries = 0  # Retry counter for 7 AM NCAAB
    while True:
        try:
            now = datetime.now(PST)
            hour = now.hour
            today = now.strftime("%Y-%m-%d")

            should_pull_slate = False
            if hour in (0, 6, 23) and last_slate_pull != f"{today}:{hour}":
                should_pull_slate = True
                last_slate_pull = f"{today}:{hour}"
            elif last_slate_pull is None:
                test_path = _slate_path("nba")
                if not os.path.exists(test_path):
                    should_pull_slate = True
                    last_slate_pull = f"{today}:boot"

            if should_pull_slate:
                active_sports = _in_season_sports()
                print(f"[SLATE] Pulling daily slate at {now.strftime('%I:%M %p PST')} — {len(active_sports)} sports in season")
                for sport in active_sports:
                    try:
                        resp = await get_odds(sport)
                        if hasattr(resp, 'body'):
                            data = json.loads(resp.body)
                            if data.get("games"):
                                _save_daily_slate(sport, data["games"])
                    except Exception as e:
                        print(f"[SLATE] Error pulling {sport}: {e}")
                    await asyncio.sleep(2)
                # Auto-grade after midnight slate pull — cards ready with analysis by morning
                if hour == 0:
                    print(f"[ANALYSIS] Midnight auto-grade — building tomorrow's cards")
                    async def _midnight_one(s):
                        async with _analysis_sem:
                            try:
                                resp = await get_analysis(s)
                                if hasattr(resp, 'body'):
                                    d = json.loads(resp.body)
                                    if d.get("games"):
                                        _save_analysis_cache(s, d)
                            except Exception as e:
                                print(f"[ANALYSIS] Midnight error {s}: {e}")
                    await asyncio.gather(*[_midnight_one(s) for s in active_sports])

            # Scheduled analysis: 10AM, 1PM, 4PM PST — keep cache warm all day
            # 10AM: morning lines locked, early edges
            # 1PM: updated lines, injury news, afternoon sports (soccer/MMA)
            # 4PM: right before tip, lineups confirmed
            should_run_analysis = False
            if hour in (10, 13, 16) and last_analysis_run != f"{today}:{hour}":
                should_run_analysis = True
                last_analysis_run = f"{today}:{hour}"

            if should_run_analysis:
                active_sports = _in_season_sports()
                print(f"[ANALYSIS] Scheduled analysis at {now.strftime('%I:%M %p PST')} — {len(active_sports)} sports")
                failed_sports = []
                async def _sched_one(s):
                    async with _analysis_sem:
                        try:
                            # Force fresh odds before analysis
                            odds_key = f"{s}:h2h,spreads,totals"
                            if odds_key in _cache:
                                del _cache[odds_key]
                            resp = await get_analysis(s)
                            if hasattr(resp, 'body'):
                                d = json.loads(resp.body)
                                if d.get("games"):
                                    _save_analysis_cache(s, d)
                                    print(f"[ANALYSIS] {s.upper()}: {len(d['games'])} games analyzed")
                                else:
                                    print(f"[ANALYSIS] {s.upper()}: No games on slate")
                            else:
                                failed_sports.append(s)
                        except Exception as e:
                            print(f"[ANALYSIS] {s.upper()} failed: {e}")
                            failed_sports.append(s)
                await asyncio.gather(*[_sched_one(s) for s in active_sports])
                # Retry failed sports once after 30min
                if failed_sports:
                    print(f"[ANALYSIS] Retrying {len(failed_sports)} failed sports in 30min: {failed_sports}")
                    await asyncio.sleep(1800)
                    async def _retry_one(s):
                        async with _analysis_sem:
                            try:
                                resp = await get_analysis(s)
                                if hasattr(resp, 'body'):
                                    d = json.loads(resp.body)
                                    if d.get("games"):
                                        _save_analysis_cache(s, d)
                                        print(f"[ANALYSIS] Retry {s.upper()}: {len(d['games'])} games")
                            except Exception as e:
                                print(f"[ANALYSIS] Retry {s.upper()} failed again: {e}")
                    await asyncio.gather(*[_retry_one(s) for s in failed_sports])

            # ── 7 AM PST NCAAB Morning Slate ──────────────────────────────────
            # March Madness & regular season: NCAAB games tip early, analyze at 7 AM
            # Retry at 7:30, 8:00 if odds aren't posted yet. Skip after 3 failures.
            if "ncaab" in _in_season_sports():
                ncaab_morning_key = f"{today}:ncaab_morning"
                minute = now.minute
                should_run_ncaab_morning = False

                if hour == 7 and minute < 5 and last_ncaab_morning != ncaab_morning_key and ncaab_morning_retries == 0:
                    should_run_ncaab_morning = True
                elif hour == 7 and 25 <= minute <= 35 and last_ncaab_morning != ncaab_morning_key and ncaab_morning_retries == 1:
                    should_run_ncaab_morning = True
                elif hour == 8 and minute < 5 and last_ncaab_morning != ncaab_morning_key and ncaab_morning_retries == 2:
                    should_run_ncaab_morning = True

                # Reset retries on new day
                if last_ncaab_morning and not last_ncaab_morning.startswith(today):
                    ncaab_morning_retries = 0

                if should_run_ncaab_morning:
                    print(f"[CRON] NCAAB morning slate: attempt {ncaab_morning_retries + 1}/3 at {now.strftime('%I:%M %p PST')}")
                    try:
                        # Force fresh odds
                        ncaab_odds_key = "ncaab:h2h,spreads,totals"
                        if ncaab_odds_key in _cache:
                            del _cache[ncaab_odds_key]
                        async with _analysis_sem:
                            resp = await get_analysis("ncaab")
                            if hasattr(resp, 'body'):
                                d = json.loads(resp.body)
                                if d.get("games") and len(d["games"]) > 0:
                                    _save_analysis_cache("ncaab", d)
                                    print(f"[CRON] NCAAB morning slate: {len(d['games'])} games analyzed")
                                    last_ncaab_morning = ncaab_morning_key
                                    ncaab_morning_retries = 0
                                else:
                                    ncaab_morning_retries += 1
                                    print(f"[CRON] NCAAB morning slate: no games/odds yet (retry {ncaab_morning_retries}/3)")
                            else:
                                ncaab_morning_retries += 1
                                print(f"[CRON] NCAAB morning slate: no response (retry {ncaab_morning_retries}/3)")
                    except Exception as e:
                        ncaab_morning_retries += 1
                        print(f"[CRON] NCAAB morning slate failed (retry {ncaab_morning_retries}/3): {e}")
                    if ncaab_morning_retries >= 3 and last_ncaab_morning != ncaab_morning_key:
                        print(f"[CRON] NCAAB morning slate: 3 failures, skipping today")
                        last_ncaab_morning = ncaab_morning_key  # Mark as done to stop retrying

        except Exception as e:
            print(f"[SLATE] Loop error: {e}")
        await asyncio.sleep(300)


async def _warm_analysis_cache():
    """On startup, check if today's analysis exists for each sport. If not, generate it.
    Runs once after boot with staggered delays to avoid hammering Azure."""
    await asyncio.sleep(120)  # Wait 2 min after boot for other services to stabilize
    active_sports = _in_season_sports()
    today = datetime.now(PST).strftime("%Y-%m-%d")
    print(f"[ANALYSIS WARM] Checking cache for {len(active_sports)} sports...")
    # First pass: check which sports need fresh analysis (sync, no API calls)
    needs_generation = []
    for sport in active_sports:
        cached = _load_analysis_cache(sport)
        if cached and cached.get("games"):
            cached_date = cached.get("generated_at", "")[:10] or cached.get("cached_at", "")[:10]
            if cached_date == today:
                print(f"[ANALYSIS WARM] {sport.upper()}: today's cache exists ({len(cached['games'])} games)")
                _set_cache(f"analysis:{sport}", cached)
                continue
        needs_generation.append(sport)

    # Second pass: generate missing caches in parallel (max 3 concurrent)
    if needs_generation:
        print(f"[ANALYSIS WARM] Generating {len(needs_generation)} sports in parallel...")
        async def _warm_one(s):
            async with _analysis_sem:
                try:
                    print(f"[ANALYSIS WARM] {s.upper()}: no today cache, generating...")
                    resp = await get_analysis(s)
                    if hasattr(resp, 'body'):
                        d = json.loads(resp.body)
                        if d.get("games"):
                            _save_analysis_cache(s, d)
                            print(f"[ANALYSIS WARM] {s.upper()}: generated {len(d['games'])} games")
                except Exception as e:
                    print(f"[ANALYSIS WARM] {s.upper()} failed: {e}")
        await asyncio.gather(*[_warm_one(s) for s in needs_generation])
    print(f"[ANALYSIS WARM] Cache warm complete")


PREFETCH_SPORTS = ["nba", "nhl", "ncaab", "mma"]
PREFETCH_INTERVAL = 600  # 10 minutes


async def _prefetch_loop():
    """Background loop: pre-warm odds/props/discrepancy cache every 10 minutes so users never wait."""
    await asyncio.sleep(30)  # Wait 30s after startup before first run
    while True:
        for sport in PREFETCH_SPORTS:
            try:
                events = await get_odds(sport)
                # Skip props/discrepancy for sports with no games on slate
                if hasattr(events, 'body'):
                    body = json.loads(events.body)
                    games = body if isinstance(body, list) else body.get("games", body.get("events", []))
                    if not games:
                        logger.info(f"[PREFETCH] {sport.upper()} no games on slate, skipping props")
                        continue
                await get_player_props(sport)
                await find_discrepancies(sport)
                await get_gap_props(sport)
                logger.info(f"[PREFETCH] {sport.upper()} cache warmed")
            except Exception as e:
                logger.warning(f"[PREFETCH] {sport} failed: {e}")
            await asyncio.sleep(2)  # Small gap between sports
        await asyncio.sleep(PREFETCH_INTERVAL)


@app.on_event("startup")
async def start_background_tasks():
    await db.init_schema()
    asyncio.create_task(_autograde_loop())
    asyncio.create_task(_daily_slate_pull())
    asyncio.create_task(_warm_analysis_cache())
    asyncio.create_task(_prefetch_loop())


@app.on_event("shutdown")
async def shutdown():
    await db.close_pool()


@app.get("/health")
async def health_check():
    """Health check for Render auto-restart."""
    return JSONResponse({"status": "ok", "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "version": BUILD_VERSION})

@app.get("/api/version")
async def get_version():
    return JSONResponse({"version": BUILD_VERSION, "built": BUILD_TS, "mode": ANALYSIS_MODE, "thinker": ANALYSIS_THINKER, "formatter": ANALYSIS_FORMATTER})


# ===== CREW AUTH =====
CREW_PIN_SALT = os.environ.get("CREW_PIN_SALT", "edge-crew-default-salt-change-me")
_CREW_DATA_DIR = "/data" if os.path.isdir("/data") else os.path.join(os.path.dirname(__file__), "data")
PROFILES_FILE = os.path.join(_CREW_DATA_DIR, "crew_profiles.json")  # persistent disk — survives deploys
_sessions = {}  # token -> {id, display_name, color, is_admin}

DEFAULT_CREW = [
    {"id": "peter", "display_name": "Peter", "color": "#D4A017", "is_admin": True},
    {"id": "jimmy", "display_name": "Jimmy", "color": "#60A5FA", "is_admin": False},
    {"id": "chinny", "display_name": "Chinny", "color": "#FF6B35", "is_admin": False},
    {"id": "alyssa", "display_name": "Alyssa", "color": "#41EAD4", "is_admin": False},
    {"id": "sintonia", "display_name": "Sinton.ia", "color": "#F72585", "is_admin": False},
    {"id": "sportsbook", "display_name": "Sportsbook.ag", "color": "#22C55E", "is_admin": False},
    {"id": "tunk", "display_name": "Tunk", "color": "#FF4500", "is_admin": False},
    {"id": "padrino", "display_name": "Padrino", "color": "#C0A062", "is_admin": False},
    {"id": "los", "display_name": "Los", "color": "#38BDF8", "is_admin": False},
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


CREW_DEFAULT_PINS = {"peter": "0000", "jimmy": "0000", "chinny": "0000",
                     "alyssa": "0000", "sintonia": "0000", "sportsbook": "0000",
                     "tunk": "1525", "padrino": "0726", "los": "4200"}

def _seed_profiles():
    """Ensure all DEFAULT_CREW members exist in profiles. Sync PINs to defaults."""
    data = _read_profiles()
    existing = {p["id"]: p for p in data.get("profiles", [])}
    updated = False
    for member in DEFAULT_CREW:
        expected_pin = CREW_DEFAULT_PINS.get(member["id"], "0000")
        expected_hash = _hash_pin(expected_pin)
        if member["id"] not in existing:
            data.setdefault("profiles", []).append({
                "id": member["id"],
                "display_name": member["display_name"],
                "pin_hash": expected_hash,
                "color": member["color"],
                "is_admin": member["is_admin"],
                "created_at": datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S"),
                "last_login": None,
            })
            updated = True
        else:
            # Sync display_name, color, is_admin, and ensure PIN hash matches
            p = existing[member["id"]]
            if p["pin_hash"] != expected_hash:
                p["pin_hash"] = expected_hash
                updated = True
            if p.get("display_name") != member["display_name"]:
                p["display_name"] = member["display_name"]
                updated = True
            if p.get("color") != member["color"]:
                p["color"] = member["color"]
                updated = True
            if p.get("is_admin") != member["is_admin"]:
                p["is_admin"] = member["is_admin"]
                updated = True
    if updated:
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
    # Kill ETags so StaticFiles can't trigger 304 Not Modified
    if "etag" in response.headers:
        del response.headers["etag"]
    if "last-modified" in response.headers:
        del response.headers["last-modified"]
    return response


ODDS_API_KEY = os.environ.get("ODDS_API_KEY_PAID", "") or os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
PREFERRED_BOOK = "hardrockbet"
FALLBACK_BOOKS = ["draftkings", "fanduel", "betmgm", "bovada"]
REGIONS = "us,us2,eu,uk,au"
# Persistent data directory — uses /data mount on Render, falls back to local ./data
DATA_DIR = "/data" if os.path.isdir("/data") else os.path.join(os.path.dirname(__file__), "data")
UPSETS_FILE = os.path.join(DATA_DIR, "upsets.json")
PICKS_FILE = os.path.join(DATA_DIR, "picks.json")
SCORES_ARCHIVE_FILE = os.path.join(DATA_DIR, "scores_archive.json")
BANKROLL_FILE = os.path.join(DATA_DIR, "bankroll.json")
GOTCHA_FILE = os.path.join(DATA_DIR, "gotcha_notes.json")

ALL_SPORTS = ["nba", "wnba", "ncaab", "ncaaf", "nhl", "mlb", "tennis", "soccer", "mma", "boxing"]

# Semaphore to limit concurrent analysis calls (avoid hammering Azure)
_analysis_sem = asyncio.Semaphore(3)

# Season map — which months each sport is active
_SEASON_MONTHS = {
    "nba": {10,11,12,1,2,3,4,5,6},
    "wnba": {5,6,7,8,9,10},
    "ncaab": {11,12,1,2,3,4},
    "ncaaf": {8,9,10,11,12,1},
    "nhl": {10,11,12,1,2,3,4,5,6},
    "mlb": {3,4,5,6,7,8,9,10,11},
    "tennis": {1,2,3,4,5,6,7,8,9,10,11},
    "soccer": {1,2,3,4,5,6,7,8,9,10,11,12},
    "mma": {1,2,3,4,5,6,7,8,9,10,11,12},
    "boxing": {1,2,3,4,5,6,7,8,9,10,11,12},
}

def _in_season_sports():
    """Return only sports currently in season."""
    month = datetime.now(PST).month
    return [s for s in ALL_SPORTS if month in _SEASON_MONTHS.get(s, set())]

# ── Daily Record (Layer 0) ──────────────────────────────────────────────────
SLATE_DIR = os.path.join(DATA_DIR, "slates")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis_cache")
os.makedirs(SLATE_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def _slate_path(sport, date_str=None):
    if not date_str:
        date_str = datetime.now(PST).strftime("%Y-%m-%d")
    return os.path.join(SLATE_DIR, f"slate_{sport}_{date_str}.json")

def _analysis_cache_path(sport, date_str=None, run="latest"):
    if not date_str:
        date_str = datetime.now(PST).strftime("%Y-%m-%d")
    return os.path.join(ANALYSIS_DIR, f"analysis_{sport}_{date_str}_{run}.json")

def _save_daily_slate(sport, games_data):
    path = _slate_path(sport)
    with open(path, "w") as f:
        json.dump({"sport": sport.upper(), "games": games_data, "fetched_at": _now_ts(), "date": datetime.now(PST).strftime("%Y-%m-%d")}, f)

def _load_daily_slate(sport):
    """Load daily slate — tries today first, falls back to yesterday."""
    path = _slate_path(sport)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Fallback: check yesterday's slate (handles midnight rollover)
    yesterday = (datetime.now(PST) - timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_path = _slate_path(sport, date_str=yesterday)
    if os.path.exists(yesterday_path):
        with open(yesterday_path) as f:
            return json.load(f)
    return None

def _save_analysis_cache(sport, analysis_data):
    path = _analysis_cache_path(sport)
    with open(path, "w") as f:
        json.dump({"sport": sport.upper(), **analysis_data, "cached_at": _now_ts()}, f)

def _load_analysis_cache(sport):
    """Load analysis from disk — tries today first, falls back to yesterday."""
    path = _analysis_cache_path(sport)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Fallback: check yesterday's cache (handles midnight date rollover)
    yesterday = (datetime.now(PST) - timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_path = _analysis_cache_path(sport, date_str=yesterday)
    if os.path.exists(yesterday_path):
        with open(yesterday_path) as f:
            data = json.load(f)
            data["_from_previous_day"] = True
            return data
    return None

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
    "soccer": [39, 140, 253, 2, 262],  # Derived below from SOCCER_LEAGUES
}

# ===== SOCCER LEAGUES REGISTRY — Single source of truth =====
SOCCER_LEAGUES = {
    # Tier 1 — Always fetched (Big 5 + UCL)
    "soccer_epl":                {"name": "EPL",          "country": "England",   "flag": "\U0001F3F4", "api_sports_id": 39,  "tier": 1},
    "soccer_spain_la_liga":      {"name": "La Liga",      "country": "Spain",     "flag": "\U0001F1EA\U0001F1F8", "api_sports_id": 140, "tier": 1},
    "soccer_germany_bundesliga": {"name": "Bundesliga",   "country": "Germany",   "flag": "\U0001F1E9\U0001F1EA", "api_sports_id": 78,  "tier": 1},
    "soccer_italy_serie_a":      {"name": "Serie A",      "country": "Italy",     "flag": "\U0001F1EE\U0001F1F9", "api_sports_id": 135, "tier": 1},
    "soccer_france_ligue_one":   {"name": "Ligue 1",      "country": "France",    "flag": "\U0001F1EB\U0001F1F7", "api_sports_id": 61,  "tier": 1},
    "soccer_uefa_champs_league": {"name": "UCL",          "country": "Europe",    "flag": "\U0001F3C6", "api_sports_id": 2,   "tier": 1},
    # Tier 2 — Always fetched (Americas + Europa)
    "soccer_usa_mls":            {"name": "MLS",          "country": "USA",       "flag": "\U0001F1FA\U0001F1F8", "api_sports_id": 253, "tier": 2},
    "soccer_mexico_ligamx":      {"name": "Liga MX",      "country": "Mexico",    "flag": "\U0001F1F2\U0001F1FD", "api_sports_id": 262, "tier": 2},
    "soccer_brazil_serie_a":     {"name": "Brasileir\u00e3o",  "country": "Brazil",    "flag": "\U0001F1E7\U0001F1F7", "api_sports_id": 71,  "tier": 2},
    "soccer_uefa_europa_league": {"name": "Europa League", "country": "Europe",   "flag": "\U0001F3C6", "api_sports_id": 3,   "tier": 2},
    # Tier 3 — On-demand only (user clicks to load, saves API quota)
    "soccer_netherlands_eredivisie":      {"name": "Eredivisie",    "country": "Netherlands", "flag": "\U0001F1F3\U0001F1F1", "api_sports_id": 88,  "tier": 3},
    "soccer_portugal_primeira_liga":      {"name": "Primeira Liga", "country": "Portugal",    "flag": "\U0001F1F5\U0001F1F9", "api_sports_id": 94,  "tier": 3},
    "soccer_turkey_super_league":         {"name": "S\u00fcper Lig",     "country": "Turkey",      "flag": "\U0001F1F9\U0001F1F7", "api_sports_id": 203, "tier": 3},
    "soccer_australia_aleague":           {"name": "A-League",      "country": "Australia",   "flag": "\U0001F1E6\U0001F1FA", "api_sports_id": 188, "tier": 3},
    "soccer_japan_j_league":             {"name": "J1 League",     "country": "Japan",       "flag": "\U0001F1EF\U0001F1F5", "api_sports_id": 98,  "tier": 3},
    "soccer_korea_kleague":              {"name": "K League",      "country": "South Korea", "flag": "\U0001F1F0\U0001F1F7", "api_sports_id": 292, "tier": 3},
    "soccer_conmebol_copa_libertadores": {"name": "Libertadores",  "country": "S. America",  "flag": "\U0001F3C6", "api_sports_id": 13,  "tier": 3},
}
# Derive SPORT_KEYS and API_SPORTS_LEAGUES from registry
SPORT_KEYS_SOCCER = [k for k, v in SOCCER_LEAGUES.items() if v["tier"] <= 2]
API_SPORTS_LEAGUES["soccer"] = [v["api_sports_id"] for v in SOCCER_LEAGUES.values() if v["tier"] <= 2]

# Azure OpenAI config
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_MODEL = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4.1")
REALTIME_DEPLOYMENT = "gpt-4o-realtime"
AZURE_BASE = AZURE_ENDPOINT.rstrip("/")

# Two-model analysis engine: Grok reasons, DeepSeek formats
ANALYSIS_THINKER = os.environ.get("ANALYSIS_THINKER", "grok-4-1-fast-reasoning")
ANALYSIS_FORMATTER = os.environ.get("ANALYSIS_FORMATTER", "DeepSeek-V3.2")
ANALYSIS_MODE = os.environ.get("ANALYSIS_MODE", "twomodel")  # "twomodel" or "single"
THINKER_ENDPOINT = os.environ.get("THINKER_ENDPOINT", "https://pwgcerp-9302-resource.services.ai.azure.com/openai/v1/")

# Anthropic config
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Challenger Model of the Week — rotation list
CHALLENGER_MODELS = [
    {"name": "gpt-5-mini", "endpoint": "azure", "display": "GPT-5 Mini"},
    {"name": "o4-mini", "endpoint": "azure", "display": "o4-mini"},
    {"name": "grok-3", "endpoint": "ai_services", "display": "Grok 3"},
    {"name": "Llama-4-Maverick-17B-128E-Instruct-FP8", "endpoint": "ai_services", "display": "Llama 4 Maverick"},
    {"name": "DeepSeek-V3.2", "endpoint": "ai_services", "display": "DeepSeek V3.2"},
    {"name": "claude-sonnet-4-6", "endpoint": "anthropic", "display": "Claude Sonnet 4.6"},
    {"name": "grok-4-fast-reasoning", "endpoint": "ai_services", "display": "Grok 4 Fast"},
]
CHALLENGER_MODEL_OVERRIDE = os.environ.get("CHALLENGER_MODEL", "")
_challenger_cache = {}  # week_num -> model dict


def _get_weekly_challenger():
    """Return this week's challenger model dict based on ISO week rotation."""
    now = datetime.now(PST)
    week_num = now.isocalendar()[1]
    if week_num in _challenger_cache:
        return _challenger_cache[week_num]
    # Check for env override
    if CHALLENGER_MODEL_OVERRIDE:
        for m in CHALLENGER_MODELS:
            if m["name"] == CHALLENGER_MODEL_OVERRIDE:
                _challenger_cache[week_num] = m
                return m
        # Override name not in list — build an ad-hoc entry (assume ai_services)
        override = {"name": CHALLENGER_MODEL_OVERRIDE, "endpoint": "ai_services", "display": CHALLENGER_MODEL_OVERRIDE}
        _challenger_cache[week_num] = override
        return override
    idx = week_num % len(CHALLENGER_MODELS)
    model = CHALLENGER_MODELS[idx]
    _challenger_cache[week_num] = model
    logger.info(f"[CHALLENGER] Week {week_num} challenger: {model['display']} ({model['name']})")
    return model

# Build version — auto-set at server startup for cache busting
_git_hash = os.environ.get("RENDER_GIT_COMMIT", "")[:7]
if not _git_hash:
    import subprocess as _sp
    try:
        _git_hash = _sp.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=_sp.DEVNULL).decode().strip()
    except Exception:
        _git_hash = "dev"
BUILD_VERSION = f"v2.{datetime.now(PST).strftime('%m%d.%H%M')}.{_git_hash}"
BUILD_TS = datetime.now(PST).strftime("%Y-%m-%d %I:%M %p PST")
logger.info(f"[BUILD] {BUILD_VERSION} — {BUILD_TS}")

# EV Agent character config for voice
EV_AGENT_VOICE = {
    "voice": "ash",
    "prompt": """You are EV, the Expected Value agent for Edge Crew — a sports betting analytics platform. You are 100% math, no gut, pure edge. You analyze odds, find value, grade picks, and break down matchups.

Your personality: sharp, confident, data-driven. You speak in short punchy sentences. You know every sport — NBA, NFL, NHL, MLB, NCAAB, NCAAF, Tennis, Soccer, MMA, Boxing, WNBA. You reference real odds, spreads, and over/unders.

When asked about picks: give the edge percentage, the why, and the confidence level. When grading: be honest, show the math. When a user is on a streak, hype them up. When they're cold, analyze what went wrong.

Keep responses under 3 sentences for voice. Be the sharpest analyst in the room."""
}

# Supabase removed — all storage on Render persistent disk (/data/)

# Simple in-memory cache to save API credits
_cache = {}
CACHE_TTL = 300  # 5 minutes
ANALYSIS_CACHE_TTL = 10800  # 3 hours — scheduled analysis runs at 12/3/6 PM PST

# Opening lines tracker — stores first-seen odds per game per day
_opening_lines = {}  # key: "YYYY-MM-DD:{game_id}" -> {spread, total, away_ml, home_ml}


def _now_ts():
    """Return current timestamp string for API responses (PST)."""
    return datetime.now(PST).strftime("%I:%M %p PST — %b %d, %Y")


def _game_not_started(game, now=None):
    """Return True if game hasn't started yet."""
    if now is None:
        now = datetime.now(PST)
    game_time_str = game.get('time', '')
    if not game_time_str:
        return False
    try:
        gt = datetime.fromisoformat(game_time_str.replace('Z', '+00:00')).astimezone(PST)
        return gt > now
    except Exception:
        return False

def _game_within_cutoff(game, cutoff):
    """Return True if game is within the cutoff time."""
    game_time_str = game.get('time', '')
    if not game_time_str:
        return False
    try:
        gt = datetime.fromisoformat(game_time_str.replace('Z', '+00:00')).astimezone(PST)
        return gt <= cutoff
    except Exception:
        return False

def _get_cached(key, ttl=None):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < (ttl or CACHE_TTL):
            return data
    return None


def _set_cache(key, data):
    _cache[key] = (data, time.time())


# ── Scores Archive — persist completed game scores locally ────────────────
def _read_scores_archive() -> dict:
    """Read archived completed scores. Keyed by game ID."""
    try:
        if os.path.exists(SCORES_ARCHIVE_FILE):
            with open(SCORES_ARCHIVE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_scores_archive(archive: dict):
    """Write scores archive. Prune games older than 30 days."""
    cutoff = (datetime.now(PST) - timedelta(days=30)).isoformat()
    pruned = {gid: g for gid, g in archive.items() if g.get("commence_time", "") >= cutoff}
    try:
        with open(SCORES_ARCHIVE_FILE, "w") as f:
            json.dump(pruned, f)
    except Exception as e:
        logger.warning(f"Scores archive save failed: {e}")


def _archive_completed_games(games: list):
    """Save completed games to local archive so we never re-fetch them."""
    completed = [g for g in games if g.get("completed") and g.get("id")]
    if not completed:
        return
    archive = _read_scores_archive()
    for g in completed:
        archive[g["id"]] = g
    _save_scores_archive(archive)
    # Dual-write completed scores to Postgres (fire-and-forget)
    asyncio.create_task(_db_write_scores(completed))


async def _db_write_scores(games: list):
    """Dual-write completed game scores to Postgres."""
    for g in games:
        try:
            await db.execute(
                """INSERT INTO scores (id, sport, home_team, away_team, home_score, away_score, completed, commence_time)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                   ON CONFLICT (id) DO UPDATE SET
                     home_score=EXCLUDED.home_score, away_score=EXCLUDED.away_score,
                     completed=EXCLUDED.completed, fetched_at=NOW()""",
                g.get("id", ""),
                g.get("sport_key", ""),
                g.get("home_team", ""),
                g.get("away_team", ""),
                _extract_score(g, "home"),
                _extract_score(g, "away"),
                g.get("completed", False),
                g.get("commence_time", ""),
            )
        except Exception as e:
            logger.error(f"DB dual-write score failed for {g.get('id')}: {e}")


def _extract_score(game: dict, side: str) -> int:
    """Extract numeric score from game dict scores array."""
    for s in game.get("scores", []):
        name = s.get("name", "").lower()
        team = game.get(f"{side}_team", "").lower()
        if name and team and name in team or team in name:
            try:
                return int(s.get("score", 0))
            except (ValueError, TypeError):
                return 0
    return 0


def _days_from_oldest_pick(picks: list) -> int:
    """Calculate daysFrom for Odds API. Always returns 3 — the API max.

    Picks older than 3 days are covered by the local scores archive instead.
    """
    return 3


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


def _persist_closing_lines():
    """Write closing lines from _opening_lines into the scores archive for completed games.

    For any completed game in the archive that doesn't have closing_spread/closing_total/
    closing_away_ml/closing_home_ml, look up the most recent values from _opening_lines
    and persist them.
    """
    archive = _read_scores_archive()
    updated = False
    for gid, game in archive.items():
        if not game.get("completed"):
            continue
        # Skip if closing lines already saved
        if game.get("closing_spread") is not None:
            continue
        # First, check if the game object itself has line data
        if game.get("home_spread") is not None:
            game["closing_spread"] = game["home_spread"]
            game["closing_total"] = game.get("total")
            game["closing_away_ml"] = game.get("away_ml")
            game["closing_home_ml"] = game.get("home_ml")
            updated = True
            continue
        # Fall back to _opening_lines (most recent values)
        best_entry = None
        for okey, oval in _opening_lines.items():
            if okey.endswith(f":{gid}"):
                if best_entry is None or oval.get("ts", 0) > best_entry.get("ts", 0):
                    best_entry = oval
        if best_entry:
            game["closing_spread"] = best_entry.get("spread")
            game["closing_total"] = best_entry.get("total")
            game["closing_away_ml"] = best_entry.get("away_ml")
            game["closing_home_ml"] = best_entry.get("home_ml")
            updated = True
    if updated:
        _save_scores_archive(archive)


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
    best_draw_ml = {"odds": None, "implied": 1.0, "book": None}
    is_three_way = False

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
                    elif outcome["name"] == "Draw":
                        is_three_way = True
                        if imp < best_draw_ml["implied"]:
                            best_draw_ml = {"odds": price, "implied": imp, "book": bk["key"]}

    if is_three_way:
        # 3-way market (soccer): must include draw in arb calc
        if best_away_ml["book"] and best_home_ml["book"] and best_draw_ml["book"]:
            total_implied = best_away_ml["implied"] + best_home_ml["implied"] + best_draw_ml["implied"]
            if total_implied < 1.0:
                profit_pct = round((1.0 / total_implied - 1.0) * 100, 2)
                arbs.append({
                    "type": "ML (3-way)",
                    "profit_pct": profit_pct,
                    "legs": [
                        {"side": away_team, "odds": best_away_ml["odds"], "book": best_away_ml["book"]},
                        {"side": "Draw", "odds": best_draw_ml["odds"], "book": best_draw_ml["book"]},
                        {"side": home_team, "odds": best_home_ml["odds"], "book": best_home_ml["book"]},
                    ],
                })
    elif best_away_ml["book"] and best_home_ml["book"]:
        # 2-way market (NBA, NHL, etc.)
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

    if arbs:
        logger.info(f"Arb found: {event.get('away_team')} @ {event.get('home_team')} — {len(arbs)} opportunities")

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
    ("star_player_status", 9, "Star player (top 2) out or limited"),
    ("rest_advantage", 9, "B2B, 3-in-4, rest days between games"),
    ("line_movement", 9, "Line movement since open — sharp action, reverse moves"),
    ("off_ranking", 8, "Offensive efficiency rating"),
    ("def_ranking", 8, "Defensive efficiency rating"),
    ("three_pt_matchup", 7, "3PT shooting matchup (attempt rate, %, defense vs 3)"),
    ("sharp_vs_public", 7, "Where the money is (sharp vs public)"),
    ("pace_matchup", 7, "Pace matchup"),
    ("recent_form", 7, "Recent form / streak (last 5-10 games)"),
    ("road_trip_length", 7, "Road trip fatigue (games on road, travel distance)"),
    ("turnover_gap", 7, "Turnover differential — possession control edge"),
    ("h2h_season", 6, "Head-to-head this season (record, margins, trends)"),
    ("paint_scoring", 6, "Points in the paint matchup"),
    ("fast_break", 6, "Fast break / transition points"),
    ("rebounding_edge", 6, "Offensive + defensive rebounding gap"),
    ("ats_trend", 6, "ATS trend (last 7/14 days)"),
    ("line_vs_model", 6, "Our composite vs the line spread — meta-edge signal"),
    ("home_away", 5, "Home/away record"),
    ("referee_bias", 5, "Referee foul-calling tendencies, pace impact"),
    ("depth_injuries", 4, "Role player / bench injuries"),
]

NHL_MATRIX = [
    ("goalie_confirmed", 9, "Goalie confirmed starter"),
    ("star_player_status", 9, "Top line center, #1 D-man out or limited"),
    ("line_movement", 9, "Line movement since open — sharp action, reverse moves"),
    ("corsi_xg", 8, "Corsi / expected goals"),
    ("pp_pk", 8, "Power play / penalty kill rankings"),
    ("b2b_fatigue", 8, "Back-to-back / schedule fatigue"),
    ("goalie_fatigue", 7, "Goalie games/minutes in last 7 days"),
    ("sharp_vs_public", 7, "Where the money is (sharp vs public)"),
    ("off_ranking", 7, "Goals for, shooting %, offensive zone time"),
    ("def_ranking", 7, "Goals against, shots against"),
    ("recent_form", 7, "Recent form / streak (last 5-10 games)"),
    ("road_trip_length", 7, "Road trip fatigue (games on road, travel distance)"),
    ("penalty_discipline", 7, "Penalty minutes trend — shorthanded exposure risk"),
    ("h2h_season", 6, "Head-to-head this season (record, margins, trends)"),
    ("save_pct_trend", 6, "Goalie save percentage trend (last 5)"),
    ("puck_luck", 6, "Shooting % anomalies — overperformance regression signal"),
    ("ats_trend", 6, "ATS trend (last 7/14 days)"),
    ("line_vs_model", 6, "Our composite vs the line spread — meta-edge signal"),
    ("home_away", 5, "Home/away record"),
    ("depth_injuries", 4, "Role player / bench injuries"),
]

SOCCER_MATRIX = [
    ("star_player_status", 9, "Key striker or GK out or limited"),
    ("btts_trend", 9, "Both Teams to Score trend"),
    ("xg_trend", 9, "xG (expected goals) trend"),
    ("line_movement", 9, "Line movement since open — sharp action, reverse moves"),
    ("home_away_form", 8, "Home/away form (last 5)"),
    ("goals_avg", 8, "Goals scored/conceded avg"),
    ("sharp_vs_public", 7, "Where the money is (sharp vs public)"),
    ("h2h_season", 7, "Head-to-head this season (record, margins)"),
    ("league_position", 7, "League table position / motivation"),
    ("recent_form", 7, "Recent form / streak (last 5-10 games)"),
    ("fixture_congestion", 7, "Midweek CL + league doubles, rotation risk"),
    ("motivation_level", 7, "Relegation battle, title race, derby — context pressure"),
    ("goalkeeper_form", 7, "Starting GK save %, error frequency, recent form"),
    ("tactical_matchup", 6, "Formation clash, style matchup advantage"),
    ("possession_matchup", 6, "Possession % matchup"),
    ("set_pieces", 6, "Corners, free kicks, set piece danger"),
    ("clean_sheet", 6, "Clean sheet pct"),
    ("ats_trend", 6, "ATS trend (last 7/14 days)"),
    ("referee_bias", 5, "Referee card tendencies, penalty award rates"),
    ("depth_injuries", 4, "Bench / rotation injuries"),
]

NCAAB_MATRIX = [
    # Tier 1 — Weight 9 (Game Changers)
    ("offensive_efficiency", 9, "KenPom/Torvik adjusted offensive efficiency — THE predictor in college basketball"),
    ("defensive_efficiency", 9, "KenPom/Torvik adjusted defensive efficiency"),
    ("star_player_status", 9, "Key player (top 2) out/limited — 8-man rotation means losing 1 = losing 12-15% of offense"),
    ("line_movement", 9, "Line movement since open — sharp action, reverse moves"),
    # Tier 2 — Weight 8 (Major Factors)
    ("tempo_matchup", 8, "Pace/tempo matchup — fast vs slow, possessions per game"),
    ("ats_trend", 8, "ATS trend (last 7/14 days) — covering trends persist more in college than NBA"),
    ("tournament_context", 8, "Tournament game? Conference tourney round? NCAA tournament? Single elimination psychology, bye advantage"),
    ("public_money_bias", 8, "Blue blood public bias — Duke/UK/UNC get hammered by public, books shade lines. Fade inflated favorites"),
    # Tier 3 — Weight 7 (Significant Factors)
    ("three_pt_reliance", 7, "3PT shooting reliance vs opponent 3PT defense — higher variance than NBA, streakier"),
    ("bench_depth", 7, "Rotation depth — college = 8-9 man rotations, thin bench + foul trouble = disaster"),
    ("rebounding", 7, "Offensive + defensive rebounding differential — second chances matter more with lower FG%"),
    ("turnover_margin", 7, "Turnover margin — live ball turnovers, steals. College teams turn it over more than NBA"),
    ("recent_form", 7, "Recent form / streak (last 5-10 games) — late season form predicts tournament performance"),
    ("coaching_tournament_record", 7, "Coach's tournament track record — some coaches consistently over/underperform in March"),
    ("net_ranking_gap", 7, "NET/RPI ranking differential — NCAA's official ranking. Large gaps = chalk, small gaps = volatility"),
    # Tier 4 — Weight 6 (Supporting Factors)
    ("neutral_site", 6, "Neutral site factor — conference tourneys, NCAA tournament. No home court. Proximity to venue matters"),
    ("conference_strength", 6, "Strength of schedule adjustment — Big Ten vs mid-major, adjusted by KenPom SOS"),
    ("ft_shooting", 6, "Free throw shooting % and FTA rate — close games decided at the line, college FT% lower than NBA"),
    ("rivalry_motivation", 6, "Rivalry, revenge spot, senior night, tournament implications — motivation is real for 19-21 year olds"),
    ("fatigue_schedule", 6, "Conference tournament fatigue — 3rd game in 3 days, late-season schedule compression"),
    ("coaching_matchup", 6, "Coach A vs Coach B history — more stable in college, coaches stay longer"),
    # Tier 5 — Weight 5 (Contextual)
    ("venue_factor", 5, "Arena-specific — tight rims, altitude, court dimensions, crowd noise in smaller gyms"),
    ("transfer_portal_impact", 5, "Key transfers in/out, roster chemistry. By March this is baked into recent_form"),
    ("line_vs_model", 5, "Our composite vs the line spread — meta-edge signal"),
]

MLB_MATRIX = [
    ("starting_pitcher", 9, "Starting pitcher quality — ERA, WHIP, K/9, recent form"),
    ("bullpen_strength", 9, "Bullpen strength — ERA, leverage usage, rest"),
    ("lineup_vs_hand", 8, "Lineup vs LHP/RHP splits — wOBA, OPS"),
    ("sharp_vs_public", 8, "Sharp money vs public betting split"),
    ("line_movement", 8, "Line movement since open — sharp action, reverse moves"),
    ("park_factor", 7, "Park factor — HR, runs, dimensions"),
    ("recent_form", 7, "Recent form / streak (last 10 games)"),
    ("ats_trend", 7, "ATS / run line trend (last 7/14 days)"),
    ("platoon_advantage", 7, "Platoon advantage — batter handedness vs pitcher"),
    ("defensive_metrics", 7, "Defensive metrics — DRS, OAA, error rate"),
    ("weather_conditions", 6, "Weather — wind, temperature, humidity, roof"),
    ("umpire_tendencies", 6, "Home plate umpire — strike zone size, run totals"),
    ("travel_rest", 6, "Travel / rest days — cross-country, day game after night"),
    ("home_away_splits", 6, "Home/away performance splits"),
    ("injuries", 6, "Key injuries — IL, DTD, lineup changes"),
    ("run_line_value", 6, "Run line value — margin of victory trends"),
    ("stolen_base_potential", 5, "Stolen base potential — team speed vs catcher pop time"),
    ("clutch_hitting", 5, "Clutch hitting — RISP, late/close situations"),
    ("manager_tendencies", 5, "Manager tendencies — bullpen usage, lineup decisions"),
    ("line_vs_model", 5, "Our composite vs the line — meta-edge signal"),
]

WNBA_MATRIX = [
    ("star_player_status", 9, "Star player (top 2) out or limited"),
    ("rest_advantage", 9, "B2B, 3-in-5, rest days between games"),
    ("line_movement", 9, "Line movement since open — sharp action, reverse moves"),
    ("off_ranking", 8, "Offensive efficiency rating"),
    ("def_ranking", 8, "Defensive efficiency rating"),
    ("three_pt_matchup", 7, "3PT shooting matchup (attempt rate, %, defense vs 3)"),
    ("sharp_vs_public", 7, "Where the money is (sharp vs public)"),
    ("pace_matchup", 7, "Pace matchup — possessions per game"),
    ("recent_form", 7, "Recent form / streak (last 5-10 games)"),
    ("turnover_gap", 7, "Turnover differential — possession control edge"),
    ("rebounding_edge", 7, "Offensive + defensive rebounding gap"),
    ("h2h_season", 6, "Head-to-head this season (record, margins, trends)"),
    ("paint_scoring", 6, "Points in the paint matchup"),
    ("fast_break", 6, "Fast break / transition points"),
    ("ats_trend", 6, "ATS trend (last 7/14 days)"),
    ("bench_depth", 6, "Bench scoring and depth"),
    ("ft_shooting", 6, "Free throw rate and percentage"),
    ("home_away", 5, "Home/away record"),
    ("travel_fatigue", 5, "Travel distance / schedule congestion"),
    ("line_vs_model", 5, "Our composite vs the line spread — meta-edge signal"),
]

NCAAF_MATRIX = [
    ("sharp_vs_public", 9, "Sharp money vs public betting split"),
    ("line_movement", 9, "Line movement since open — sharp action, reverse moves"),
    ("offensive_efficiency", 8, "Offensive yards/play, points/drive, EPA"),
    ("defensive_efficiency", 8, "Defensive yards/play, points/drive allowed"),
    ("star_player_status", 8, "Key player out or limited — QB, top RB/WR"),
    ("recruiting_rankings", 7, "Recruiting talent — composite rankings, blue chip ratio"),
    ("transfer_portal", 7, "Transfer portal impact — key additions/losses"),
    ("recent_form", 7, "Recent form / streak (last 3-5 games)"),
    ("ats_trend", 7, "ATS trend (last 7/14 days)"),
    ("turnover_margin", 7, "Turnover margin — takeaways vs giveaways"),
    ("rivalry_factor", 7, "Rivalry / motivation — conference title, bowl implications"),
    ("home_crowd", 6, "Home field advantage — stadium capacity, noise factor"),
    ("coaching_matchup", 6, "Coaching matchup — scheme, play-calling tendencies"),
    ("rushing_attack", 6, "Rushing offense vs run defense matchup"),
    ("passing_attack", 6, "Passing offense vs pass defense matchup"),
    ("special_teams", 6, "Special teams — return game, kicking, punt coverage"),
    ("red_zone", 5, "Red zone efficiency — offense and defense"),
    ("weather_conditions", 5, "Weather — wind, rain, cold, altitude"),
    ("rest_days", 5, "Rest days / bye week advantage"),
    ("line_vs_model", 5, "Our composite vs the line spread — meta-edge signal"),
]

TENNIS_MATRIX = [
    ("surface_advantage", 9, "Surface win % — clay, hard, grass specialization"),
    ("h2h_record", 9, "Head-to-head record and recent matchup trends"),
    ("recent_form", 8, "Recent form — wins/losses last 5-10 matches"),
    ("serve_stats", 8, "Serve — ace rate, 1st serve %, hold rate"),
    ("return_game", 8, "Return game — break point conversion, return points won"),
    ("sharp_vs_public", 7, "Sharp money vs public betting split"),
    ("line_movement", 7, "Line movement since open"),
    ("ranking_gap", 7, "ATP/WTA ranking gap — form-adjusted"),
    ("fitness_injury", 7, "Fitness / injury status — MTO, retirement risk"),
    ("break_point_conversion", 7, "Break point conversion rate — clutch factor"),
    ("mental_toughness", 7, "Mental toughness — tiebreak record, deciding sets"),
    ("fatigue_schedule", 6, "Fatigue — tournament schedule, travel, consecutive matches"),
    ("set_handicap_value", 6, "Set handicap value — how tight are sets"),
    ("match_length_trend", 6, "Match length trend — fitness advantage in long matches"),
    ("clutch_performance", 6, "Clutch performance — deciding set record, deuce games"),
    ("surface_win_pct", 6, "Surface-specific win % this season"),
    ("travel_adjustment", 5, "Travel / time zone adjustment"),
    ("weather_conditions", 5, "Weather — wind, heat, indoor/outdoor"),
    ("motivation", 5, "Motivation — ranking points defense, Slam pressure"),
    ("coaching", 5, "Coaching changes / new tactics"),
]

MMA_MATRIX = [
    # Tier 1 — Weight 9
    ("style_matchup", 9, "Style matchup — striker vs grappler, range fighter vs pressure"),
    # Tier 2 — Weight 8
    ("recent_form", 8, "Recent form — wins, finishes, decision losses"),
    ("sharp_vs_public", 8, "Sharp money vs public betting split"),
    ("line_movement", 8, "Line movement since open"),
    ("grappling_advantage", 8, "Grappling — wrestling credentials, submission threat, TDD"),
    ("striking_advantage", 8, "Striking — volume, accuracy, power, defense"),
    # Tier 3 — Weight 7
    ("age_differential", 7, "Age / age gap — fighters over 32 lose 62% of the time, post-34 KO vulnerability spikes"),
    ("cardio_endurance", 7, "Cardio / gas tank — late-round performance"),
    ("chin_durability", 7, "Chin durability — absorption rate, knockdown history"),
    ("takedown_defense", 7, "Takedown defense % — ability to keep fight standing"),
    ("fight_iq", 7, "Fight IQ — cage control, adjustments, game planning"),
    ("octagon_control", 7, "Octagon control — center positioning, cage pressure, control time"),
    ("camp_quality", 7, "Camp quality — training partners, coaching, intel on opponent"),
    ("line_vs_model", 7, "Our composite vs the line — meta-edge signal, THE edge indicator"),
    # Tier 4 — Weight 6
    ("weight_cut", 6, "Weight cut — missed weight history, size at weigh-in"),
    ("injury_history", 6, "Injury history — layoff, surgery, chronic issues"),
    ("experience_gap", 6, "Experience gap — UFC fights, five-round experience"),
    ("motivation", 6, "Motivation — title shot, rivalry, contract fight"),
    # Tier 5 — Weight 5
    ("reach_advantage", 5, "Reach / height advantage — minimal correlation with strike success per research"),
    ("finishing_rate", 5, "Finishing rate — KO/TKO/SUB vs decisions"),
    ("short_notice", 5, "Short notice fight — replacement fighters on <2 weeks notice perform measurably worse"),
    ("method_of_victory_lean", 4, "Method of victory lean — KO, SUB, or DEC value (better as prop signal than fight grade)"),
]

BOXING_MATRIX = [
    # Tier 1 — Weight 9
    ("style_matchup", 9, "Style matchup — counter-puncher vs pressure, southpaw vs orthodox (includes counter-punching as a style)"),
    ("line_vs_model", 9, "Our composite vs the line — meta-edge signal, THE edge indicator. Discrepancy = opportunity"),
    # Tier 2 — Weight 8
    ("recent_form", 8, "Recent form — wins, stoppages, level of opposition"),
    ("sharp_vs_public", 8, "Sharp money vs public betting split"),
    ("line_movement", 8, "Line movement since open"),
    ("punch_output", 8, "Punch output — volume, connect rate, jab effectiveness, body work effectiveness"),
    ("power_advantage", 8, "Power — KO %, one-punch ability"),
    ("age_differential", 8, "Age / age gap — the 38-year-old rule. Post-peak decline is the #1 predictor in boxing research"),
    ("chin_durability", 8, "Chin durability — been hurt/dropped, recovery, absorption history"),
    # Tier 3 — Weight 7
    ("ring_generalship", 7, "Ring generalship — cutting off ring, controlling distance"),
    ("cardio_endurance", 7, "Cardio / gas tank — late-round performance"),
    ("defensive_skill", 7, "Defensive skill — head movement, guard, footwork"),
    ("experience_gap", 7, "Experience gap — rounds fought, championship experience, 12-round fights"),
    ("judges_location", 7, "Judges / location — home advantage, scoring tendencies, sanctioning body"),
    # Tier 4 — Weight 6
    ("camp_quality", 6, "Camp quality — trainer, sparring partners"),
    ("weight_management", 6, "Weight management — rehydration, size advantage"),
    ("injury_history", 6, "Injury history — cuts, hand issues, layoff"),
    ("activity_level", 6, "Activity level / ring rust — time between fights, 12+ month layoff = measurably worse"),
    # Tier 5 — Weight 5
    ("reach_advantage", 5, "Reach / height advantage — range management (less predictive than commonly believed)"),
    ("motivation", 4, "Motivation — title shot, legacy, redemption (narrative noise, traps public bettors)"),
]

SPORT_MATRICES = {
    "nba": NBA_MATRIX, "nhl": NHL_MATRIX, "soccer": SOCCER_MATRIX,
    "ncaab": NCAAB_MATRIX, "mlb": MLB_MATRIX, "wnba": WNBA_MATRIX,
    "ncaaf": NCAAF_MATRIX, "tennis": TENNIS_MATRIX, "mma": MMA_MATRIX,
    "boxing": BOXING_MATRIX,
}

# ============================================================
# CHAIN DETECTION ENGINE — Compound signal bonuses
# ============================================================
# Each chain: name, description, bonus, variables dict {var: (operator, threshold)}
# operator: ">=" means score must be >= threshold, "<=" means score must be <= threshold

CHAINS = {
    "nba": [
        {
            "name": "THE MISPRICING",
            "desc": "Star out + sharp money + line hasn't moved — book is asleep",
            "bonus": 1.0,
            "vars": {"star_player_status": (">=", 8), "line_movement": ("<=", 4), "sharp_vs_public": (">=", 7)},
        },
        {
            "name": "FATIGUE FADE",
            "desc": "Rested team vs tired squad on long road trip with depth issues",
            "bonus": 1.0,
            "vars": {"rest_advantage": (">=", 8), "road_trip_length": (">=", 7), "depth_injuries": (">=", 6)},
        },
        {
            "name": "FORM WAVE",
            "desc": "Hot team firing on all cylinders — offense, ATS, H2H all aligned",
            "bonus": 1.5,
            "vars": {"recent_form": (">=", 8), "off_ranking": (">=", 7), "ats_trend": (">=", 7), "h2h_season": (">=", 6)},
        },
        {
            "name": "TRAP GAME",
            "desc": "Public hammering one side but sharps + line movement say opposite",
            "bonus": 1.0,
            "vars": {"sharp_vs_public": ("<=", 3), "line_movement": (">=", 8), "recent_form": (">=", 7)},
        },
        {
            "name": "VALUE DOG",
            "desc": "Model says line is off + sharps agree + star missing on other side",
            "bonus": 1.0,
            "vars": {"line_vs_model": (">=", 8), "sharp_vs_public": (">=", 7), "star_player_status": ("<=", 4)},
        },
        {
            "name": "SHARP DOG",
            "desc": "Sharps on inflated dog line + strong ATS trend + star out on favorite",
            "bonus": 1.0,
            "vars": {"sharp_vs_public": (">=", 7), "ats_trend": (">=", 7), "star_player_status": (">=", 8)},
        },
    ],
    "nhl": [
        {
            "name": "THE MISPRICING",
            "desc": "Star out + sharp money + line hasn't moved — book is asleep",
            "bonus": 1.0,
            "vars": {"star_player_status": (">=", 8), "line_movement": ("<=", 4), "sharp_vs_public": (">=", 7)},
        },
        {
            "name": "GOALIE EDGE",
            "desc": "Confirmed starter with hot save %, rested, and strong underlying numbers",
            "bonus": 1.5,
            "vars": {"goalie_confirmed": (">=", 8), "save_pct_trend": (">=", 7), "goalie_fatigue": (">=", 7), "corsi_xg": (">=", 7)},
        },
        {
            "name": "FATIGUE FADE",
            "desc": "B2B fatigue + long road trip + depth issues — fade the tired team",
            "bonus": 1.0,
            "vars": {"b2b_fatigue": (">=", 8), "road_trip_length": (">=", 7), "depth_injuries": (">=", 6)},
        },
        {
            "name": "TRAP GAME",
            "desc": "Public on one side but sharps + line movement disagree",
            "bonus": 0.5,
            "vars": {"sharp_vs_public": ("<=", 3), "line_movement": (">=", 8)},
        },
    ],
    "soccer": [
        {
            "name": "THE MISPRICING",
            "desc": "Star out + sharp money + line hasn't moved — book is asleep",
            "bonus": 1.0,
            "vars": {"star_player_status": (">=", 8), "line_movement": ("<=", 4), "sharp_vs_public": (">=", 7)},
        },
        {
            "name": "BTTS LOCK",
            "desc": "Both teams scoring + high goals avg + strong xG + no clean sheets",
            "bonus": 1.5,
            "vars": {"btts_trend": (">=", 8), "goals_avg": (">=", 8), "xg_trend": (">=", 7), "clean_sheet": ("<=", 4)},
        },
        {
            "name": "CONGESTION FADE",
            "desc": "Fixture pile-up + low motivation + depth depleted — fade the tired side",
            "bonus": 1.0,
            "vars": {"fixture_congestion": (">=", 8), "motivation_level": ("<=", 4), "depth_injuries": (">=", 7)},
        },
        {
            "name": "FORM WAVE",
            "desc": "Hot team with xG backing it up — form, ATS, H2H all aligned",
            "bonus": 1.5,
            "vars": {"recent_form": (">=", 8), "xg_trend": (">=", 7), "ats_trend": (">=", 7), "h2h_season": (">=", 6)},
        },
    ],
    "ncaab": [
        {
            "name": "TEMPO TRAP",
            "desc": "Pace mismatch + public on the fast team + 3PT variance",
            "bonus": 1.0,
            "vars": {"tempo_matchup": (">=", 8), "three_pt_reliance": (">=", 7), "public_money_bias": (">=", 7)},
        },
        {
            "name": "TOURNEY FADE",
            "desc": "Tournament fatigue + thin bench + 3rd game in 3 days",
            "bonus": 1.0,
            "vars": {"tournament_context": (">=", 7), "fatigue_schedule": (">=", 7), "bench_depth": ("<=", 4)},
        },
        {
            "name": "FORM WAVE",
            "desc": "Hot team firing — ATS, form, efficiency all aligned heading into tournament",
            "bonus": 1.5,
            "vars": {"recent_form": (">=", 8), "ats_trend": (">=", 7), "offensive_efficiency": (">=", 7)},
        },
        {
            "name": "BLUE BLOOD TRAP",
            "desc": "Public hammering blue blood but line isn't moving and NET gap is small — book shaded for public money",
            "bonus": 1.0,
            "vars": {"public_money_bias": (">=", 8), "line_movement": ("<=", 4), "net_ranking_gap": ("<=", 5)},
        },
        {
            "name": "CINDERELLA SIGNAL",
            "desc": "Lower seed with elite defense and hot recent form — classic tournament upset profile",
            "bonus": 1.0,
            "vars": {"net_ranking_gap": (">=", 7), "recent_form": (">=", 8), "defensive_efficiency": (">=", 7)},
        },
    ],
    "mlb": [
        {
            "name": "ACE MISMATCH",
            "desc": "Elite SP vs weak lineup + park suppresses runs + sharps agree",
            "bonus": 1.5,
            "vars": {"starting_pitcher": (">=", 8), "lineup_vs_hand": ("<=", 4), "park_factor": ("<=", 4)},
        },
        {
            "name": "BULLPEN FADE",
            "desc": "Overworked bullpen + recent form dropping + travel fatigue",
            "bonus": 1.0,
            "vars": {"bullpen_strength": ("<=", 4), "recent_form": ("<=", 4), "travel_rest": (">=", 7)},
        },
        {
            "name": "SHARP STEAM",
            "desc": "Sharp money pouring in + line moving + platoon advantage",
            "bonus": 1.0,
            "vars": {"sharp_vs_public": (">=", 8), "line_movement": (">=", 7), "platoon_advantage": (">=", 7)},
        },
    ],
    "wnba": [
        {
            "name": "FATIGUE FADE",
            "desc": "Rested team vs tired squad with travel + depth issues",
            "bonus": 1.0,
            "vars": {"rest_advantage": (">=", 8), "travel_fatigue": (">=", 7), "bench_depth": ("<=", 4)},
        },
        {
            "name": "FORM WAVE",
            "desc": "Hot team — offense, ATS, H2H all aligned",
            "bonus": 1.5,
            "vars": {"recent_form": (">=", 8), "off_ranking": (">=", 7), "ats_trend": (">=", 7)},
        },
        {
            "name": "THE MISPRICING",
            "desc": "Star out + sharp money + line hasn't moved",
            "bonus": 1.0,
            "vars": {"star_player_status": (">=", 8), "line_movement": ("<=", 4), "sharp_vs_public": (">=", 7)},
        },
    ],
    "ncaaf": [
        {
            "name": "RIVALRY TRAP",
            "desc": "Rivalry game + sharps on underdog + recent ATS trend supports",
            "bonus": 1.0,
            "vars": {"rivalry_factor": (">=", 8), "sharp_vs_public": (">=", 7), "ats_trend": (">=", 7)},
        },
        {
            "name": "TALENT GAP",
            "desc": "Recruiting edge + transfers bolster + defense dominates",
            "bonus": 1.5,
            "vars": {"recruiting_rankings": (">=", 8), "transfer_portal": (">=", 7), "defensive_efficiency": (">=", 7)},
        },
        {
            "name": "FORM WAVE",
            "desc": "Hot team with momentum — offense, ATS, rushing all clicking",
            "bonus": 1.0,
            "vars": {"recent_form": (">=", 8), "ats_trend": (">=", 7), "rushing_attack": (">=", 7)},
        },
    ],
    "tennis": [
        {
            "name": "SURFACE KING",
            "desc": "Surface specialist + strong H2H + serve firing",
            "bonus": 1.5,
            "vars": {"surface_advantage": (">=", 8), "h2h_record": (">=", 7), "serve_stats": (">=", 7)},
        },
        {
            "name": "FATIGUE FADE",
            "desc": "Opponent fatigued + long matches piling up + fitness edge",
            "bonus": 1.0,
            "vars": {"fatigue_schedule": (">=", 7), "match_length_trend": (">=", 7), "fitness_injury": (">=", 7)},
        },
        {
            "name": "CLUTCH MASTER",
            "desc": "Mental toughness + break point conversion + tiebreak record",
            "bonus": 1.0,
            "vars": {"mental_toughness": (">=", 8), "break_point_conversion": (">=", 7), "clutch_performance": (">=", 7)},
        },
    ],
    "mma": [
        {
            "name": "STYLE CLASH",
            "desc": "Dominant style matchup + reach controls range + TDD nullifies grappling",
            "bonus": 1.5,
            "vars": {"style_matchup": (">=", 8), "reach_advantage": (">=", 7), "takedown_defense": (">=", 7)},
        },
        {
            "name": "CARDIO KILLER",
            "desc": "Superior gas tank + fight IQ + opponent fades late",
            "bonus": 1.0,
            "vars": {"cardio_endurance": (">=", 8), "fight_iq": (">=", 7), "chin_durability": (">=", 7)},
        },
        {
            "name": "CAMP EDGE",
            "desc": "Elite camp + sharp money + motivation factor",
            "bonus": 1.0,
            "vars": {"camp_quality": (">=", 7), "sharp_vs_public": (">=", 7), "motivation": (">=", 7)},
        },
    ],
    "boxing": [
        {
            "name": "STYLE CLASH",
            "desc": "Dominant style matchup + reach controls range + power advantage",
            "bonus": 1.5,
            "vars": {"style_matchup": (">=", 8), "reach_advantage": (">=", 7), "power_advantage": (">=", 7)},
        },
        {
            "name": "BODY BREAKER",
            "desc": "Body attack + cardio edge + opponent fades in late rounds",
            "bonus": 1.0,
            "vars": {"body_attack": (">=", 7), "cardio_endurance": (">=", 7), "punch_output": (">=", 7)},
        },
        {
            "name": "RING GENERAL",
            "desc": "Ring generalship + defensive skill + counter-punching = points machine",
            "bonus": 1.0,
            "vars": {"ring_generalship": (">=", 8), "defensive_skill": (">=", 7), "counter_punching": (">=", 7)},
        },
    ],
}


def _detect_chains(game, sport, matrix_scores):
    """Detect compound signal chains from matrix scores. Returns list of triggered chains."""
    sport_chains = CHAINS.get(sport.lower(), [])
    triggered = []
    for chain in sport_chains:
        all_met = True
        var_scores = {}
        for var_name, (op, threshold) in chain["vars"].items():
            score_data = matrix_scores.get(var_name)
            if score_data is None:
                all_met = False
                break
            score = score_data.get("score", 5) if isinstance(score_data, dict) else (score_data if isinstance(score_data, (int, float)) else 5)
            score = max(1, min(10, int(score)))
            if op == ">=" and score < threshold:
                all_met = False
                break
            elif op == "<=" and score > threshold:
                all_met = False
                break
            var_scores[var_name] = score
        if all_met:
            triggered.append({
                "name": chain["name"],
                "desc": chain["desc"],
                "bonus": chain["bonus"],
                "var_scores": var_scores,
            })
            print(f"[CHAIN] {game.get('matchup')} — {chain['name']} triggered (+{chain['bonus']})")
    return triggered


def _recalculate_grade(game, sport):
    """Server-side grade recalculation from matrix scores. Overrides GPT's grade.

    GPT provides the individual variable scores (1-10). We do the math.
    For sports without matrices, GPT's grade stands.
    """
    if game.get("grade") in ("TBD", "INCOMPLETE"):
        return  # Don't override special grades

    matrix = SPORT_MATRICES.get(sport.lower())
    if not matrix:
        return  # No matrix for this sport, GPT's grade stands

    matrix_scores = game.get("matrix_scores", {})
    if not matrix_scores:
        # GPT didn't return matrix scores — fall back to GPT's composite_score for grading
        gpt_score = game.get("composite_score")
        if isinstance(gpt_score, (int, float)) and gpt_score > 0:
            if gpt_score >= 8.5:
                game["grade"] = "A+"
            elif gpt_score >= 7.8:
                game["grade"] = "A"
            elif gpt_score >= 7.0:
                game["grade"] = "A-"
            elif gpt_score >= 6.5:
                game["grade"] = "B+"
            elif gpt_score >= 6.0:
                game["grade"] = "B"
            elif gpt_score >= 5.5:
                game["grade"] = "B-"
            elif gpt_score >= 5.0:
                game["grade"] = "C+"
            elif gpt_score >= 4.0:
                game["grade"] = "C"
            elif gpt_score >= 3.0:
                game["grade"] = "D"
            else:
                game["grade"] = "F"
            game["grade_source"] = "gpt_composite_fallback"
            print(f"[GRADE FALLBACK] {game.get('matchup')} — no matrix_scores, used GPT composite {gpt_score} → {game['grade']}")
        elif not game.get("grade") or game.get("grade") == "--":
            game["grade"] = "N/A"
            game["grade_source"] = "no_data"
            print(f"[GRADE MISSING] {game.get('matchup')} — no matrix_scores, no composite, grade set to N/A")
        return

    # Normalize model's keys to match matrix variable names
    # The formatter may return exact keys, human-readable labels, or paraphrased names
    valid_keys = {name for name, _, _ in matrix}
    label_to_key = {}
    for key, _, label in matrix:
        label_to_key[label.lower()] = key
        # Also map normalized label → key
        norm_label = label.lower().replace(" / ", "_").replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace(",", "")
        label_to_key[norm_label] = key

    def _best_key_match(raw_key):
        """Find the best matching matrix variable key for a model-returned key."""
        # Direct match
        if raw_key in valid_keys:
            return raw_key
        # Normalize
        norm = raw_key.lower().replace(" / ", "_").replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace(",", "")
        # Exact normalized match
        if norm in valid_keys:
            return norm
        # Prefix/suffix match against variable names
        for var_name in valid_keys:
            if norm == var_name or norm.startswith(var_name) or var_name.startswith(norm):
                return var_name
        # Exact label match
        if norm in label_to_key:
            return label_to_key[norm]
        # Word overlap scoring — pick the matrix key whose label shares the most words
        norm_words = set(norm.split("_")) - {"the", "a", "an", "vs", "of", "and", "or", "is", "in", "for"}
        best_match = None
        best_score = 0
        for key, _, label in matrix:
            label_words = set(label.lower().replace("/", " ").replace("-", " ").replace("(", " ").replace(")", " ").replace(",", " ").split()) - {"the", "a", "an", "vs", "of", "and", "or", "is", "in", "for"}
            key_words = set(key.split("_"))
            # Check overlap with both label words and key words
            overlap = len(norm_words & (label_words | key_words))
            if overlap > best_score:
                best_score = overlap
                best_match = key
        if best_score >= 1:
            return best_match
        return None

    normalized_scores = {}
    unmatched = []
    for k, v in matrix_scores.items():
        matched_key = _best_key_match(k)
        if matched_key and matched_key not in normalized_scores:
            normalized_scores[matched_key] = v
        else:
            unmatched.append(k)
    if unmatched:
        print(f"[MATRIX NORM] {game.get('matchup')} — unmatched keys: {unmatched}")
    matrix_scores = normalized_scores

    # Calculate max possible weighted score
    max_possible = sum(w for _, w, _ in matrix) * 10

    # Calculate actual weighted sum from GPT's individual scores
    # Missing variables get default score 5 (neutral) instead of being skipped
    total_weighted = 0
    missing_vars = []
    for var_name, weight, _ in matrix:
        if var_name in matrix_scores:
            score_data = matrix_scores[var_name]
            score = score_data.get("score", 5) if isinstance(score_data, dict) else 5
            score = max(1, min(10, int(score)))  # Clamp 1-10
            weighted = score * weight
            # Override GPT's weighted value with our calculation
            if isinstance(score_data, dict):
                score_data["weighted"] = weighted
            total_weighted += weighted
        else:
            # Default to 5 (neutral) for unmatched variables
            total_weighted += 5 * weight
            missing_vars.append(var_name)
    if missing_vars:
        print(f"[MATRIX DEFAULT] {game.get('matchup')} — {len(missing_vars)} vars defaulted to 5: {missing_vars[:5]}")

    # Recalculate composite
    recalculated = round((total_weighted / max_possible) * 10, 1) if max_possible > 0 else 5.0

    # Log if GPT's score differs significantly
    gpt_score = game.get("composite_score", 0)
    if isinstance(gpt_score, (int, float)) and abs(gpt_score - recalculated) > 0.5:
        print(f"[GRADE OVERRIDE] {game.get('matchup')} — GPT said {gpt_score}, actual is {recalculated}")

    # Override composite_score with our math
    game["composite_score"] = recalculated
    game["gpt_original_score"] = gpt_score

    # Chain detection — compound signal bonuses
    chains_triggered = _detect_chains(game, sport, matrix_scores)
    chain_bonus = min(sum(c["bonus"] for c in chains_triggered), 3.0)
    if chain_bonus > 0:
        game["composite_score_pre_chain"] = recalculated
        game["chain_bonus"] = chain_bonus
        game["chains"] = chains_triggered
        recalculated = min(10.0, recalculated + chain_bonus)
        game["composite_score"] = recalculated
        print(f"[CHAIN BONUS] {game.get('matchup')} — +{chain_bonus} → {recalculated} (was {game['composite_score_pre_chain']})")

    # Apply grade from thresholds — WE decide the grade, not GPT
    gpt_grade = game.get("grade", "")
    if recalculated >= 8.5:
        game["grade"] = "A+"
    elif recalculated >= 7.8:
        game["grade"] = "A"
    elif recalculated >= 7.0:
        game["grade"] = "A-"
    elif recalculated >= 6.5:
        game["grade"] = "B+"
    elif recalculated >= 6.0:
        game["grade"] = "B"
    elif recalculated >= 5.5:
        game["grade"] = "B-"
    elif recalculated >= 5.0:
        game["grade"] = "C+"
    elif recalculated >= 4.0:
        game["grade"] = "C"
    elif recalculated >= 3.0:
        game["grade"] = "D"
    else:
        game["grade"] = "F"

    if gpt_grade and gpt_grade != game["grade"]:
        print(f"[GRADE CHANGED] {game.get('matchup')} — GPT said {gpt_grade}, we say {game['grade']} (score: {recalculated})")

    game["gpt_original_grade"] = gpt_grade
    # Dual grade display: "Matrix: B+ / GPT: A-" side by side
    if gpt_grade and gpt_grade != game["grade"]:
        game["dual_grade"] = f"{game['grade']} / {gpt_grade}"
    else:
        game["dual_grade"] = game["grade"]

    # Check if GPT returned all variables
    expected_vars = [name for name, _, _ in matrix]
    missing = [v for v in expected_vars if v not in matrix_scores]
    if missing:
        game["matrix_warning"] = f"Missing variables: {', '.join(missing)}"
        print(f"[MATRIX INCOMPLETE] {game.get('matchup')} missing {missing}")


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
    lines.append(f"\n=== {sport.upper()} SCORING MATRIX ({len(matrix)} Variables, Weighted 1-10) ===")
    lines.append(f"Score each variable 1-10 for EACH game. Weight x Score = Weighted.")
    lines.append(f"Max possible: {max_score}. Composite = sum / {max_score} x 10.\n")
    lines.append("SCORING RULES — READ CAREFULLY:")
    lines.append("- Use the FULL 1-10 range. Do NOT cluster scores between 5-7.")
    lines.append("- 1-2 = Strong negative signal for this side (e.g., star player OUT, 4th game in 5 nights)")
    lines.append("- 3-4 = Moderate negative")
    lines.append("- 5 = Neutral / no signal either way")
    lines.append("- 6-7 = Moderate positive signal")
    lines.append("- 8-9 = Strong positive signal (e.g., team is 8-2 ATS last 10, opponent on B2B)")
    lines.append("- 10 = Extreme edge (e.g., MVP-level player out and line barely moved)")
    lines.append("If a variable has a clear strong signal, score it 8-10. If it clearly hurts the pick, score it 1-3.")
    lines.append("The point of this matrix is to FIND EDGE — not to hedge everything to a C+.")
    lines.append("A game where a star is out, the line hasn't moved, and sharp money disagrees with the public should NOT score 6-7 across the board.")
    lines.append("BE DECISIVE.\n")
    for i, (key, weight, label) in enumerate(matrix, 1):
        lines.append(f"  {i}. KEY: \"{key}\" — {label} (weight: {weight})")
    lines.append("\nCOMPOSITE THRESHOLDS:")
    lines.append("  9.0-10.0 = BEST BET (load up, full unit)")
    lines.append("  7.5-8.9  = STRONG PLAY (A-/B+)")
    lines.append("  6.0-7.4  = MODERATE EDGE (B/B-)")
    lines.append("  4.5-5.9  = LEAN (C, small or pass)")
    lines.append("  Below 4.5 = NO EDGE (D/F, pass)")
    return "\n".join(lines)


# ============================================================
# USER AGENT PROFILES — Per-user weighted matrix customization
# ============================================================

USER_AGENTS_DIR = os.path.join(DATA_DIR, "user_agents")
os.makedirs(USER_AGENTS_DIR, exist_ok=True)

# Rate limit: max custom analysis runs per sport per day per user
_user_analysis_counts = {}  # key: f"{user_id}:{sport}:{date}" -> count
MAX_USER_ANALYSIS_PER_SPORT_PER_DAY = 3


def _user_profile_path(user_id):
    """Safe path for a user agent profile."""
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', user_id)
    if not safe_id:
        return None
    return os.path.join(USER_AGENTS_DIR, f"{safe_id}.json")


def _load_user_profile(user_id):
    """Load a user agent profile from disk. Returns None if not found."""
    path = _user_profile_path(user_id)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _save_user_profile(profile):
    """Save a user agent profile to disk."""
    user_id = profile.get("user_id")
    if not user_id:
        return False
    path = _user_profile_path(user_id)
    if not path:
        return False
    profile["updated_at"] = _now_ts()
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    return True


def _resolve_user_matrix(sport, profile):
    """Resolve a user's matrix: defaults + overrides + custom_vars - exclusions.
    Returns list of (key, weight, description) tuples."""
    default_matrix = SPORT_MATRICES.get(sport.lower(), [])
    if not default_matrix:
        return []

    matrices = profile.get("matrices", {})
    sport_config = matrices.get(sport.lower(), {})
    overrides = sport_config.get("overrides", {})
    excluded = set(sport_config.get("excluded", []))
    custom_vars = sport_config.get("custom_vars", [])

    # Start with defaults, apply overrides and exclusions
    resolved = []
    for key, weight, desc in default_matrix:
        if key in excluded:
            continue
        if key in overrides:
            new_weight = overrides[key].get("weight", weight)
            note = overrides[key].get("note", "")
            desc_with_note = f"{desc} [User override: {note}]" if note else desc
            resolved.append((key, new_weight, desc_with_note))
        else:
            resolved.append((key, weight, desc))

    # Append custom variables (max 5)
    for cv in custom_vars[:5]:
        resolved.append((cv["key"], cv.get("weight", 5), cv.get("description", cv["key"])))

    return resolved


def _build_user_matrix_section(sport, profile):
    """Build matrix section text using a user's resolved matrix."""
    resolved = _resolve_user_matrix(sport, profile)
    if not resolved:
        return _build_matrix_section(sport)

    total_weight = sum(w for _, w, _ in resolved)
    max_score = total_weight * 10
    display_name = profile.get("display_name", profile.get("user_id", "User"))

    lines = []
    lines.append(f"\n=== {sport.upper()} SCORING MATRIX — {display_name}'s Custom Weights ({len(resolved)} Variables) ===")
    lines.append(f"Score each variable 1-10 for EACH game. Weight x Score = Weighted.")
    lines.append(f"Max possible: {max_score}. Composite = sum / {max_score} x 10.\n")
    lines.append("SCORING RULES — READ CAREFULLY:")
    lines.append("- Use the FULL 1-10 range. Do NOT cluster scores between 5-7.")
    lines.append("- 1-2 = Strong negative signal for this side")
    lines.append("- 3-4 = Moderate negative")
    lines.append("- 5 = Neutral / no signal either way")
    lines.append("- 6-7 = Moderate positive signal")
    lines.append("- 8-9 = Strong positive signal")
    lines.append("- 10 = Extreme edge")
    lines.append("BE DECISIVE.\n")
    for i, (key, weight, label) in enumerate(resolved, 1):
        lines.append(f"  {i}. KEY: \"{key}\" — {label} (weight: {weight})")
    lines.append("\nCOMPOSITE THRESHOLDS:")
    lines.append("  9.0-10.0 = BEST BET (load up, full unit)")
    lines.append("  7.5-8.9  = STRONG PLAY (A-/B+)")
    lines.append("  6.0-7.4  = MODERATE EDGE (B/B-)")
    lines.append("  4.5-5.9  = LEAN (C, small or pass)")
    lines.append("  Below 4.5 = NO EDGE (D/F, pass)")
    return "\n".join(lines)


def _user_analysis_cache_path(sport, user_id, date_str=None):
    """Cache path for user-specific analysis."""
    if not date_str:
        date_str = datetime.now(PST).strftime("%Y-%m-%d")
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', user_id)
    return os.path.join(ANALYSIS_DIR, f"analysis_{sport}_{date_str}_{safe_id}.json")


def _check_user_rate_limit(user_id, sport):
    """Check if user has exceeded custom analysis rate limit. Returns True if OK."""
    date_str = datetime.now(PST).strftime("%Y-%m-%d")
    key = f"{user_id}:{sport}:{date_str}"
    count = _user_analysis_counts.get(key, 0)
    return count < MAX_USER_ANALYSIS_PER_SPORT_PER_DAY


def _increment_user_rate_limit(user_id, sport):
    """Increment the user's analysis count for today."""
    date_str = datetime.now(PST).strftime("%Y-%m-%d")
    key = f"{user_id}:{sport}:{date_str}"
    _user_analysis_counts[key] = _user_analysis_counts.get(key, 0) + 1


# Voice agent tool definitions for realtime session
AGENT_VOICE_TOOLS = [
    {
        "type": "function",
        "name": "update_weight",
        "description": "Update the weight of a scoring variable in the user's matrix. Example: 'bump venue to 8' or 'set home_court weight to 3'.",
        "parameters": {
            "type": "object",
            "properties": {
                "sport": {"type": "string", "description": "Sport code (ncaab, nba, nhl, mlb, etc). Infer from conversation context."},
                "variable": {"type": "string", "description": "Variable key name from the matrix (e.g. home_court, star_player_status)"},
                "weight": {"type": "number", "description": "New weight value (1-10)"},
                "note": {"type": "string", "description": "Brief reason for the override"}
            },
            "required": ["sport", "variable", "weight"]
        }
    },
    {
        "type": "function",
        "name": "add_custom_variable",
        "description": "Add a custom scoring variable to the user's matrix. Max 5 custom variables per sport. Example: 'add tight rims as a factor, weight 7'.",
        "parameters": {
            "type": "object",
            "properties": {
                "sport": {"type": "string", "description": "Sport code"},
                "key": {"type": "string", "description": "Variable key (snake_case, e.g. rim_tightness)"},
                "weight": {"type": "number", "description": "Weight value (1-10)"},
                "description": {"type": "string", "description": "What this variable measures"}
            },
            "required": ["sport", "key", "weight", "description"]
        }
    },
    {
        "type": "function",
        "name": "exclude_variable",
        "description": "Exclude a default variable from the user's matrix. Example: 'I don't care about rivalry'.",
        "parameters": {
            "type": "object",
            "properties": {
                "sport": {"type": "string", "description": "Sport code"},
                "variable": {"type": "string", "description": "Variable key to exclude"}
            },
            "required": ["sport", "variable"]
        }
    },
    {
        "type": "function",
        "name": "show_my_matrix",
        "description": "Show the user their current resolved matrix with all overrides, custom variables, and exclusions.",
        "parameters": {
            "type": "object",
            "properties": {
                "sport": {"type": "string", "description": "Sport code"}
            },
            "required": ["sport"]
        }
    },
    {
        "type": "function",
        "name": "reset_to_default",
        "description": "Reset a sport's matrix back to default, removing all overrides, custom variables, and exclusions.",
        "parameters": {
            "type": "object",
            "properties": {
                "sport": {"type": "string", "description": "Sport code to reset"}
            },
            "required": ["sport"]
        }
    }
]


def _process_agent_action(user_id, action_name, args):
    """Process a voice agent tool call. Returns result dict."""
    profile = _load_user_profile(user_id)
    if not profile:
        return {"error": f"Profile not found for {user_id}"}

    if "matrices" not in profile:
        profile["matrices"] = {}

    sport = args.get("sport", "").lower()
    if sport and sport not in SPORT_MATRICES:
        return {"error": f"Unknown sport: {sport}. Valid: {', '.join(SPORT_MATRICES.keys())}"}

    if sport and sport not in profile["matrices"]:
        profile["matrices"][sport] = {"overrides": {}, "excluded": [], "custom_vars": []}

    sport_cfg = profile["matrices"].get(sport, {}) if sport else {}

    if action_name == "update_weight":
        variable = args.get("variable", "")
        weight = args.get("weight", 5)
        note = args.get("note", "")
        # Validate variable exists in default matrix
        default_keys = [k for k, _, _ in SPORT_MATRICES.get(sport, [])]
        if variable not in default_keys:
            return {"error": f"Variable '{variable}' not found in {sport.upper()} matrix. Available: {', '.join(default_keys)}"}
        weight = max(1, min(10, int(weight)))
        if "overrides" not in sport_cfg:
            sport_cfg["overrides"] = {}
        # Find default weight
        default_weight = next((w for k, w, _ in SPORT_MATRICES[sport] if k == variable), 5)
        sport_cfg["overrides"][variable] = {"weight": weight, "note": note}
        profile["matrices"][sport] = sport_cfg
        _save_user_profile(profile)
        return {"success": True, "message": f"{variable} weight set to {weight} for {sport.upper()} (was {default_weight})"}

    elif action_name == "add_custom_variable":
        key = args.get("key", "").lower().replace(" ", "_")
        weight = max(1, min(10, int(args.get("weight", 5))))
        description = args.get("description", key)
        if "custom_vars" not in sport_cfg:
            sport_cfg["custom_vars"] = []
        if len(sport_cfg["custom_vars"]) >= 5:
            return {"error": "Max 5 custom variables per sport. Remove one first."}
        # Check for duplicate
        existing_keys = [cv["key"] for cv in sport_cfg["custom_vars"]]
        if key in existing_keys:
            # Update existing
            for cv in sport_cfg["custom_vars"]:
                if cv["key"] == key:
                    cv["weight"] = weight
                    cv["description"] = description
            profile["matrices"][sport] = sport_cfg
            _save_user_profile(profile)
            return {"success": True, "message": f"Updated custom variable '{key}' to weight {weight}"}
        sport_cfg["custom_vars"].append({"key": key, "weight": weight, "description": description})
        profile["matrices"][sport] = sport_cfg
        _save_user_profile(profile)
        return {"success": True, "message": f"Added custom variable '{key}' with weight {weight} to {sport.upper()}"}

    elif action_name == "exclude_variable":
        variable = args.get("variable", "")
        default_keys = [k for k, _, _ in SPORT_MATRICES.get(sport, [])]
        if variable not in default_keys:
            return {"error": f"Variable '{variable}' not found in {sport.upper()} matrix."}
        if "excluded" not in sport_cfg:
            sport_cfg["excluded"] = []
        if variable not in sport_cfg["excluded"]:
            sport_cfg["excluded"].append(variable)
        profile["matrices"][sport] = sport_cfg
        _save_user_profile(profile)
        return {"success": True, "message": f"Excluded '{variable}' from {sport.upper()} matrix"}

    elif action_name == "show_my_matrix":
        resolved = _resolve_user_matrix(sport, profile)
        matrix_text = []
        default_dict = {k: w for k, w, _ in SPORT_MATRICES.get(sport, [])}
        overrides = sport_cfg.get("overrides", {})
        excluded = set(sport_cfg.get("excluded", []))
        custom_keys = {cv["key"] for cv in sport_cfg.get("custom_vars", [])}

        for key, weight, desc in resolved:
            tag = ""
            if key in custom_keys:
                tag = " (CUSTOM)"
            elif key in overrides:
                orig = default_dict.get(key, "?")
                tag = f" (was {orig})"
            matrix_text.append(f"{key}: {weight}{tag}")

        if excluded:
            matrix_text.append(f"\nExcluded: {', '.join(excluded)}")

        return {"success": True, "matrix": matrix_text, "sport": sport.upper(),
                "message": f"{sport.upper()} matrix for {profile.get('display_name', user_id)}: " + ", ".join([f"{k}={w}" for k, w, _ in resolved[:5]]) + "..."}

    elif action_name == "reset_to_default":
        profile["matrices"][sport] = {"overrides": {}, "excluded": [], "custom_vars": []}
        _save_user_profile(profile)
        return {"success": True, "message": f"Reset {sport.upper()} matrix to defaults"}

    return {"error": f"Unknown action: {action_name}"}


def _build_agent_voice_prompt(profile):
    """Build personalized voice prompt for a user's agent."""
    display_name = profile.get("display_name", profile.get("user_id", "User"))
    sports = profile.get("sport_preferences", [])
    style = profile.get("betting_style", {})
    learned = profile.get("conversation_context", {}).get("learned_preferences", [])

    sports_str = ", ".join(s.upper() for s in sports) if sports else "all sports"
    style_notes = style.get("notes", "")
    periods = ", ".join(style.get("periods", [])) if style.get("periods") else "full game"
    strategy = ", ".join(style.get("strategy", [])) if style.get("strategy") else "standard"

    # Build matrix summary for context
    matrix_summary = []
    for sport in sports[:3]:
        resolved = _resolve_user_matrix(sport, profile)
        default_dict = {k: w for k, w, _ in SPORT_MATRICES.get(sport, [])}
        matrices_cfg = profile.get("matrices", {}).get(sport, {})
        overrides = matrices_cfg.get("overrides", {})
        custom_vars = matrices_cfg.get("custom_vars", [])
        excluded = matrices_cfg.get("excluded", [])
        changes = []
        for k, v in overrides.items():
            orig = default_dict.get(k, "?")
            changes.append(f"{k}={v['weight']} (was {orig})")
        for cv in custom_vars:
            changes.append(f"+{cv['key']}={cv.get('weight', 5)}")
        for ex in excluded:
            changes.append(f"-{ex}")
        if changes:
            matrix_summary.append(f"{sport.upper()}: {', '.join(changes)}")

    learned_str = "\n".join(f"- {l}" for l in learned) if learned else "None yet"
    matrix_str = "\n".join(matrix_summary) if matrix_summary else "All defaults"

    # Check if this is a new user (no overrides, no custom vars, no learned prefs)
    has_customizations = bool(matrix_summary) or bool(learned)

    if has_customizations:
        onboarding_block = f"""Current matrix changes:
{matrix_str}

Things I've learned about {display_name}:
{learned_str}"""
    else:
        onboarding_block = f"""This is a NEW user with NO customizations yet. Guide them through onboarding:

ONBOARDING FLOW (use this when they first connect or ask "how does this work"):
1. Explain: "We grade every game using a 20-variable weighted matrix — things like injuries, rest, sharp money, matchup data. Each variable gets a weight from 1-10 based on how much it matters."
2. Ask: "Want me to walk you through the variables for [their sport] so you can set your own weights? We can go one by one."
3. If yes: Read each variable name and its default weight. Ask if they want to keep it, bump it, drop it, or remove it entirely. Use the tools to make changes as you go.
4. After finishing: "Your custom matrix is saved. Hit 'Analyze DJ's Way' on the page to see how your grades compare to the defaults."

GUIDED RESPONSES for common first-time questions:
- "Show me the slate" / "What games are on?" → Explain they can see the slate on the page above. List the sport they're looking at and mention the games are graded A through F.
- "How does grading work?" → Explain the 20-variable weighted matrix. Each game scored 1-10 per variable, multiplied by weight, summed, normalized to a composite 0-10. A = strong edge, F = trap.
- "What props look good?" → Explain the screener: we pull every player's last 5 game logs, compare their floor to the book's line. Floor beats line = mispriced.
- "Run demo" → Walk them through the full system: matrix, grading, screener, multi-book lines, then offer to build their custom matrix."""

    return f"""You are {display_name}'s personal Edge Agent for Edge Crew — a sports betting analytics platform. You help them understand the system, find edges, and build their own custom scoring matrix.

{display_name}'s profile:
- Sports: {sports_str}
- Periods: {periods}
- Strategy: {strategy}
- Notes: {style_notes}

{onboarding_block}

CAPABILITIES — what you can do:
1. EXPLAIN the system: how grading works, what each variable means, how the screener finds mispriced props, how multi-book comparison works
2. CUSTOMIZE the matrix: update weights (1-10), add custom variables (max 5 per sport), exclude variables, show current matrix, reset to defaults
3. GUIDE through onboarding: walk through variables one by one, build their personal grading sheet
4. ANSWER questions about tonight's slate, best plays, what to look for

RULES:
- Always confirm changes back to the user after making them
- Keep responses under 4 sentences unless walking through a guided flow
- Be sharp, direct, confident — you're a betting analyst, not a customer service bot
- When they ask to change a weight, infer the sport from context. If ambiguous, ask which sport
- Use the tools provided to make changes — never just say you'll do it, actually call the function
- When they say "show me the slate" or "what games are on" — describe what's on the page, you don't have live game data in this session but the page does"""


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
        # Soccer: ML is enough (spreads rare, totals often missing)
        elif sport_label == "SOCCER" and has_ml:
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
    # Extract league metadata from SOCCER_LEAGUES registry
    event_sport_key = event.get("sport_key", "")
    league_meta = SOCCER_LEAGUES.get(event_sport_key, {})

    game = {
        "id": event["id"],
        "sport": sport_label,
        "sport_key": event_sport_key,
        "league": league_meta.get("name", event.get("sport_title", "")),
        "league_flag": league_meta.get("flag", ""),
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

    # Soccer fallback: if no US book has odds, use first available book (e.g. bet365, pinnacle)
    if game["bookmaker"] is None and bookmakers_data:
        for bk in event.get("bookmakers", []):
            markets = {}
            for market in bk.get("markets", []):
                markets[market["key"]] = market["outcomes"]
            if markets:
                game["bookmaker"] = bk["key"]
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
    else:
        game["total"] = None

    if h2h:
        has_ml = True
        for h in h2h:
            if h["name"] == event["away_team"]:
                game["away_ml"] = h.get("price", 0)
            elif h["name"] == event["home_team"]:
                game["home_ml"] = h.get("price", 0)

    # BTTS (Both Teams to Score) — soccer
    btts = game["markets"].get("btts", [])
    if btts:
        for b in btts:
            if b["name"] == "Yes":
                game["btts_yes"] = b.get("price", 0)
            elif b["name"] == "No":
                game["btts_no"] = b.get("price", 0)

    # Draw No Bet — soccer
    dnb = game["markets"].get("draw_no_bet", [])
    if dnb:
        for d in dnb:
            if d["name"] == event["away_team"]:
                game["dnb_away"] = d.get("price", 0)
            elif d["name"] == event["home_team"]:
                game["dnb_home"] = d.get("price", 0)

    # Draw ML (3-way h2h) — soccer
    if h2h:
        for h in h2h:
            if h["name"] == "Draw":
                game["draw_ml"] = h.get("price", 0)

    game["lines_available"] = has_spread or has_total or has_ml
    # Any sport with ML data is considered "complete" for grading
    game["lines_complete"] = has_ml

    # Track line shifts (opening vs current)
    shifts = _track_opening_lines(game)
    if shifts:
        game["shifts"] = shifts

    # Detect arbitrage across bookmakers
    arbs = _detect_arbitrage(event, sport_label)
    if arbs:
        game["arbs"] = arbs

    # Store ALL bookmakers data for multi-book comparison (DJ page)
    all_books = []
    for bk in event.get("bookmakers", []):
        book_entry = {"key": bk.get("key", ""), "title": bk.get("title", bk.get("key", "")), "markets": {}}
        for market in bk.get("markets", []):
            book_entry["markets"][market["key"]] = market["outcomes"]
        if book_entry["markets"]:
            all_books.append(book_entry)
    game["all_bookmakers"] = all_books

    return game


SPORT_KEYS = {
    "nba": ["basketball_nba"],
    "wnba": ["basketball_wnba"],
    "ncaab": ["basketball_ncaab"],
    "ncaaf": ["americanfootball_ncaaf"],
    "nhl": ["icehockey_nhl"],
    "nfl": ["americanfootball_nfl"],
    "mlb": ["baseball_mlb", "baseball_mlb_preseason"],
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
    "soccer": SPORT_KEYS_SOCCER,
}


_active_baseball_cache = {"keys": [], "fetched_at": 0}


async def _get_active_baseball_keys():
    """Fetch currently active baseball keys from The Odds API (MLB, WBC — no college). Cached 1 hour."""
    now = time.time()
    if _active_baseball_cache["keys"] and now - _active_baseball_cache["fetched_at"] < 3600:
        return _active_baseball_cache["keys"]
    try:
        url = f"{ODDS_API_BASE}/"
        params = {"apiKey": ODDS_API_KEY}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            if r.status_code == 200:
                sports = r.json()
                # Only MLB and WBC keys — exclude college baseball (ncaa, college, etc)
                exclude = ("ncaa", "college", "university")
                keys = [
                    s["key"] for s in sports
                    if s["key"].startswith("baseball_") and s.get("active")
                    and not any(x in s["key"].lower() for x in exclude)
                ]
                if keys:
                    _active_baseball_cache["keys"] = keys
                    _active_baseball_cache["fetched_at"] = now
                    return keys
    except Exception:
        pass
    return SPORT_KEYS.get("mlb", ["baseball_mlb"])  # fallback to hardcoded


_wbc_cache = {"games": [], "fetched_at": 0}


async def _fetch_wbc_games():
    """Fetch World Baseball Classic games from ESPN. Cached 15 min."""
    now = time.time()
    if _wbc_cache["games"] and now - _wbc_cache["fetched_at"] < 900:
        return _wbc_cache["games"]
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/baseball/world-baseball-classic/scoreboard"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            if r.status_code == 200:
                data = r.json()
                games = []
                for event in data.get("events", []):
                    comp = event.get("competitions", [{}])[0]
                    competitors = comp.get("competitors", [])
                    if len(competitors) < 2:
                        continue
                    away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[0])
                    home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[1])
                    away_name = away.get("team", {}).get("displayName", "")
                    home_name = home.get("team", {}).get("displayName", "")
                    game_time = event.get("date", "")
                    status = comp.get("status", {}).get("type", {}).get("description", "")
                    away_score = away.get("score", "0")
                    home_score = home.get("score", "0")
                    game = {
                        "away": away_name,
                        "home": home_name,
                        "time": game_time,
                        "sport": "MLB",
                        "league": "WBC",
                        "away_ml": "",
                        "home_ml": "",
                        "home_spread": None,
                        "away_spread": None,
                        "total": None,
                        "bookmaker": "ESPN",
                        "lines_complete": False,
                        "lines_available": False,
                        "fetched_at": _now_ts(),
                        "wbc": True,
                        "status": status,
                        "away_score": away_score,
                        "home_score": home_score,
                    }
                    games.append(game)
                _wbc_cache["games"] = games
                _wbc_cache["fetched_at"] = now
                return games
    except Exception as e:
        logger.warning(f"WBC fetch failed: {e}")
    return _wbc_cache.get("games", [])


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
    # Soccer gets extra markets: BTTS, draw no bet
    if sport_lower == "soccer" and markets == "h2h,spreads,totals":
        markets = "h2h,spreads,totals,btts,draw_no_bet"
    cache_key = f"{sport_lower}:{markets}"
    cached = _get_cached(cache_key)
    if cached:
        # Filter out games that have started since cache was built
        now = datetime.now(PST)
        if cached.get("games"):
            cached["games"] = [g for g in cached["games"] if _game_not_started(g, now)]
            cached["count"] = len(cached["games"])
        return JSONResponse(cached)

    label = sport.upper()
    all_games = []
    source_name = ""

    # --- PRIMARY: The Odds API ---
    if ODDS_API_KEY:
        if sport_lower == "tennis":
            keys = await _get_active_tennis_keys()
        elif sport_lower == "mlb":
            keys = await _get_active_baseball_keys()
        else:
            keys = SPORT_KEYS.get(sport_lower, [sport_lower])
        for key in keys:
            games = await _fetch_sport_odds(key, markets, label)
            all_games.extend(games)
        if all_games:
            source_name = "The Odds API"

    # --- WBC: Merge World Baseball Classic from ESPN ---
    if sport_lower == "mlb":
        try:
            wbc_games = await _fetch_wbc_games()
            if wbc_games:
                all_games.extend(wbc_games)
                logger.info(f"[WBC] Added {len(wbc_games)} World Baseball Classic games")
        except Exception as e:
            logger.warning(f"WBC merge failed: {e}")

    # --- FALLBACK: SharpAPI ---
    if not all_games and SHARPAPI_KEY and sport_lower in SHARPAPI_LEAGUES:
        all_games = await _fetch_sharpapi_odds(sport_lower, label)
        if all_games:
            source_name = "SharpAPI (DraftKings)"

    if not all_games and not SHARPAPI_KEY and not ODDS_API_KEY:
        return JSONResponse({"error": "No odds API configured (set SHARPAPI_KEY or ODDS_API_KEY)"}, status_code=500)

    # Filter: only future games within 24h window
    now = datetime.now(PST)
    cutoff = now + timedelta(hours=24)
    filtered_games = []
    for g in all_games:
        game_time_str = g.get('time', '')
        has_book = g.get('bookmaker') not in (None, 'None', '')
        is_wbc = g.get('wbc', False)
        if not game_time_str or (not has_book and not is_wbc):
            continue
        try:
            gt = datetime.fromisoformat(game_time_str.replace('Z', '+00:00')).astimezone(PST)
            if gt > now and gt <= cutoff:
                filtered_games.append(g)
        except Exception:
            continue
    all_games = filtered_games

    # Deduplicate games by matchup (away + home)
    seen_matchups = set()
    deduped = []
    for g in all_games:
        key = (g.get("away", "").strip().lower(), g.get("home", "").strip().lower())
        if key not in seen_matchups:
            seen_matchups.add(key)
            deduped.append(g)
    all_games = deduped

    # Save to daily record + invalidate stale analysis cache
    if all_games:
        _save_daily_slate(sport_lower, all_games)
        # Clear analysis caches so next analysis uses fresh odds
        analysis_key = f"analysis:{sport_lower}"
        if analysis_key in _cache:
            del _cache[analysis_key]

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


# ===== SOCCER HUB ENDPOINTS =====

@app.get("/api/soccer/leagues")
async def get_soccer_leagues():
    """Return full soccer league registry with live game counts."""
    slate = _load_daily_slate("soccer")
    counts = {}
    if slate and slate.get("games"):
        for g in slate["games"]:
            sk = g.get("sport_key", "")
            counts[sk] = counts.get(sk, 0) + 1
    return JSONResponse([
        {"key": k, **v, "game_count": counts.get(k, 0)}
        for k, v in SOCCER_LEAGUES.items()
    ])


@app.get("/api/soccer/high-edge")
async def get_soccer_high_edge():
    """Return only B+ or above soccer games from analysis cache."""
    HIGH_GRADES = {"A+", "A", "A-", "B+"}
    analysis = _load_analysis_cache("soccer")
    results = []
    if analysis and analysis.get("games"):
        for game in analysis["games"]:
            if game.get("grade") in HIGH_GRADES:
                results.append(game)
        # Sort by composite_score descending
        results.sort(key=lambda g: g.get("composite_score", 0), reverse=True)
    if not results:
        # Fallback: show first 5 upcoming games from slate with "analysis pending" note
        slate = _load_daily_slate("soccer")
        if slate and slate.get("games"):
            now = datetime.now(PST)
            upcoming = sorted(
                [g for g in slate["games"] if g.get("time")],
                key=lambda g: g.get("time", "")
            )[:5]
            for g in upcoming:
                g["grade"] = "PENDING"
                g["analysis_note"] = "Analysis pending — check back after next slate refresh"
                results.append(g)
    return JSONResponse({"games": results, "count": len(results), "filter": "high-edge"})


@app.get("/api/odds/soccer/league/{league_key}")
async def get_soccer_league_odds(league_key: str):
    """On-demand fetch for a specific soccer league (Tier 3 support)."""
    if league_key not in SOCCER_LEAGUES:
        return JSONResponse({"error": f"Unknown league: {league_key}"}, status_code=400)

    cache_key = f"soccer_league:{league_key}"
    cached = _get_cached(cache_key)
    if cached:
        return JSONResponse(cached)

    markets = "h2h,spreads,totals,btts,draw_no_bet"
    label = "SOCCER"
    games = await _fetch_sport_odds(league_key, markets, label)

    # Filter: only future games within 24h
    now = datetime.now(PST)
    cutoff = now + timedelta(hours=24)
    filtered = []
    for g in games:
        t = g.get("time", "")
        if not t:
            continue
        try:
            gt = datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(PST)
            if now < gt <= cutoff:
                filtered.append(g)
        except Exception:
            continue

    league_meta = SOCCER_LEAGUES[league_key]
    result = {
        "league": league_meta["name"],
        "league_key": league_key,
        "games": filtered,
        "count": len(filtered),
        "fetched_at": _now_ts(),
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


@app.get("/api/slate/cached")
async def get_cached_slate():
    now = datetime.now(PST)
    all_games = []
    for sport in ALL_SPORTS:
        slate = _load_daily_slate(sport)
        if slate and slate.get("games"):
            all_games.extend([g for g in slate["games"] if _game_not_started(g, now)])
    return JSONResponse({"games": all_games, "count": len(all_games), "source": "daily_record", "fetched_at": _now_ts(), "cached": True})

@app.get("/api/slate/cached/{sport}")
async def get_cached_sport_slate(sport: str):
    slate = _load_daily_slate(sport.lower())
    if slate and slate.get("games"):
        now = datetime.now(PST)
        slate["games"] = [g for g in slate["games"] if _game_not_started(g, now)]
        if slate["games"]:
            slate["cached"] = True
            slate["count"] = len(slate["games"])
            return JSONResponse(slate)
    return await get_odds(sport)


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
async def get_player_props(sport: str, game: str = None):
    """Fetch real player prop lines from The Odds API with edge analysis.
    Optional game filter: ?game=event_id to show only one matchup."""
    sport_lower = sport.lower()
    if sport_lower not in PROP_MARKETS:
        return JSONResponse({"sport": sport.upper(), "props": [], "count": 0, "fetched_at": _now_ts(), "message": f"No prop markets configured for {sport}"})

    cache_key = f"props:{sport_lower}"
    cached = _get_cached(cache_key, ttl=PROPS_CACHE_TTL)
    if cached:
        return JSONResponse(cached)

    if not ODDS_API_KEY:
        return JSONResponse({"error": "No ODDS_API_KEY configured"}, status_code=500)

    # Pre-fetch rosters for cross-check (if not cached)
    if not _get_cached(f"rosters:{sport_lower}", ttl=3600):
        try:
            await get_rosters(sport)
        except Exception:
            pass

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

                # Fetch props for each event (limit to 12 games to cover full slate)
                for event in events[:12]:
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

    # ===== ROSTER CROSS-CHECK: Flag stale team assignments =====
    roster_cache = _get_cached(f"rosters:{sport_lower}", ttl=3600)
    if roster_cache:
        # Build player -> team lookup from ESPN rosters
        player_team_map = {}  # "jimmy butler iii" -> "Golden State Warriors"
        for abbr, info in roster_cache.get("rosters", {}).items():
            team_name = info.get("team", "")
            for p in info.get("players", []):
                pname = p.get("name", "").lower().strip()
                if pname:
                    player_team_map[pname] = team_name
        # Cross-check each prop
        for prop in all_props:
            pname = prop.get("player", "").lower().strip()
            matchup = prop.get("matchup", "")
            actual_team = player_team_map.get(pname)
            if actual_team:
                prop["actual_team"] = actual_team
                # Check if actual team is in the matchup string
                if actual_team.lower() not in matchup.lower():
                    # Player is listed under wrong game — stale bookmaker data
                    prop["roster_flag"] = f"ROSTER: {prop['player']} is on {actual_team}"
                    prop["matchup"] = f"{matchup} [{prop['player']} now on {actual_team}]"

    # ===== FILTER: Remove garbage props =====
    _odd_even_re = re.compile(r'\b(odd|even)\b', re.IGNORECASE)
    filtered_props = []
    for prop in all_props:
        stat = prop.get("stat", "")
        # Remove even/odd props
        if _odd_even_re.search(stat):
            continue
        # Game filter — only show props for a specific event
        if game and prop.get("event_id") and prop.get("event_id") != game:
            continue
        filtered_props.append(prop)

    # Quality gate: filter out low-production player props
    quality_filtered = []
    for prop in filtered_props:
        stat_lower = prop.get("stat", "").lower()
        line = prop.get("line", 0)
        # NHL: skip goal props for players with line < 0.5 (averaging < 0.3 goals/game)
        if sport_lower == "nhl" and "goal" in stat_lower and line is not None and float(line) < 0.5:
            continue
        # Soccer: skip assist props for players with line < 0.5
        if sport_lower == "soccer" and "assist" in stat_lower and line is not None and float(line) < 0.5:
            continue
        quality_filtered.append(prop)
    all_props = quality_filtered

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
    """Alyssa's Edge — Math + Lineups. No AI. Pure Edge.

    Contrarian value analysis calculated from odds + injury/roster data:
    - Public fade: flag heavy favorites where ML implies >70% implied prob
    - Line value: identify spreads that look off
    - Upset score: underdog value rating 1-10 (adjusted for injuries)
    - Sharp signals: when line movement doesn't match public direction
    - Injury impact: star players OUT shift the edge math
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

    # ===== FETCH INJURY + ROSTER DATA =====
    injuries_by_team = {}
    try:
        inj_cache = _get_cached(f"frontend_injuries:{sport_lower}", ttl=1800)
        if not inj_cache:
            inj_resp = await get_lineups(sport)
            if hasattr(inj_resp, 'body'):
                inj_cache = json.loads(inj_resp.body)
        if inj_cache:
            injuries_by_team = inj_cache.get("injuries_by_team", {})
    except Exception as e:
        print(f"Edge injury fetch error: {e}")

    # Build roster lookup: team_name -> [player names] (top 8 = starters/key rotation)
    roster_top = {}  # team_name -> list of top player names
    try:
        roster_cache = _get_cached(f"rosters:{sport_lower}", ttl=3600)
        if not roster_cache:
            try:
                await get_rosters(sport)
                roster_cache = _get_cached(f"rosters:{sport_lower}", ttl=3600)
            except Exception:
                pass
        if roster_cache:
            for abbr, info in roster_cache.get("rosters", {}).items():
                team_name = info.get("team", "")
                # Top 8 on roster = starters + key rotation (ESPN lists by importance)
                top = [p["name"].lower() for p in info.get("players", [])[:8]]
                roster_top[team_name.lower()] = top
                roster_top[abbr.lower()] = top
    except Exception as e:
        print(f"Edge roster fetch error: {e}")

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

        # --- INJURY ANALYSIS ---
        # Check injuries for both teams. Star OUT on favorite = edge for dog.
        def _count_injuries(team_name):
            """Count OUT and GTD players for a team, check if any are key (top 8 roster)."""
            out_players = []
            gtd_players = []
            out_stars = []
            # Try matching team name in injuries_by_team (CBS uses city names, full names vary)
            team_lower = team_name.lower()
            matched_injuries = []
            for inj_team, players in injuries_by_team.items():
                # Fuzzy match: "Golden State" in "Golden State Warriors", or "Warriors" in team
                inj_lower = inj_team.lower()
                if inj_lower in team_lower or team_lower in inj_lower or any(w in inj_lower for w in team_lower.split() if len(w) > 3):
                    matched_injuries = players
                    break
            # Get top roster players for this team
            top_roster = []
            for rkey, rplayers in roster_top.items():
                if rkey in team_lower or team_lower in rkey or any(w in rkey for w in team_lower.split() if len(w) > 3):
                    top_roster = rplayers
                    break
            for p in matched_injuries:
                pname = p.get("player", "").lower()
                if p.get("status") == "OUT":
                    out_players.append(p["player"])
                    if any(pname in rp or rp in pname for rp in top_roster):
                        out_stars.append(p["player"])
                elif p.get("status") == "GTD":
                    gtd_players.append(p["player"])
            return {"out": out_players, "gtd": gtd_players, "out_stars": out_stars, "out_count": len(out_players), "gtd_count": len(gtd_players)}

        fav_injuries = _count_injuries(fav)
        dog_injuries = _count_injuries(dog)

        # --- UPSET SCORE (1-10) ---
        # Base: dog implied probability. Adjusted by injuries.
        upset_score = min(10, max(1, round(dog_prob * 20)))

        # Injury adjustments: star OUT on favorite = huge edge shift
        injury_boost = 0
        if fav_injuries["out_stars"]:
            injury_boost += min(3, len(fav_injuries["out_stars"]) * 2)  # +2 per star OUT on fav
        if fav_injuries["out_count"] >= 3:
            injury_boost += 1  # Depth hit on favorite
        if dog_injuries["out_stars"]:
            injury_boost -= min(2, len(dog_injuries["out_stars"]))  # -1 per star OUT on dog
        upset_score = min(10, max(1, upset_score + injury_boost))

        # Build injury summary
        injury_flags = []
        if fav_injuries["out_stars"]:
            injury_flags.append(f"EDGE: {fav} missing {', '.join(fav_injuries['out_stars'])}")
        if fav_injuries["out_count"] > 0 and not fav_injuries["out_stars"]:
            injury_flags.append(f"{fav}: {fav_injuries['out_count']} OUT (role players)")
        if dog_injuries["out_stars"]:
            injury_flags.append(f"CAUTION: {dog} missing {', '.join(dog_injuries['out_stars'])}")
        if fav_injuries["gtd_count"] > 0:
            injury_flags.append(f"{fav}: {fav_injuries['gtd_count']} GTD — monitor")
        if dog_injuries["gtd_count"] > 0:
            injury_flags.append(f"{dog}: {dog_injuries['gtd_count']} GTD — monitor")

        # --- PUBLIC FADE FLAG ---
        # If favorite has >72% implied prob, the public is heavy on them
        public_fade = fav_prob > 0.72

        # --- VALUE RATING ---
        # Dogs with >30% true prob are value plays
        # Dogs with >40% are strong value
        # Injury boost: if fav missing stars, lower the threshold
        value_threshold_shift = 0.03 if fav_injuries["out_stars"] else 0
        value_tag = ""
        if dog_prob >= (0.42 - value_threshold_shift):
            value_tag = "STRONG VALUE"
        elif dog_prob >= (0.35 - value_threshold_shift):
            value_tag = "VALUE"
        elif dog_prob >= (0.28 - value_threshold_shift):
            value_tag = "SLIGHT VALUE"
        # Special: if fav missing 2+ stars and dog has decent prob, it's sharp value
        if len(fav_injuries["out_stars"]) >= 2 and dog_prob >= 0.25 and not value_tag:
            value_tag = "INJURY VALUE"

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
            "injury_boost": injury_boost,
            "public_fade": public_fade,
            "value_tag": value_tag,
            "spread_ml_flag": spread_ml_flag,
            "total_lean": total_lean,
            "dog_ml": dog_ml,
            "injury_flags": injury_flags,
            "fav_out": fav_injuries["out_count"],
            "dog_out": dog_injuries["out_count"],
            "fav_stars_out": fav_injuries["out_stars"],
            "dog_stars_out": dog_injuries["out_stars"],
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
async def get_analysis(sport: str, cached_only: bool = False, force: bool = False, user_id: str = None, model: str = None):
    """Generate AI analysis for a sport based on current live odds.

    Params:
        cached_only: If True, return cached results only (never trigger new GPT call).
                     Used by autopicker to read scheduled analysis without burning API tokens.
        force: If True, bypass all caches and run fresh analysis. Used by manual Analyze button.
        user_id: If provided, run analysis with user's custom matrix weights.

    RULE: Do NOT grade any game without complete line data.
    Games missing spread, total, or ML get grade "INCOMPLETE".
    """
    if not AZURE_ENDPOINT or not AZURE_KEY:
        return JSONResponse({"error": "Azure OpenAI not configured"}, status_code=500)

    sport_lower = sport.lower()

    # User-specific analysis handling
    user_profile = None
    if user_id:
        user_profile = _load_user_profile(user_id)
        if not user_profile:
            return JSONResponse({"error": f"User profile '{user_id}' not found"}, status_code=404)
        if not _check_user_rate_limit(user_id, sport_lower):
            return JSONResponse({"error": f"Rate limit exceeded. Max {MAX_USER_ANALYSIS_PER_SPORT_PER_DAY} custom runs per sport per day."}, status_code=429)

    cache_key = f"analysis:{sport_lower}:{user_id}" if user_id else f"analysis:{sport_lower}"

    # cached_only mode: return whatever we have, never trigger GPT
    if cached_only:
        cached = _get_cached(cache_key, ttl=ANALYSIS_CACHE_TTL)
        if cached:
            cached["cached"] = True
            return JSONResponse(cached)
        cached_analysis = _load_analysis_cache(sport_lower)
        if cached_analysis and cached_analysis.get("games"):
            cached_analysis["cached"] = True
            return JSONResponse(cached_analysis)
        return JSONResponse({
            "sport": sport.upper(), "games": [], "cached": True,
            "gotcha": "No cached analysis available. Wait for next scheduled run.",
            "generated_at": _now_ts(),
        })

    # Normal flow: check caches unless force=True
    if not force:
        cached = _get_cached(cache_key, ttl=ANALYSIS_CACHE_TTL)
        if cached:
            return JSONResponse(cached)

        # Check file-based analysis cache — always serve if today's analysis exists
        cached_analysis = _load_analysis_cache(sport_lower)
        if cached_analysis and cached_analysis.get("games"):
            # Serve disk cache and warm memory cache so subsequent requests are instant
            _set_cache(cache_key, cached_analysis)
            cached_analysis["cached"] = True
            return JSONResponse(cached_analysis)

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

    # Filter out started games
    if odds_data.get("games"):
        now = datetime.now(PST)
        cutoff = now + timedelta(hours=24)
        odds_data["games"] = [
            g for g in odds_data["games"]
            if _game_not_started(g, now) and _game_within_cutoff(g, cutoff)
        ]

    # Fallback: if live odds are empty, try the cached daily slate
    if not odds_data.get("games"):
        slate_data = _load_daily_slate(sport_lower)
        if slate_data:
            # _load_daily_slate returns the full JSON object {"sport":..., "games":[...]}
            # extract just the games list
            slate_games = slate_data.get("games", []) if isinstance(slate_data, dict) else slate_data
            now = datetime.now(PST)
            cutoff = now + timedelta(hours=48)
            future_games = [
                g for g in slate_games
                if _game_not_started(g, now) and _game_within_cutoff(g, cutoff)
            ]
            if future_games:
                logger.info(f"[ANALYSIS] Live odds empty for {sport_lower} — using cached slate ({len(future_games)} future games)")
                odds_data = {"games": future_games, "count": len(future_games), "_source": "cached_slate"}

    if not odds_data.get("games"):
        # Before returning empty, check if we have cached analysis to serve
        stale = _load_analysis_cache(sport_lower)
        if stale and stale.get("games"):
            stale["cached"] = True
            stale["_recovery"] = "no_odds_available"
            _set_cache(cache_key, stale)
            return JSONResponse(stale)
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
        roster_cache = _get_cached(f"rosters:{sport_lower}", ttl=3600)
        if not roster_cache:
            # Trigger ESPN roster fetch if not cached
            try:
                await get_rosters(sport)
                roster_cache = _get_cached(f"rosters:{sport_lower}", ttl=3600)
            except Exception:
                pass
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
                    top_players = [p["name"] for p in info.get("players", [])[:15]]
                    roster_lines.append(f"{abbr} ({team_full}): {', '.join(top_players)}")
            if len(roster_lines) > 1:
                roster_context = "\n".join(roster_lines)
    except Exception as e:
        print(f"Roster context error: {e}")

    # ===== FETCH ESPN INJURIES (supplement CBS/RotoWire) =====
    espn_injury_text = ""
    try:
        espn_injury_text = await _fetch_espn_injuries(sport_lower)
        if espn_injury_text:
            injury_context += "\n" + espn_injury_text
    except Exception as e:
        print(f"ESPN injury fetch error: {e}")

    # ===== FETCH PLAYER GAME LOGS (L10 data for prompt) =====
    game_log_context = ""
    try:
        today_teams = set()
        for g in odds_data.get("games", []):
            today_teams.add(g.get("away", ""))
            today_teams.add(g.get("home", ""))
        if today_teams:
            player_logs = await _fetch_player_game_logs(sport_lower, list(today_teams))
            game_log_context = _format_game_logs_for_prompt(player_logs)
    except Exception as e:
        print(f"Game log fetch error: {e}")

    # ===== BUILD WEIGHTED MATRIX SECTION =====
    if user_profile:
        matrix_section = _build_user_matrix_section(sport_lower, user_profile)
    else:
        matrix_section = _build_matrix_section(sport_lower)

    # ===== FETCH REAL PROP LINES (from The Odds API) =====
    prop_lines_text = ""
    prop_sports_with_markets = {"nba", "nhl", "nfl", "mlb"}  # Sports that have prop markets
    if sport_lower in prop_sports_with_markets:
        try:
            prop_cache_key = f"props:{sport_lower}"
            prop_data = _get_cached(prop_cache_key, ttl=PROPS_CACHE_TTL)
            if not prop_data:
                # Fetch fresh props (reuses the same logic as /api/props/{sport})
                prop_resp = await get_player_props(sport)
                if hasattr(prop_resp, 'body'):
                    prop_data = json.loads(prop_resp.body)
                else:
                    prop_data = None
            if prop_data and prop_data.get("props"):
                # Group props by matchup for easy batch filtering
                props_by_matchup = {}
                for prop in prop_data["props"]:
                    m = prop.get("matchup", "unknown")
                    if m not in props_by_matchup:
                        props_by_matchup[m] = []
                    side = prop.get("side", "Over")
                    line = prop.get("line", "?")
                    stat = prop.get("stat", "?")
                    player = prop.get("player", "?")
                    best_odds = prop.get("best_odds", "?")
                    best_book = prop.get("best_book", "?")
                    props_by_matchup[m].append(f"{player} {side} {line} {stat} ({best_odds}) [{best_book}]")
                # Build text grouped by matchup
                prop_lines = []
                for matchup, lines in props_by_matchup.items():
                    prop_lines.append(f"\n{matchup}:")
                    for l in lines[:20]:  # Cap per game to keep prompt manageable
                        prop_lines.append(f"  - {l}")
                prop_lines_text = "\n".join(prop_lines)
                logger.info(f"Analysis {sport}: injected {len(prop_data['props'])} real prop lines")
        except Exception as e:
            logger.warning(f"Prop lines fetch for analysis failed: {e}")
            prop_lines_text = ""

    # ===== BATCH ANALYSIS — split games into groups of 4 for parallel Azure calls =====
    BATCH_SIZE = 4
    MAX_BATCHES = 6  # Cap at 6 batches (24 games) to avoid excessive Azure calls
    MAX_GAMES = BATCH_SIZE * MAX_BATCHES

    def _build_batch_prompt(batch_games_text, batch_incomplete_note, is_first_batch, batch_prop_lines=""):
        """Build the Azure prompt for a batch of games."""
        gotcha_instruction = ""
        if is_first_batch:
            gotcha_instruction = f'"gotcha": "HTML unordered list (<ul><li>...</li></ul>). 4-8 bullet points covering: KEY INJURIES affecting tonight\'s lines (star players OUT/GTD), B2B/rest flags, line movement alerts, traps to avoid, weather (outdoor), sharp money indicators. Bold critical items with <strong>. End with: <li><em>Analysis generated {now_time} | Data: The Odds API + ESPN + CBS Sports + RotoWire</em></li>",'
        else:
            gotcha_instruction = '"gotcha": "",'

        return f"""You are Edge Finder v2, a sharp sports betting analyst. You have access to REAL injury data, REAL player game logs, and a weighted scoring matrix.

CRITICAL ROSTER RULE: Your training data is STALE. ONLY reference players who appear in the CURRENT ROSTERS and PLAYER GAME LOGS sections below. If a player is NOT in the data below, DO NOT mention them. Do NOT guess team rosters from memory. This is non-negotiable.

NO STALE DATA RULE: Do NOT reference any statistics, records, or player performance data from before the current 2025-26 season. ONLY use data provided in the PLAYER GAME LOGS and INJURY sections below. If you don't have current data for a claim, say "data not available" instead of guessing.

CREW: Peter (heavy/value/sharp, sizes up on conviction), Jimmy (new, learning), Alyssa (pure math/EV edge), Sinton.ia (card builder/grader), Tunk (wild/aggressive, tracks everything, high volume).
RULES: "Why is the market wrong?" = required for every grade. No answer = NO BET (grade D/F). Valid edges: news not priced in, public overreaction, rest/schedule, matchup-specific, sharp vs public, situational. Invalid: "better team", "should win", "volume play".

Today's {sport.upper()} slate - {today} (pulled at {now_time}):

=== ODDS DATA (THIS BATCH) ===
{batch_games_text}
{batch_incomplete_note}

=== INJURY & LINEUP INTELLIGENCE (CBS Sports + RotoWire + ESPN + API-Sports) ===
{injury_context}

{roster_context}

{game_log_context}

{matrix_section}

=== REAL PLAYER PROP LINES (FROM SPORTSBOOKS) ===
{batch_prop_lines if batch_prop_lines else "No real prop lines available — skip player_props for this batch."}

=== EDGE QUESTION (MANDATORY for every game) ===
Before grading: "Why is the market wrong here?" If you cannot answer, grade D or F.

Generate analysis in this EXACT JSON format:
{{
  {gotcha_instruction}
  "games": [
    {{
      "matchup": "AWAY @ HOME",
      "composite_score": 7.2,
      "grade": "A+/A/A-/B+/B/B-/C+/C/D/F/TBD/INCOMPLETE",
      "edge_question": "Why is the market wrong? 1-2 sentences. If no answer: 'No clear edge identified.'",
      "tags": ["B2B", "SHARP", "TRAP", "UPSET", "PASS", "BEST BET", "REST-EDGE", "INJURY-IMPACT", "INCOMPLETE"],
      "matrix_scores": {{
        "star_player_status": {{"score": 7, "weight": 10, "weighted": 70, "note": "why this score"}},
        "rest_advantage": {{"score": 5, "weight": 9, "weighted": 45, "note": "why this score"}}
      }},
      "injury_impact": "Which key players are OUT/GTD and how it changes the line. Be specific.",
      "rest_schedule": "B2B? Days rest for each team? Travel? 3-in-5?",
      "edge_summary": "One line: the edge in plain English",
      "peter_zone": "2-3 sentences. The play: conviction level, sizing suggestion (full unit / half / small / pass), line value assessment.",
      "trends": ["ATS trend", "O/U trend", "H2H trend"],
      "flags": ["injury flag 1", "schedule flag", "sharp money flag"],
      "edge_pick": {{
        "team": "TEAM NAME or PASS",
        "bet_type": "SPREAD / ML / TOTAL OVER / TOTAL UNDER / PASS",
        "line": "The specific line (e.g. -3.5, +150, O 218.5)",
        "confidence": "A+/A/A-/B+/B/B-/C/D/F",
        "reasoning": "One sentence: why this is THE play on this game."
      }},
      "player_props": [
        {{
          "player": "Player Name",
          "prop": "OVER/UNDER 24.5 Points",
          "line": "-115",
          "grade": "A/B+/B/C",
          "l10_avg": "28.1 PTS (from PLAYER GAME LOGS data)",
          "season_avg": "27.4 PTS",
          "games_played": 58,
          "status": "Active / GTD / Questionable",
          "edge": "+1.6 pts over line | why this prop hits"
        }}
      ],
      "data_status": "COMPLETE or INCOMPLETE or TBD - state exactly what is missing or pending (lineups, injury report, etc.)",
      "book_source": "which sportsbook"
    }}
  ]
}}

HARD RULES — NEVER BREAK THESE:
1. VALUE RANGE: -180 to +400. This is where we find edges. ML worse than -180 = too juiced, take the spread instead. ML past +400 = longshot, not a value play. If a game's best ML play falls outside -180 to +400, flag it and redirect to spread/total.
8. If lineups are NOT confirmed (game hasn't posted starters, injury report pending, lineups TBD), grade the game "TBD" — not a letter grade. We do not grade games without confirmed lineups. Period.
2. NEVER suggest a player prop without considering games played this season. If a player has < 20 games, FLAG IT: "Returning from injury / limited sample / possible minutes restriction." Grade that prop C or lower regardless of the line.
3. ALL player prop analysis uses LAST 10 GAME averages from the PLAYER GAME LOGS table above, not season averages. Season stats lie for guys who missed time. Reference the L10 PTS/REB/AST columns. If a player is NOT in the game logs table, say: "No recent game log available" — do NOT estimate or guess stats.
4. FULL LINEUP CONTEXT before grading ANY game. If a star player is OUT, LIMITED, or returning from injury, the ENTIRE team analysis changes — scoring output, pace, defensive matchups, everything. The spread already prices this in — your analysis must too. Don't grade a team as if they have their full roster when they don't.
5. If you don't have current injury/lineup data, SAY SO. "Injury report not confirmed" is better than a confident wrong analysis. Do not fabricate injury status.
6. Be SPECIFIC in injury_impact. Name the player, their status (OUT/GTD/QUESTIONABLE), and exactly how it changes the line. "Key players may be out" is garbage. "Jimmy Butler OUT — Heat lose 20 PPG, 6 APG, primary creator" is analysis.
7. edge_question must have a REAL answer or the grade is D/F. "Team X is better" is not an edge. "Line hasn't moved despite Butler being ruled out 2 hours ago" IS an edge.

GRADING RULES:
- Use the SCORING MATRIX above. Fill in matrix_scores for each game with ALL variables scored 1-10.
- composite_score = (sum of all weighted scores) / max_possible * 10. Round to 1 decimal.
- Map composite to grade using the thresholds above.
- For NHL: totals-only or ML-only ARE gradeable. Only INCOMPLETE if NO data at all.
- For MMA/Boxing: ML-only IS gradeable.
- For NBA: missing spread, total, OR ML = INCOMPLETE.
- For Soccer: ML is required. Spread and totals are optional (many leagues don't offer them). ML-only IS gradeable.
- injury_impact is MANDATORY. Check the CBS Sports + ESPN + RotoWire data above. Name specific players.
- If injury data is unavailable from all sources, flag it: "Injury data not confirmed - grade with caution."
- rest_schedule is MANDATORY. Check game times for B2B detection.
- Player props: Select ONLY from the REAL PLAYER PROP LINES section above. Do NOT invent or estimate prop lines — use the exact lines and odds from sportsbooks. If no real prop lines are available for a game, set player_props to empty array []. Top 2-3 per game, A- grade minimum. Skip for INCOMPLETE or TBD. Each prop MUST include: player name, exact prop line, actual odds, individual grade, L10 average from PLAYER GAME LOGS table (l10_avg field), season average (season_avg field), games_played count, player status, and edge explanation showing the differential between L10 avg and the prop line. EVERY prop must note games played this season. < 20 games = auto-flag.
- ABSOLUTE ROSTER LOCK: A player prop can ONLY be suggested for a player who appears in the CURRENT ROSTERS data for one of the two teams in that specific game. If "Jayson Tatum" is in the BOS roster and the game is NYK @ DEN, you CANNOT suggest a Tatum prop on that game — he is NOT playing in it. This is non-negotiable. Check the roster list, find the player, confirm the team matches the game. If the player is not in either team's roster for that game, DO NOT suggest the prop. Period.
- PROP QUALITY RULES (NON-NEGOTIABLE):
  1. NO NHL goal props. They hit at <25%. Not +EV at any line. Skip them entirely.
  2. NO soccer assist props unless the player averages 0.3+ assists per game AND is a confirmed starter. Bench/sub = auto-skip.
  3. MAX 2 props per player per game. Do not stack 4 props on one role player.
  4. ROLE PLAYER FILTER: If a player averages under 15 MPG or under 8 PPG, their stat lines are too volatile. Skip unless the edge is overwhelming (A grade only).
  5. STICK TO HIGH-VOLUME STATS: Points for scorers, rebounds for bigs, assists for playmakers. Do NOT pick rebounds for a guard who averages 3 RPG or assists for a center who averages 1 APG — the variance is too high.
  6. Odds range for props: -200 to +250. Outside that range = too juiced or too unlikely. Skip it.
- PASS games get grade D or F with explicit reason.
- Be brutally honest. C means marginal. D means no edge. F means trap. "Slight edge" with no specifics = D grade, not B.

Return ONLY valid JSON. No markdown. No explanation."""

    def _clean_json(raw):
        """Strip markdown fences and parse JSON from model output."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
        return json.loads(raw)

    # ── Raw analysis persistence ──
    def _save_raw_analysis(raw_text, batch_idx):
        """Save Grok's raw analysis to disk so it's never lost if DeepSeek fails."""
        try:
            today_str = datetime.now(PST).strftime("%Y-%m-%d")
            raw_dir = os.path.join("data", "raw_analysis")
            os.makedirs(raw_dir, exist_ok=True)
            path = os.path.join(raw_dir, f"{sport_lower}_{today_str}_{batch_idx}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(raw_text)
            logger.info(f"[RAW SAVED] {path} ({len(raw_text)} chars)")
            # Cleanup files > 7 days old
            cutoff = datetime.now(PST) - timedelta(days=7)
            for fname in os.listdir(raw_dir):
                fpath = os.path.join(raw_dir, fname)
                if os.path.isfile(fpath):
                    mtime = datetime.fromtimestamp(os.path.getmtime(fpath), tz=PST)
                    if mtime < cutoff:
                        os.remove(fpath)
        except Exception as e:
            logger.warning(f"[RAW SAVE FAILED] {e}")

    # ── Build thinker prompt (no JSON pressure) ──
    def _build_thinker_prompt(batch_games_text, batch_incomplete_note, is_first_batch, batch_prop_lines=""):
        """Build prompt for the reasoning model — data + questions, no JSON schema."""
        matrix = SPORT_MATRICES.get(sport_lower, [])
        var_list = "\n".join(
            f"  {i}. \"{key}\" (weight {weight}) — {desc}"
            for i, (key, weight, desc) in enumerate(matrix, 1)
        )
        total_weight = sum(w for _, w, _ in matrix)
        max_score = total_weight * 10

        gotcha_task = ""
        if is_first_batch:
            gotcha_task = f"""
GOTCHA SECTION (first batch only):
Before the game analysis, write a "Gotcha" briefing covering:
- KEY INJURIES affecting tonight's lines (star players OUT/GTD)
- B2B/rest flags
- Line movement alerts
- Traps to avoid
- Weather (if outdoor)
- Sharp money indicators
(Analysis generated {now_time} | Data: The Odds API + ESPN + CBS Sports + RotoWire)
"""

        return f"""You are a sharp sports betting analyst with access to REAL-TIME injury data, player game logs, and a weighted scoring matrix.

CRITICAL ROSTER RULE: Your training data is STALE. ONLY reference players who appear in the CURRENT ROSTERS and PLAYER GAME LOGS sections below. If a player is NOT in the data below, DO NOT mention them.

NO STALE DATA RULE: Do NOT reference any statistics from before the current 2025-26 season. ONLY use data provided below.

CREW: Peter (heavy/value/sharp), Jimmy (new, learning), Alyssa (pure math/EV), Sinton.ia (card builder/grader).

Today's {sport.upper()} slate - {today} (pulled at {now_time}):

=== ODDS DATA (THIS BATCH) ===
{batch_games_text}
{batch_incomplete_note}

=== INJURY & LINEUP INTELLIGENCE (CBS Sports + RotoWire + ESPN + API-Sports) ===
{injury_context}

{roster_context}

{game_log_context}

=== REAL PLAYER PROP LINES (FROM SPORTSBOOKS) ===
{batch_prop_lines if batch_prop_lines else "No real prop lines available — skip player props for this batch."}
{gotcha_task}
=== {sport.upper()} SCORING MATRIX ({len(matrix)} Variables) ===
Score each variable 1-10 for EACH game. Use the FULL 1-10 range — do NOT cluster 5-7.
1-2 = Strong negative | 3-4 = Moderate negative | 5 = Neutral | 6-7 = Moderate positive | 8-9 = Strong positive | 10 = Extreme edge
Max possible weighted score: {max_score}. Composite = sum of weighted / {max_score} x 10.

{var_list}

COMPOSITE THRESHOLDS:
  9.0-10.0 = BEST BET (A+, load up)
  7.5-8.9  = STRONG PLAY (A-/B+)
  6.0-7.4  = MODERATE EDGE (B/B-)
  4.5-5.9  = LEAN (C, small or pass)
  Below 4.5 = NO EDGE (D/F, pass)

=== YOUR TASK FOR EACH GAME ===
1. EDGE QUESTION: "Why is the market wrong here?" If no answer → grade D/F.
2. INJURY IMPACT: Name specific players OUT/GTD and how it changes the line.
3. REST/SCHEDULE: B2B? Days rest? Travel? 3-in-5?
4. Score ALL {len(matrix)} variables by their KEY NAME (e.g. "star_player_status": 8 because...). Show reasoning for each.
5. Compute composite score using the weights above.
6. EDGE PICK: team, bet type (spread/ML/total), specific line, confidence grade, one-sentence reasoning.
   CRITICAL — TEAM NAME RULE: The "team" in edge_pick MUST match the side you are recommending.
   - If recommending the UNDERDOG (taking points), use the underdog's team name with the POSITIVE spread (e.g. "PHI +8", NOT the favorite's name).
   - If recommending the FAVORITE (laying points), use the favorite's team name with the NEGATIVE spread (e.g. "POR -8").
   - The spread data shows "Spread: HOME SPREAD_VALUE". If SPREAD_VALUE is negative, HOME is the favorite and AWAY is the dog. If positive, HOME is the dog.
   - VALUE DOG signal = you are recommending the underdog. The edge_pick team MUST be the dog (the +points team). Double-check before writing.
7. PLAYER PROPS: Top 2-3 from the real prop lines above. Use L10 averages from game logs. No props if no real lines available.
8. FLAGS: injury flags, schedule flags, sharp money flags.
9. PETER ZONE: The play — conviction, sizing (full unit / half / small / pass), line value.
10. VALUE RANGE: -180 to +400 for ML. Outside that, redirect to spread/total.

HARD RULES:
- If lineups NOT confirmed, grade TBD — not a letter grade.
- All prop analysis uses LAST 10 GAME averages, not season averages.
- If a player has < 20 games this season, flag it.
- Be brutally honest. "Slight edge" with no specifics = D grade.
- PASS games get D or F with explicit reason.

Think step by step. Be thorough. The next model will format your output into structured JSON."""

    # ── Build formatter prompt (sport-specific variable keys) ──
    def _build_formatter_prompt(raw_analysis, is_first_batch):
        """Build prompt for the formatting model — raw analysis + exact JSON schema + variable keys."""
        matrix = SPORT_MATRICES.get(sport_lower, [])
        var_key_list = "\n".join(
            f'  - "{key}" (weight: {weight}) — {desc}'
            for key, weight, desc in matrix
        )
        total_weight = sum(w for _, w, _ in matrix)
        max_score = total_weight * 10

        gotcha_instruction = ""
        if is_first_batch:
            gotcha_instruction = '"gotcha": "Extract the Gotcha briefing from the analysis and format as HTML: <ul><li>...</li></ul>. Bold critical items with <strong>.",'
        else:
            gotcha_instruction = '"gotcha": "",'

        return f"""You are a precise JSON formatter. Convert the sports betting analysis below into our exact JSON schema.

=== RAW ANALYSIS FROM REASONING MODEL ===
{raw_analysis}

=== {sport.upper()} MATRIX VARIABLE KEYS (you MUST use these exact keys) ===
{var_key_list}

Max weighted score: {max_score}. Composite = sum of all weighted / {max_score} * 10, round to 1 decimal.

=== MAPPING RULES ===
- Match the analyst's variable scores to the correct key by MEANING, not exact wording.
- "weighted" = score * weight for that variable.
- If the analyst scored a variable but used different wording, map it to the closest key.
- If a variable was NOT analyzed, use score 5 (neutral) with note "Not explicitly analyzed".
- ALL {len(matrix)} variables MUST appear in matrix_scores for every game.

=== GRADE THRESHOLDS ===
  9.0-10.0 = A+ | 7.5-8.9 = A-/B+ | 6.0-7.4 = B/B- | 4.5-5.9 = C | Below 4.5 = D/F

=== OUTPUT FORMAT (return ONLY this JSON, no markdown, no explanation) ===
{{
  {gotcha_instruction}
  "games": [
    {{
      "matchup": "AWAY @ HOME",
      "composite_score": 7.2,
      "grade": "A+/A/A-/B+/B/B-/C+/C/D/F/TBD/INCOMPLETE",
      "edge_question": "Why is the market wrong? 1-2 sentences.",
      "tags": ["B2B", "SHARP", "TRAP", "UPSET", "PASS", "BEST BET", "REST-EDGE", "INJURY-IMPACT", "INCOMPLETE"],
      "matrix_scores": {{
        "variable_key": {{"score": 7, "weight": 9, "weighted": 63, "note": "reason"}}
      }},
      "injury_impact": "specific injury analysis",
      "rest_schedule": "B2B and rest info",
      "edge_summary": "one line edge in plain English",
      "peter_zone": "2-3 sentences. The play: conviction, sizing, line value.",
      "trends": ["ATS trend", "O/U trend", "H2H trend"],
      "flags": ["injury flag", "schedule flag", "sharp money flag"],
      "edge_pick": {{
        "team": "TEAM NAME or PASS",
        "bet_type": "SPREAD / ML / TOTAL OVER / TOTAL UNDER / PASS",
        "line": "specific line (e.g. -3.5, +150, O 218.5)",
        "confidence": "A+/A/A-/B+/B/B-/C/D/F",
        "reasoning": "one sentence"
      }},
      "player_props": [
        {{
          "player": "Player Name",
          "prop": "OVER/UNDER 24.5 Points",
          "line": "-115",
          "grade": "A/B+/B/C",
          "l10_avg": "28.1 PTS",
          "season_avg": "27.4 PTS",
          "games_played": 58,
          "status": "Active / GTD / Questionable",
          "edge": "+1.6 pts over line | reason"
        }}
      ],
      "data_status": "COMPLETE or INCOMPLETE or TBD",
      "book_source": "sportsbook name"
    }}
  ]
}}

Return ONLY valid JSON. No markdown fences. No explanation."""

    # ── Core model call functions ──
    def _call_thinker(prompt_text, batch_idx=0):
        """Call the reasoning model (Grok). Returns (raw_analysis_text, metadata)."""
        client = OpenAI(
            base_url=THINKER_ENDPOINT,
            api_key=AZURE_KEY,
            timeout=90,  # 90s — fail fast, fallback to single-model before 120s batch cutoff
        )
        logger.info(f"[THINKER] Batch {batch_idx} ({sport.upper()}) → {ANALYSIS_THINKER} via {THINKER_ENDPOINT}")
        think_start = time.time()
        think_response = client.chat.completions.create(
            model=ANALYSIS_THINKER,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=16000,
        )
        raw_analysis = think_response.choices[0].message.content.strip()
        think_secs = round(time.time() - think_start, 1)
        think_tokens = getattr(think_response.usage, 'total_tokens', '?')
        logger.info(f"[THINKER] Done in {think_secs}s, {think_tokens} tokens, {len(raw_analysis)} chars")
        return raw_analysis, {"secs": think_secs, "tokens": think_tokens, "chars": len(raw_analysis)}

    def _call_formatter(raw_analysis, is_first_batch, batch_idx=0, attempt=1):
        """Call the formatting model (DeepSeek). Returns (parsed_json, metadata)."""
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_KEY,
            api_version="2024-10-21",
        )
        fmt_prompt = _build_formatter_prompt(raw_analysis, is_first_batch)
        logger.info(f"[FORMATTER] Batch {batch_idx} → {ANALYSIS_FORMATTER} (attempt {attempt})")
        fmt_start = time.time()
        fmt_response = client.chat.completions.create(
            model=ANALYSIS_FORMATTER,
            messages=[{"role": "user", "content": fmt_prompt}],
            temperature=0.1,
            max_tokens=10000,
        )
        raw = fmt_response.choices[0].message.content.strip()
        fmt_secs = round(time.time() - fmt_start, 1)
        fmt_tokens = getattr(fmt_response.usage, 'total_tokens', '?')
        result = _clean_json(raw)
        games_with_ms = sum(1 for g in result.get("games", []) if g.get("matrix_scores"))
        total_games = len(result.get("games", []))
        logger.info(f"[FORMATTER] Done in {fmt_secs}s, {fmt_tokens} tokens, {games_with_ms}/{total_games} games with matrix_scores")
        return result, {"secs": fmt_secs, "tokens": fmt_tokens}

    def _call_single_model(prompt_text, model_name=None):
        """Fallback: single-model call to AZURE_MODEL (gpt-4.1) with full JSON prompt."""
        use_model = model_name or AZURE_MODEL
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_KEY,
            api_version="2024-10-21",
        )
        logger.info(f"[FALLBACK] Single-model call to {use_model}")
        fb_start = time.time()
        response = client.chat.completions.create(
            model=use_model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.4,
            max_tokens=8000,
        )
        raw = response.choices[0].message.content.strip()
        fb_secs = round(time.time() - fb_start, 1)
        fb_tokens = getattr(response.usage, 'total_tokens', '?')
        logger.info(f"[FALLBACK] {use_model} done in {fb_secs}s, {fb_tokens} tokens")
        result = _clean_json(raw)
        for game in result.get("games", []):
            game["_thinker"] = use_model
            game["_formatter"] = use_model
            game["_think_time"] = fb_secs
        return result

    def _call_challenger(prompt_text, batch_idx=0):
        """Call this week's challenger model for independent grading. Returns parsed JSON or None."""
        try:
            challenger = _get_weekly_challenger()
            if not challenger:
                return None
            model_name = challenger["name"]
            endpoint_type = challenger["endpoint"]

            # Skip Anthropic models if API key is missing
            if endpoint_type == "anthropic" and not ANTHROPIC_API_KEY:
                logger.warning(f"[CHALLENGER] Skipping {challenger['display']} — ANTHROPIC_API_KEY not set")
                return None

            logger.info(f"[CHALLENGER] Batch {batch_idx} ({sport.upper()}) → {challenger['display']} ({model_name})")
            ch_start = time.time()

            if endpoint_type == "anthropic":
                anth_client = anthropic_sdk.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=120)
                anth_response = anth_client.messages.create(
                    model=model_name,
                    max_tokens=8000,
                    temperature=0.4,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                raw = anth_response.content[0].text.strip()
                ch_secs = round(time.time() - ch_start, 1)
                ch_tokens = getattr(anth_response.usage, 'input_tokens', 0) + getattr(anth_response.usage, 'output_tokens', 0)
                logger.info(f"[CHALLENGER] {challenger['display']} done in {ch_secs}s, {ch_tokens} tokens")
            else:
                if endpoint_type == "azure":
                    client = AzureOpenAI(
                        azure_endpoint=AZURE_ENDPOINT,
                        api_key=AZURE_KEY,
                        api_version="2024-10-21",
                        timeout=120,
                    )
                else:
                    client = OpenAI(
                        base_url=THINKER_ENDPOINT,
                        api_key=AZURE_KEY,
                        timeout=120,
                    )

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.4,
                    max_tokens=8000,
                )
                raw = response.choices[0].message.content.strip()
                ch_secs = round(time.time() - ch_start, 1)
                ch_tokens = getattr(response.usage, 'total_tokens', '?')
                logger.info(f"[CHALLENGER] {challenger['display']} done in {ch_secs}s, {ch_tokens} tokens")
            result = _clean_json(raw)
            for game in result.get("games", []):
                game["_challenger_model"] = challenger["display"]
            return result
        except Exception as e:
            logger.warning(f"[CHALLENGER] Call failed for {_get_weekly_challenger().get('display', '?')}: {e}")
            return None

    # ── Quality validation helpers ──
    def _validate_thinker_output(raw_analysis, batch_games):
        """Check that thinker referenced teams and injuries from the batch. Non-blocking."""
        raw_lower = raw_analysis.lower()
        # Check team names
        team_count = 0
        team_total = 0
        for game_line in batch_games:
            parts = game_line.split("|")[0].strip()
            for team in parts.replace(" @ ", "|").split("|"):
                team_name = team.strip()
                team_total += 1
                if team_name.lower() in raw_lower:
                    team_count += 1
        # Check injury names from injury_context
        injury_count = 0
        injury_total = 0
        if injury_context:
            for line in injury_context.split("\n"):
                line = line.strip()
                if line and ("OUT" in line.upper() or "GTD" in line.upper() or "QUESTIONABLE" in line.upper()):
                    injury_total += 1
                    # Extract player name (usually first part before status)
                    name_part = line.split("(")[0].split("-")[0].split(":")[0].strip()
                    if len(name_part) > 3 and name_part.lower() in raw_lower:
                        injury_count += 1
        logger.info(f"[QUALITY] {team_count}/{team_total} teams referenced, {injury_count}/{injury_total} injuries mentioned")
        return team_total == 0 or (team_count / max(team_total, 1)) > 0.5

    def _validate_formatter_output(result):
        """Check that formatter produced quality matrix_scores. Non-blocking."""
        games = result.get("games", [])
        if not games:
            return False
        games_with_ms = sum(1 for g in games if g.get("matrix_scores"))
        # Check score variance (not all clustering at 5-7)
        low_variance = 0
        for g in games:
            ms = g.get("matrix_scores", {})
            if ms:
                scores = [v.get("score", 5) for v in ms.values() if isinstance(v, dict)]
                if len(scores) > 3:
                    sd = statistics.stdev(scores)
                    if sd <= 1.0:
                        low_variance += 1
                        logger.info(f"[QUALITY] {g.get('matchup', '?')} — score StdDev={sd:.2f} (too clustered)")
        ms_pct = games_with_ms / len(games) if games else 0
        logger.info(f"[QUALITY] {games_with_ms}/{len(games)} games have matrix_scores, {low_variance} with low variance")
        return ms_pct > 0.5

    # ── Main orchestrator ──
    def _call_azure_batch(prompt_text, batch_idx=0, is_first_batch=False, batch_games=None, batch_prop_lines="", batch_incomplete_note="", model_override=None):
        """Two-model analysis engine: Grok thinks, DeepSeek formats, gpt-4.1 fallback.

        ANALYSIS_MODE='single': uses gpt-4.1 with full JSON prompt (safe fallback).
        ANALYSIS_MODE='twomodel': Grok reasons → DeepSeek formats → fallback if needed.
        model_override: if set, force single-model with this deployment (e.g. 'gpt-4.1-mini' for DJ page).
        """
        logger.info(f"[ANALYSIS MODE] {ANALYSIS_MODE} — thinker={ANALYSIS_THINKER}, formatter={ANALYSIS_FORMATTER}")

        # ── Model override (e.g. DJ page uses gpt-4.1-mini) ──
        if model_override:
            logger.info(f"[MODEL OVERRIDE] Forcing single-model: {model_override}")
            return _call_single_model(prompt_text, model_override)

        # ── Single-model mode (kill switch) ──
        if ANALYSIS_MODE == "single":
            return _call_single_model(prompt_text)

        # ── Two-model mode ──
        # Build thinker prompt from batch data
        batch_text = "\n".join(batch_games) if batch_games else "(no complete games in this batch)"

        thinker_prompt = _build_thinker_prompt(
            batch_text,
            batch_incomplete_note=batch_incomplete_note,
            is_first_batch=is_first_batch,
            batch_prop_lines=batch_prop_lines
        )

        # Step 2: Call thinker (Grok)
        try:
            raw_analysis, think_meta = _call_thinker(thinker_prompt, batch_idx)
        except Exception as e:
            logger.error(f"[THINKER FAIL] {e} — falling back to {AZURE_MODEL} single-model")
            return _call_single_model(prompt_text)

        # Step 3: Validate thinker output (non-blocking)
        if len(raw_analysis) < 200:
            logger.warning(f"[THINKER] Output too short ({len(raw_analysis)} chars) — falling back to {AZURE_MODEL}")
            _save_raw_analysis(raw_analysis, batch_idx)
            return _call_single_model(prompt_text)

        if batch_games:
            _validate_thinker_output(raw_analysis, batch_games)

        # Step 4: Call formatter (DeepSeek) — retry once on failure
        result = None
        for attempt in range(1, 3):
            try:
                result, fmt_meta = _call_formatter(raw_analysis, is_first_batch, batch_idx, attempt)
                # Validate: >50% games must have matrix_scores
                if _validate_formatter_output(result):
                    break
                elif attempt == 1:
                    logger.warning(f"[FORMATTER] Low quality output — retrying (attempt 2)")
                    result = None
                    continue
            except Exception as e:
                logger.error(f"[FORMATTER JSON ERROR] attempt {attempt} — {'retrying' if attempt == 1 else 'giving up'}: {e}")
                if attempt == 2:
                    _save_raw_analysis(raw_analysis, batch_idx)
                    logger.info(f"[FALLBACK] Using {AZURE_MODEL} single-model for batch {batch_idx}")
                    return _call_single_model(prompt_text)

        if result is None:
            _save_raw_analysis(raw_analysis, batch_idx)
            logger.info(f"[FALLBACK] Formatter failed both attempts — using {AZURE_MODEL} single-model for batch {batch_idx}")
            return _call_single_model(prompt_text)

        # Step 5: Tag games with model metadata
        for game in result.get("games", []):
            game["_thinker"] = ANALYSIS_THINKER
            game["_formatter"] = ANALYSIS_FORMATTER
            game["_think_time"] = think_meta["secs"]

        return result

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

        # Build prompts for each batch (single-model prompt used as fallback)
        batch_prompts = []     # Single-model prompts (fallback)
        batch_game_lists = []  # Raw game line lists per batch (for thinker)
        batch_prop_texts = []  # Filtered prop text per batch
        for idx, batch in enumerate(game_batches):
            batch_text = "\n".join(batch) if batch else "(no complete games in this batch)"
            # Only include incomplete note in first batch
            batch_inc = incomplete_note if idx == 0 else ""
            # Filter prop lines to only include matchups in this batch
            batch_props = ""
            if prop_lines_text:
                # Extract team names from batch games to filter props
                batch_teams = set()
                for game_line in batch:
                    # Game line format: "AWAY @ HOME | Spread: ..."
                    parts = game_line.split("|")[0].strip()
                    for team in parts.replace(" @ ", "|").split("|"):
                        batch_teams.add(team.strip().lower())
                # Filter prop lines by matching teams
                if batch_teams:
                    filtered_props = []
                    current_matchup = ""
                    include_matchup = False
                    for line in prop_lines_text.split("\n"):
                        if line.strip() and not line.startswith("  "):
                            # Matchup header line
                            current_matchup = line.strip().rstrip(":")
                            include_matchup = any(t in current_matchup.lower() for t in batch_teams if len(t) > 3)
                            if include_matchup:
                                filtered_props.append(line)
                        elif include_matchup and line.strip():
                            filtered_props.append(line)
                    batch_props = "\n".join(filtered_props) if filtered_props else ""
            batch_prompts.append(_build_batch_prompt(batch_text, batch_inc, is_first_batch=(idx == 0), batch_prop_lines=batch_props))
            batch_game_lists.append(batch)
            batch_prop_texts.append(batch_props)

        # Run all batches in parallel via asyncio threads — 120s hard cutoff per batch
        async def _run_batch_with_timeout(i):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        _call_azure_batch, batch_prompts[i],
                        batch_idx=i, is_first_batch=(i == 0), batch_games=batch_game_lists[i],
                        batch_prop_lines=batch_prop_texts[i],
                        batch_incomplete_note=(incomplete_note if i == 0 else ""),
                        model_override=model,
                    ),
                    timeout=120  # 120s per batch — never let a model hang block the card
                )
            except asyncio.TimeoutError:
                logger.warning(f"[BATCH {i}] Timed out after 120s — skipping batch")
                return {"games": [], "gotcha": "" if i > 0 else "Some analysis batches timed out. Partial results shown."}
        tasks = [_run_batch_with_timeout(i) for i in range(len(batch_prompts))]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results: first batch provides gotcha, all provide games
        all_analyzed_games = []
        gotcha_html = ""
        failed_batches = []
        for idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Analysis batch {idx} failed: {result}")
                log_error("Analysis", f"Batch {idx} failed", str(result))
                failed_batches.append(idx)
                continue
            if idx == 0 and result.get("gotcha"):
                gotcha_html = result["gotcha"]
            if result.get("games"):
                all_analyzed_games.extend(result["games"])

        # Retry failed batches once
        if failed_batches:
            logger.info(f"Analysis {sport}: retrying {len(failed_batches)} failed batches")
            retry_tasks = [
                asyncio.to_thread(
                    _call_azure_batch, batch_prompts[i],
                    batch_idx=i, is_first_batch=(i == 0), batch_games=batch_game_lists[i],
                    batch_prop_lines=batch_prop_texts[i],
                    batch_incomplete_note=(incomplete_note if i == 0 else "")
                )
                for i in failed_batches
            ]
            retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
            for ri, idx in enumerate(failed_batches):
                result = retry_results[ri]
                if isinstance(result, Exception):
                    logger.error(f"Analysis batch {idx} retry also failed: {result}")
                    continue
                if idx == 0 and not gotcha_html and result.get("gotcha"):
                    gotcha_html = result["gotcha"]
                if result.get("games"):
                    all_analyzed_games.extend(result["games"])
                    logger.info(f"Analysis batch {idx} retry succeeded: {len(result['games'])} games recovered")

        # ===== MATCHUP NORMALIZATION — ensure model output matches odds format =====
        # Build expected matchup set from odds data (e.g., "NYK @ IND")
        expected_matchups = set()
        for g in odds_data.get("games", []):
            m = f"{g.get('away', '?')} @ {g.get('home', '?')}"
            expected_matchups.add(m)
        # Build reverse lookup: team name fragments → abbreviation
        # Extract all team names from odds games for fuzzy matching
        _team_lookup = {}
        for g in odds_data.get("games", []):
            away_abbr = g.get("away", "")
            home_abbr = g.get("home", "")
            if away_abbr:
                _team_lookup[away_abbr.lower()] = away_abbr
                # Add common fragments (e.g., "knicks" → "NYK")
                for word in away_abbr.lower().split():
                    if len(word) > 2:
                        _team_lookup[word] = away_abbr
            if home_abbr:
                _team_lookup[home_abbr.lower()] = home_abbr
                for word in home_abbr.lower().split():
                    if len(word) > 2:
                        _team_lookup[word] = home_abbr
            # Also map full team names if available
            away_full = g.get("away_full", "")
            home_full = g.get("home_full", "")
            if away_full:
                _team_lookup[away_full.lower()] = away_abbr
                for word in away_full.lower().split():
                    if len(word) > 3:
                        _team_lookup[word] = away_abbr
            if home_full:
                _team_lookup[home_full.lower()] = home_abbr
                for word in home_full.lower().split():
                    if len(word) > 3:
                        _team_lookup[word] = home_abbr

        for game in all_analyzed_games:
            raw_matchup = game.get("matchup", "")
            if raw_matchup in expected_matchups:
                continue  # Already matches
            # Try to resolve full team names → abbreviations
            parts = raw_matchup.split(" @ ") if " @ " in raw_matchup else raw_matchup.split(" vs ")
            if len(parts) == 2:
                away_raw, home_raw = parts[0].strip(), parts[1].strip()
                away_resolved = _team_lookup.get(away_raw.lower())
                home_resolved = _team_lookup.get(home_raw.lower())
                # Try last word (usually the team name: "New York Knicks" → "Knicks")
                if not away_resolved:
                    last_word = away_raw.split()[-1].lower() if away_raw.split() else ""
                    away_resolved = _team_lookup.get(last_word)
                if not home_resolved:
                    last_word = home_raw.split()[-1].lower() if home_raw.split() else ""
                    home_resolved = _team_lookup.get(last_word)
                if away_resolved and home_resolved:
                    fixed = f"{away_resolved} @ {home_resolved}"
                    if fixed in expected_matchups:
                        logger.info(f"[MATCHUP FIX] '{raw_matchup}' → '{fixed}'")
                        game["matchup"] = fixed

        # ===== LOG: Raw model output before our grading =====
        for game in all_analyzed_games:
            gpt_g = game.get("grade", "NONE")
            gpt_s = game.get("composite_score", "NONE")
            has_ms = bool(game.get("matrix_scores"))
            ms_count = len(game.get("matrix_scores", {}))
            logger.info(f"[PRE-GRADE] {game.get('matchup')} — GPT grade={gpt_g}, GPT score={gpt_s}, matrix_scores={'YES ('+str(ms_count)+' vars)' if has_ms else 'MISSING'}")

        # ===== RETRY: Re-analyze games missing matrix_scores =====
        missing_ms_games = [g for g in all_analyzed_games if not g.get("matrix_scores") and g.get("grade") not in ("TBD", "INCOMPLETE")]
        if missing_ms_games and SPORT_MATRICES.get(sport_lower):
            logger.info(f"[MATRIX RETRY] {len(missing_ms_games)} games missing matrix_scores — retrying as single batch")
            retry_matchups = [g.get("matchup", "?") for g in missing_ms_games]
            # Find the original odds lines for these games
            retry_lines = []
            for line in complete_games:
                for m in retry_matchups:
                    if m and m.replace(" ", "").upper() in line.replace(" ", "").upper():
                        retry_lines.append(line)
                        break
            if retry_lines:
                retry_prompt = _build_batch_prompt("\n".join(retry_lines), "", is_first_batch=False, batch_prop_lines="")
                try:
                    retry_result = await asyncio.to_thread(_call_azure_batch, retry_prompt, batch_idx=99, is_first_batch=False, batch_games=retry_lines)
                    if retry_result and retry_result.get("games"):
                        # Replace the missing-matrix games with retry results
                        retry_by_matchup = {}
                        for rg in retry_result["games"]:
                            rm = rg.get("matchup", "").replace(" ", "").upper()
                            if rm:
                                retry_by_matchup[rm] = rg
                        replaced = 0
                        for i, game in enumerate(all_analyzed_games):
                            gm = game.get("matchup", "").replace(" ", "").upper()
                            if gm in retry_by_matchup and not game.get("matrix_scores"):
                                retry_game = retry_by_matchup[gm]
                                if retry_game.get("matrix_scores"):
                                    all_analyzed_games[i] = retry_game
                                    replaced += 1
                                    logger.info(f"[MATRIX RETRY OK] {game.get('matchup')} — got {len(retry_game['matrix_scores'])} vars on retry")
                        logger.info(f"[MATRIX RETRY] Replaced {replaced}/{len(missing_ms_games)} games")
                except Exception as e:
                    logger.warning(f"[MATRIX RETRY FAILED] {e}")
                    log_error("Analysis", "Matrix retry failed", str(e))

        # ===== SERVER-SIDE GRADE RECALCULATION — we grade, not GPT =====
        grade_log = {"total": 0, "matrix_graded": 0, "fallback_graded": 0, "kept_gpt": 0, "no_grade": 0}
        for game in all_analyzed_games:
            grade_log["total"] += 1
            pre_grade = game.get("grade", "")
            _recalculate_grade(game, sport_lower)
            post_grade = game.get("grade", "")
            source = game.get("grade_source", "matrix")
            if source == "gpt_composite_fallback":
                grade_log["fallback_graded"] += 1
            elif source == "no_data":
                grade_log["no_grade"] += 1
            elif post_grade != pre_grade:
                grade_log["matrix_graded"] += 1
            else:
                grade_log["kept_gpt"] += 1
            thinker_tag = game.get("_thinker", "unknown")
            formatter_tag = game.get("_formatter", "unknown")
            logger.info(f"[POST-GRADE] {game.get('matchup')} — {pre_grade} -> {post_grade} (source={source}, score={game.get('composite_score', '?')}, thinker={thinker_tag}, formatter={formatter_tag})")
        logger.info(f"[GRADE SUMMARY] {sport}: {grade_log}")

        # ===== ALT GRADE — team profile + H2H based secondary grade =====
        for game in all_analyzed_games:
            try:
                matchup = game.get("matchup", "")
                if not matchup:
                    continue
                away_abbr, home_abbr = _parse_matchup_teams(matchup)
                if not away_abbr or not home_abbr:
                    continue
                away_name = _expand_abbrevs(away_abbr, sport_lower)
                home_name = _expand_abbrevs(home_abbr, sport_lower)
                away_prof = _build_team_profile(away_name, sport_lower)
                home_prof = _build_team_profile(home_name, sport_lower)
                h2h_data = _build_h2h(away_name, home_name, sport_lower)
                alt = _calculate_alt_grade(away_prof, home_prof, h2h_data)
                game["alt_grade"] = alt
            except Exception as e:
                logger.warning(f"Alt grade failed for {game.get('matchup', '?')}: {e}")

        # ===== CROSS-VALIDATION GATE (WS4) — validate players before caching =====
        validation_log = []
        if all_analyzed_games:
            try:
                roster_data = _get_cached(f"rosters:{sport_lower}", ttl=3600)
                all_analyzed_games, validation_log = _validate_analysis_players(
                    all_analyzed_games, roster_data, injury_context
                )
                if validation_log:
                    logger.info(f"Analysis {sport}: Player validation — {len(validation_log)} exclusions")
                    for vlog in validation_log[:10]:
                        logger.info(f"  {vlog}")
            except Exception as e:
                logger.warning(f"Player validation error: {e}")

        # Add incomplete games to the response so frontend can display them
        for ig in incomplete_games:
            all_analyzed_games.append({
                "matchup": ig["matchup"],
                "grade": "INCOMPLETE",
                "data_status": f"MISSING: {', '.join(ig['missing'])}",
                "edge_summary": f"Cannot grade — missing {', '.join(ig['missing'])} data.",
                "peter_zone": "Waiting for complete line data before grading.",
                "composite_score": 0,
                "tags": ["INCOMPLETE"],
            })

        # ===== GPT GRADE VALIDATION GATE =====
        # Server-side verification: recalculate composite_score from matrix_scores,
        # then map to correct grade using thresholds. Override GPT if math doesn't match.
        gate_grade_overrides = 0
        gate_score_overrides = 0
        matrix = SPORT_MATRICES.get(sport_lower, [])
        if matrix:
            max_possible = sum(w for _, w, _ in matrix) * 10
            for game in all_analyzed_games:
                ms = game.get("matrix_scores", {})
                if not ms:
                    continue

                # Step 1: Recalculate composite_score from matrix_scores
                recalc_sum = 0
                valid_scores = 0
                for var_key, var_data in ms.items():
                    if isinstance(var_data, dict):
                        w = var_data.get("weight", 0)
                        s = var_data.get("score", 0)
                        weighted = w * s
                        if var_data.get("weighted") != weighted:
                            var_data["weighted"] = weighted
                        recalc_sum += weighted
                        valid_scores += 1

                if valid_scores == 0:
                    continue

                recalc_composite = round(recalc_sum / max_possible * 10, 1)
                gpt_composite = game.get("composite_score", 0)

                # Override composite_score if GPT's math is off by > 0.2
                if abs(recalc_composite - gpt_composite) > 0.2:
                    logger.warning(f"Grade gate [{sport_lower}] {game.get('matchup','?')}: "
                                   f"GPT score {gpt_composite} → recalculated {recalc_composite}")
                    game["composite_score"] = recalc_composite
                    game["_score_override"] = {"gpt_claimed": gpt_composite, "server_calculated": recalc_composite}
                    gate_score_overrides += 1

                # Step 2: Map verified composite_score → correct grade
                verified_score = game["composite_score"]
                if game.get("grade") in ("TBD", "INCOMPLETE"):
                    continue

                if verified_score >= 9.0:
                    correct_grade = "A+"
                elif verified_score >= 8.2:
                    correct_grade = "A"
                elif verified_score >= 7.5:
                    correct_grade = "A-"
                elif verified_score >= 7.0:
                    correct_grade = "B+"
                elif verified_score >= 6.5:
                    correct_grade = "B"
                elif verified_score >= 6.0:
                    correct_grade = "B-"
                elif verified_score >= 5.5:
                    correct_grade = "C+"
                elif verified_score >= 4.5:
                    correct_grade = "C"
                elif verified_score >= 3.0:
                    correct_grade = "D"
                else:
                    correct_grade = "F"

                gpt_grade = game.get("grade", "")
                if gpt_grade != correct_grade:
                    logger.warning(f"Grade gate [{sport_lower}] {game.get('matchup','?')}: "
                                   f"GPT grade '{gpt_grade}' → corrected '{correct_grade}' (score: {verified_score})")
                    game["grade"] = correct_grade
                    game["_grade_override"] = {"gpt_claimed": gpt_grade, "server_assigned": correct_grade}
                    gate_grade_overrides += 1

        if gate_grade_overrides or gate_score_overrides:
            logger.info(f"Grade gate [{sport_lower}]: {gate_score_overrides} score overrides, {gate_grade_overrides} grade overrides across {len(all_analyzed_games)} games")
        # ===== END GRADE VALIDATION GATE =====

        # ===== CHALLENGER MODEL OF THE WEEK =====
        challenger_model_display = ""
        try:
            challenger = _get_weekly_challenger()
            if challenger and batch_prompts:
                challenger_model_display = challenger["display"]
                logger.info(f"[CHALLENGER] Running {challenger['display']} on batch 0 ({len(batch_prompts)} total batches)")
                try:
                    challenger_result = await asyncio.wait_for(
                        asyncio.to_thread(_call_challenger, batch_prompts[0], 0),
                        timeout=60  # 60s hard cutoff — never block the card
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[CHALLENGER] {challenger['display']} timed out after 60s — skipping")
                    challenger_result = None
                if challenger_result:
                    matched = 0
                    for cg in challenger_result.get("games", []):
                        c_matchup = (cg.get("matchup", "")).replace(" ", "").upper()
                        for game in all_analyzed_games:
                            g_matchup = (game.get("matchup", "")).replace(" ", "").upper()
                            if c_matchup == g_matchup:
                                game["challenger_grade"] = cg.get("grade", "")
                                game["challenger_score"] = cg.get("composite_score", 0)
                                game["challenger_model"] = challenger["display"]
                                matched += 1
                                break
                    logger.info(f"[CHALLENGER] Matched {matched}/{len(challenger_result.get('games', []))} games from {challenger['display']}")
                else:
                    logger.warning("[CHALLENGER] No result returned")
        except Exception as e:
            logger.warning(f"[CHALLENGER] Failed: {e}")
        # ===== END CHALLENGER MODEL =====

        analysis = {
            "gotcha": gotcha_html or "Analysis partially generated. Some batches may have failed.",
            "games": all_analyzed_games,
            "sport": sport.upper(),
            "generated_at": _now_ts(),
            "source": f"Azure OpenAI ({ANALYSIS_MODE}: {ANALYSIS_THINKER}+{ANALYSIS_FORMATTER})" if ANALYSIS_MODE == "twomodel" else f"Azure OpenAI ({AZURE_MODEL})",
            "books_used": odds_data.get("books_used", []),
            "games_complete": len(complete_games),
            "games_incomplete": len(incomplete_games),
            "games_analyzed": len(all_analyzed_games),
            "batches": len(game_batches),
            "injury_source": "CBS Sports + ESPN + RotoWire + API-Sports" if injury_context else "none",
            "injury_data_length": len(injury_context),
            "matrix": sport_lower in SPORT_MATRICES,
            "player_validation": len(validation_log),
            "game_logs_injected": bool(game_log_context),
            "grade_overrides": sum(1 for g in all_analyzed_games if g.get("gpt_original_grade") and g.get("gpt_original_grade") != g.get("grade")),
            "grade_stats": grade_log,
            "matrix_retried": len(missing_ms_games) if missing_ms_games else 0,
            "grade_gate": {"score_overrides": gate_score_overrides, "grade_overrides": gate_grade_overrides},
            "challenger_model": challenger_model_display,
        }

        _set_cache(cache_key, analysis)
        # Save to disk — user-specific analyses get separate files, don't overwrite defaults
        if user_id:
            user_cache_path = _user_analysis_cache_path(sport_lower, user_id)
            with open(user_cache_path, "w") as f:
                json.dump({"sport": sport.upper(), **analysis, "cached_at": _now_ts(), "user_id": user_id}, f)
            _increment_user_rate_limit(user_id, sport_lower)
            analysis["user_id"] = user_id
        else:
            _save_analysis_cache(sport_lower, analysis)
        return JSONResponse(analysis)

    except json.JSONDecodeError:
        # Serve stale cache rather than returning empty
        stale = _load_analysis_cache(sport_lower)
        if stale and stale.get("games"):
            stale["cached"] = True
            stale["_recovery"] = "json_decode_error"
            return JSONResponse(stale)
        return JSONResponse({
            "sport": sport.upper(),
            "gotcha": "Analysis generation returned invalid format. Refresh to retry.",
            "games": [],
            "generated_at": _now_ts(),
        })
    except Exception as e:
        # Serve stale cache rather than returning error
        stale = _load_analysis_cache(sport_lower)
        if stale and stale.get("games"):
            stale["cached"] = True
            stale["_recovery"] = f"exception: {str(e)}"
            return JSONResponse(stale)
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


async def _update_pick_result(pick_id: str, result: str, prop_data: dict | None = None):
    """Update a pick's result in local file storage + Postgres.

    prop_data (optional): enrichment from _grade_prop_pick() —
        {actual_value, line, stat, player, pct_over}
    """
    if not pick_id:
        return
    data = _read_picks()
    for pick in data["picks"]:
        if pick.get("id") == pick_id:
            pick["result"] = result
            graded_at = datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S")
            pick["graded_at"] = graded_at
            # Persist prop enrichment on the pick object
            if prop_data:
                pick["actual_value"] = prop_data.get("actual_value")
                pick["prop_line"] = prop_data.get("line")
                pick["prop_stat"] = prop_data.get("stat")
                pick["prop_player"] = prop_data.get("player")
                pick["pct_over"] = prop_data.get("pct_over")
            data["updated_at"] = datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S")
            _write_picks(data)
            # Dual-write grade to Postgres
            try:
                if prop_data:
                    await db.execute(
                        "UPDATE picks SET result=$1, graded_at=$2, actual_value=$3, prop_line=$4, prop_stat=$5, prop_player=$6, pct_over=$7 WHERE id=$8",
                        result,
                        datetime.strptime(graded_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=PST),
                        prop_data.get("actual_value"),
                        prop_data.get("line"),
                        prop_data.get("stat"),
                        prop_data.get("player"),
                        prop_data.get("pct_over"),
                        pick_id,
                    )
                else:
                    await db.execute(
                        "UPDATE picks SET result=$1, graded_at=$2 WHERE id=$3",
                        result,
                        datetime.strptime(graded_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=PST),
                        pick_id,
                    )
            except Exception as e:
                logger.error(f"DB dual-write grade failed for {pick_id}: {e}")
            return


@app.get("/api/challenger")
async def get_challenger():
    """Return current week's challenger model and full rotation schedule."""
    challenger = _get_weekly_challenger()
    week = datetime.now(PST).isocalendar()[1]
    return JSONResponse({
        "model": challenger["name"] if challenger else None,
        "display": challenger["display"] if challenger else None,
        "week": week,
        "rotation": [{"name": m["name"], "display": m["display"]} for m in CHALLENGER_MODELS],
    })


@app.get("/api/picks")
async def get_picks(date: str = ""):
    """Return picks, optionally filtered by date (YYYY-MM-DD or 'today')."""
    if date == "today":
        date = datetime.now(PST).strftime("%Y-%m-%d")

    data = _read_picks()
    picks = data.get("picks", [])
    if date:
        picks = [p for p in picks if p.get("date", "").startswith(date)]
    return JSONResponse({"picks": picks, "count": len(picks)})


@app.post("/api/picks/seed")
async def seed_picks(request: Request):
    """One-time migration: seed picks from Supabase export into local file."""
    body = await request.json()
    incoming = body.get("picks", [])
    if not incoming:
        return JSONResponse({"error": "No picks to seed"}, status_code=400)
    data = _read_picks()
    existing_ids = {p.get("id") for p in data["picks"]}
    added = 0
    for p in incoming:
        if p.get("id") not in existing_ids:
            data["picks"].append(p)
            added += 1
    data["picks"].sort(key=lambda x: x.get("created_at", ""), reverse=True)
    data["updated_at"] = datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S")
    _write_picks(data)
    return JSONResponse({"status": "ok", "added": added, "total": len(data["picks"])})


@app.get("/api/picks/unique")
async def get_unique_picks():
    """Return scored unique/contrarian picks from graded history."""
    data = _read_picks()
    wins = [p for p in data.get("picks", []) if p.get("result") == "W"]

    graded_results = [{"pick_id": p.get("id"), "result": "W"} for p in wins]
    unique = _score_unique_picks(graded_results, wins)
    return JSONResponse({"unique_picks": unique, "count": len(unique)})


def _normalize_pick_type(raw: str) -> str:
    """Normalize pick type to canonical values: Moneyline, Spread, Over/Under, Prop, Parlay."""
    t = raw.strip().lower()
    if t in ("ml", "moneyline", "money line", "money_line"):
        return "Moneyline"
    if t in ("spread", "ats", "point spread"):
        return "Spread"
    if t in ("o/u", "over/under", "total", "totals", "over", "under"):
        return "Over/Under"
    if t in ("prop", "props", "player prop", "player_prop"):
        return "Prop"
    if t in ("parlay", "parlays"):
        return "Parlay"
    return "Spread"  # default


def _normalize_sport(raw: str) -> str:
    """Normalize sport to uppercase canonical values."""
    s = raw.strip().upper()
    # Map common variants
    aliases = {"PROPS": "NBA", "BASKETBALL": "NBA", "HOCKEY": "NHL", "FOOTBALL": "NFL",
               "COLLEGE BASKETBALL": "NCAAB", "COLLEGE FOOTBALL": "NCAAF", "UFC": "MMA",
               "FIGHT": "MMA", "FIGHTS": "MMA", "FÚTBOL": "SOCCER"}
    return aliases.get(s, s)


@app.post("/api/picks")
async def save_pick(request: Request):
    """Save a new pick from the bet slip popup."""
    body = await request.json()
    crew = _get_crew(request)
    name = crew["display_name"] if crew else body.get("name", "")
    matchup = body.get("matchup", "")
    selection = body.get("selection", "")

    logger.info(f"Pick submission: name={name}, matchup={matchup}, selection={selection}")

    if not name or not matchup or not selection:
        logger.warning(f"Pick rejected — missing fields: name={bool(name)}, matchup={bool(matchup)}, selection={bool(selection)}")
        return JSONResponse({"error": "name, matchup, and selection are required"}, status_code=400)

    pick = {
        "id": str(uuid.uuid4())[:8],
        "name": _sanitize(name),
        "sport": _normalize_sport(body.get("sport", "")),
        "type": _normalize_pick_type(body.get("type", "Spread")),
        "matchup": _sanitize(matchup),
        "selection": _sanitize(selection),
        "odds": _sanitize(body.get("odds", "-110")),
        "units": _sanitize(body.get("units", "1")),
        "confidence": _sanitize(body.get("confidence", "Lean")),
        "notes": _sanitize(body.get("notes", "")),
        "date": _sanitize(body.get("date", "")) or datetime.now(PST).strftime("%Y-%m-%d"),
        "time": datetime.now(PST).strftime("%I:%M %p"),
        "placed": True,
        "placed_at": datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S"),
        "result": None,
        "graded_at": None,
        "created_at": datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        data = _read_picks()
        # Dedup check: reject if same name+date+selection+matchup already exists
        new_key = _pick_key(pick)
        for existing in data.get("picks", []):
            if _pick_key(existing) == new_key:
                logger.info(f"Duplicate pick rejected: {new_key}")
                return JSONResponse({"status": "ok", "pick": existing, "duplicate": True})
        data["picks"].insert(0, pick)
        data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _write_picks(data)
    except Exception as e:
        logger.error(f"Pick save failed: {e}")
        log_error("Picks", f"Save failed for {name}", str(e))
        return JSONResponse({"error": f"Failed to save pick: {e}"}, status_code=500)

    # Dual-write to Postgres (fire-and-forget — JSON is still source of truth)
    try:
        await db.execute(
            """INSERT INTO picks (id, name, sport, type, matchup, selection, odds, units, confidence, notes, date, time, placed, placed_at, created_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
               ON CONFLICT (id) DO NOTHING""",
            pick["id"], pick["name"], pick["sport"], pick["type"],
            pick["matchup"], pick["selection"], pick["odds"], pick["units"],
            pick["confidence"], pick["notes"],
            datetime.strptime(pick["date"], "%Y-%m-%d").date() if pick.get("date") else None,
            pick.get("time"), pick.get("placed", True),
            datetime.strptime(pick["placed_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=PST) if pick.get("placed_at") else None,
            datetime.strptime(pick["created_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=PST) if pick.get("created_at") else None,
        )
    except Exception as e:
        logger.error(f"DB dual-write pick failed: {e}")

    return JSONResponse({"status": "ok", "pick": pick})


@app.post("/api/picks/normalize")
async def normalize_picks():
    """One-time cleanup: normalize all pick types and sports in historical data."""
    data = _read_picks()
    fixed_type = 0
    fixed_sport = 0
    for pick in data.get("picks", []):
        old_type = pick.get("type", "")
        new_type = _normalize_pick_type(old_type)
        if old_type != new_type:
            pick["type"] = new_type
            fixed_type += 1
        old_sport = pick.get("sport", "")
        new_sport = _normalize_sport(old_sport)
        if old_sport != new_sport:
            pick["sport"] = new_sport
            fixed_sport += 1
    if fixed_type or fixed_sport:
        _write_picks(data)
    return JSONResponse({"status": "ok", "fixed_type": fixed_type, "fixed_sport": fixed_sport, "total_picks": len(data.get("picks", []))})


@app.post("/api/picks/place")
async def place_pick(request: Request):
    """Mark a pick as placed on Hard Rock Bet."""
    body = await request.json()
    pick_id = body.get("id", "")
    placed = body.get("placed", True)

    if not pick_id:
        return JSONResponse({"error": "id is required"}, status_code=400)

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

    if not pick_id:
        return JSONResponse({"error": "id is required"}, status_code=400)
    if result_val not in ("W", "L", "P", ""):
        return JSONResponse({"error": "result must be W, L, P, or empty to ungrade"}, status_code=400)
    # Empty string = ungrade (reset to null)
    if result_val == "":
        result_val = None

    data = _read_picks()
    for pick in data["picks"]:
        if pick["id"] == pick_id:
            pick["result"] = result_val
            pick["graded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if result_val else None
            data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _write_picks(data)
            return JSONResponse({"status": "ok", "pick": pick})

    return JSONResponse({"error": "Pick not found"}, status_code=404)


@app.get("/api/scores")
async def get_scores():
    """Live scoreboard — all sports, today's games with scores."""
    today_pst = datetime.now(PST).strftime("%Y-%m-%d")
    yesterday_pst = (datetime.now(PST) - timedelta(days=1)).strftime("%Y-%m-%d")
    active_sports = ["nba", "nhl", "mlb", "wnba", "ncaab", "soccer", "mma", "boxing"]
    fetch_tasks = []
    fetch_sports = []
    for sport in active_sports:
        for key in SPORT_KEYS.get(sport, []):
            fetch_tasks.append(_fetch_scores(key))
            fetch_sports.append(sport)

    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    games = []
    for sport, result in zip(fetch_sports, results):
        if isinstance(result, Exception) or not result:
            continue
        for g in result:
            commence = g.get("commence_time", "")
            try:
                game_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                game_date = game_dt.astimezone(PST).strftime("%Y-%m-%d")
                game_time = game_dt.astimezone(PST).strftime("%I:%M %p")
            except (ValueError, TypeError):
                game_date = ""
                game_time = ""
            # Show today's games + yesterday's completed (final scores)
            is_today = game_date == today_pst
            is_yesterday_final = game_date == yesterday_pst and g.get("completed", False)
            if not is_today and not is_yesterday_final:
                continue
            scores = g.get("scores") or []
            score_map = {}
            for s in scores:
                score_map[s["name"]] = s.get("score", "")
            is_combat = sport in ("mma", "boxing")
            away_team = g.get("away_team", "")
            home_team = g.get("home_team", "")
            away_score = score_map.get(away_team, "")
            home_score = score_map.get(home_team, "")
            # Combat sports: Odds API returns scores as "0"/"1" (loser/winner)
            # or null. Translate to "W"/"L" for display.
            winner = ""
            if is_combat and g.get("completed", False):
                if away_score == "1" and home_score == "0":
                    winner = away_team
                elif home_score == "1" and away_score == "0":
                    winner = home_team
                # If scores are null/empty for completed fight, winner unknown
                away_score = "W" if winner == away_team else ("L" if winner else "")
                home_score = "W" if winner == home_team else ("L" if winner else "")
            games.append({
                "sport": sport.upper(),
                "away": away_team,
                "home": home_team,
                "away_score": away_score,
                "home_score": home_score,
                "completed": g.get("completed", False),
                "time": game_time,
                "date": game_date,
                "is_yesterday": game_date == yesterday_pst,
                "commence_time": commence,
                "is_combat": is_combat,
                "winner": winner,
            })

    games.sort(key=lambda x: x.get("commence_time", ""))
    return JSONResponse({
        "date": today_pst,
        "games": games,
        "count": len(games),
        "timestamp": _now_ts(),
    })


async def _fetch_scores(sport_key: str, days_from: int = 3) -> list:
    """Fetch game scores from The Odds API + local archive. Cached 5 min.

    1. Load archived completed scores (free — local file)
    2. Fetch fresh scores from API (daysFrom = dynamic based on oldest pick)
    3. Archive any newly completed games
    4. Merge and dedupe by game ID
    """
    cache_key = f"scores:{sport_key}:{days_from}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    # Step 1: Pull archived completed games for this sport
    archive = _read_scores_archive()
    archived_games = [g for g in archive.values()
                      if g.get("sport_key", "") == sport_key or sport_key in g.get("sport_key", "")]

    # Step 2: Fetch fresh from API
    api_games = []
    if ODDS_API_KEY:
        url = f"{ODDS_API_BASE}/{sport_key}/scores/"
        params = {"apiKey": ODDS_API_KEY, "daysFrom": days_from}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(url, params=params)
                logger.info(f"Scores fetch {sport_key} (daysFrom={days_from}): status={r.status_code}")
                if r.status_code == 200:
                    api_games = r.json()
                    # Tag with sport_key for archive filtering
                    for g in api_games:
                        if "sport_key" not in g:
                            g["sport_key"] = sport_key
                    logger.info(f"Scores fetch {sport_key}: got {len(api_games)} games")
                    # Step 3: Archive newly completed games
                    _archive_completed_games(api_games)
                else:
                    logger.warning(f"Scores fetch {sport_key}: HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"Scores fetch failed for {sport_key}: {type(e).__name__}: {e}")
            log_error("Scores", f"Fetch failed: {sport_key}", str(e))
    else:
        logger.warning("No ODDS_API_KEY configured for scores fetch")

    # Step 4: Merge API + archive, dedupe by game ID (API wins on conflicts)
    merged = {}
    for g in archived_games:
        gid = g.get("id")
        if gid:
            merged[gid] = g
    for g in api_games:
        gid = g.get("id")
        if gid:
            merged[gid] = g  # API data is fresher

    result = list(merged.values())
    _set_cache(cache_key, result)
    return result


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
    "ana": "ducks", "ari": "coyotes", "bos": "bruins", "bru": "bruins",
    "buf": "sabres", "car": "hurricanes", "cgy": "flames",
    "cbj": "blue jackets", "col": "avalanche", "dal": "stars",
    "edm": "oilers", "fla": "panthers", "lak": "kings",
    "min": "wild", "mon": "canadiens", "mtl": "canadiens", "njd": "devils",
    "nsh": "predators", "nyi": "islanders", "nyr": "rangers", "ott": "senators",
    "pit": "penguins", "sea": "kraken", "sj": "sharks", "sjs": "sharks",
    "stl": "blues", "tb": "lightning",
    "tbl": "lightning", "uta": "utah hockey club", "van": "canucks",
    "vgk": "golden knights",
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
        else:
            expanded.append(clean)
    return " ".join(expanded)


# ===== ESPN BOX SCORE FETCHER + PROP GRADING =====
_ESPN_SPORT_PATHS = {
    "nba": "basketball/nba",
    "wnba": "basketball/wnba",
    "nhl": "hockey/nhl",
    "nfl": "football/nfl",
    "mlb": "baseball/mlb",
}

# Map prop stat keywords to ESPN stat header names
_PROP_STAT_TO_ESPN = {
    # NBA
    "points": "PTS", "pts": "PTS",
    "rebounds": "REB", "reb": "REB",
    "assists": "AST", "ast": "AST",
    "threes": "3PM", "3pm": "3PM", "three pointers": "3PM", "3pt": "3PM",
    "steals": "STL", "stl": "STL",
    "blocks": "BLK", "blk": "BLK",
    "turnovers": "TO", "to": "TO",
    # NHL
    "goals": "G", "saves": "SV", "shots": "SOG",
    # MLB
    "strikeouts": "K", "hits": "H", "runs": "R", "rbi": "RBI",
    "total bases": "TB", "hrs": "HR", "home runs": "HR",
    "walks": "BB",
    # NFL
    "pass yds": "YDS", "passing yards": "YDS",
    "rush yds": "YDS", "rushing yards": "YDS",
    "reception yds": "YDS", "receiving yards": "YDS",
    "touchdowns": "TD", "pass tds": "TD",
    "completions": "CMP", "receptions": "REC",
}


async def _fetch_espn_box_score(sport: str, home_team: str, away_team: str, game_date: str) -> dict | None:
    """Fetch player stats from ESPN game summary.

    Returns {normalized_player_name: {stat_name: value, ...}} or None on failure.
    """
    espn_path = _ESPN_SPORT_PATHS.get(sport.lower())
    if not espn_path:
        return None

    cache_key = f"espn_box:{sport}:{home_team}:{away_team}:{game_date}"
    cached = _get_cached(cache_key, ttl=3600)  # Cache 1 hour
    if cached is not None:
        return cached

    date_fmt = game_date.replace("-", "")  # YYYYMMDD

    try:
        async with httpx.AsyncClient(timeout=15, headers={"User-Agent": "Mozilla/5.0"}) as client:
            # Step 1: Get scoreboard for the date
            scoreboard_url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_path}/scoreboard?dates={date_fmt}"
            resp = await client.get(scoreboard_url)
            if resp.status_code != 200:
                logger.warning(f"ESPN scoreboard fetch failed: {resp.status_code}")
                return None

            scoreboard = resp.json()
            events = scoreboard.get("events", [])

            # Step 2: Find the matching game
            game_id = None
            home_norm = _normalize_team(home_team)
            away_norm = _normalize_team(away_team)
            for event in events:
                competitors = event.get("competitions", [{}])[0].get("competitors", [])
                if len(competitors) < 2:
                    continue
                teams_in_event = [_normalize_team(c.get("team", {}).get("displayName", "")) for c in competitors]
                # Check if both teams match
                home_match = any(t in home_norm or home_norm in t for t in teams_in_event)
                away_match = any(t in away_norm or away_norm in t for t in teams_in_event)
                if home_match and away_match:
                    game_id = event.get("id")
                    break

            if not game_id:
                logger.info(f"ESPN: No matching game for {away_team} @ {home_team} on {game_date}")
                _set_cache(cache_key, {})  # Cache empty to avoid re-fetching
                return {}

            # Step 3: Fetch game summary with box score
            summary_url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_path}/summary?event={game_id}"
            summary_resp = await client.get(summary_url)
            if summary_resp.status_code != 200:
                logger.warning(f"ESPN summary fetch failed: {summary_resp.status_code}")
                return None

            summary = summary_resp.json()

            # Step 4: Parse player stats from boxscore
            player_stats = {}
            boxscore = summary.get("boxscore", {})
            for team_group in boxscore.get("players", []):
                for stat_group in team_group.get("statistics", []):
                    stat_names = stat_group.get("names", [])  # ["MIN", "FG", "3PT", ... "PTS"]
                    for athlete in stat_group.get("athletes", []):
                        player_name = athlete.get("athlete", {}).get("displayName", "")
                        if not player_name:
                            continue
                        stats_arr = athlete.get("stats", [])
                        # Build stat dict mapping header name -> value
                        pstats = {}
                        for i, name in enumerate(stat_names):
                            if i < len(stats_arr):
                                # Try to parse as float
                                try:
                                    val = stats_arr[i]
                                    # Handle fractional stats like "5-10" (FG made-attempted)
                                    if "-" in str(val) and not str(val).startswith("-"):
                                        pstats[name] = float(str(val).split("-")[0])
                                    else:
                                        pstats[name] = float(val)
                                except (ValueError, TypeError):
                                    pstats[name] = str(stats_arr[i])
                        player_stats[player_name.lower()] = pstats

            logger.info(f"ESPN box score: {away_team} @ {home_team} — {len(player_stats)} players")
            _set_cache(cache_key, player_stats)
            return player_stats

    except Exception as e:
        logger.warning(f"ESPN box score fetch failed: {e}")
        return None


def _grade_prop_pick(pick: dict, player_stats: dict) -> dict | None:
    """Grade a player prop pick against actual ESPN box score stats.

    Returns dict with result + enrichment data, or None if can't grade.
    {result, actual_value, line, stat, player, over_under}
    """
    selection = pick.get("selection", "")
    if not selection:
        return None

    sel_lower = selection.lower().strip()

    # Parse: "Player Name Over/Under XX.X Stat"
    # Patterns: "Tatum Over 20.5 Points", "Over 20.5 Points - Tatum", "Jayson Tatum O 20.5 Pts"
    over_under = None
    if "over" in sel_lower or " o " in f" {sel_lower} ":
        over_under = "over"
    elif "under" in sel_lower or " u " in f" {sel_lower} ":
        over_under = "under"

    if not over_under:
        return None

    # Extract the line number
    line_match = re.search(r'(\d+\.?\d*)', selection)
    if not line_match:
        return None
    line = float(line_match.group(1))

    # Determine stat type from selection text
    espn_stat = None
    for keyword, espn_key in _PROP_STAT_TO_ESPN.items():
        if keyword in sel_lower:
            espn_stat = espn_key
            break

    if not espn_stat:
        # Try common abbreviations in the selection
        for abbr in ["pts", "reb", "ast", "stl", "blk", "3pm"]:
            if abbr in sel_lower:
                espn_stat = _PROP_STAT_TO_ESPN.get(abbr)
                break

    if not espn_stat:
        return None

    # Find the player in box score — fuzzy last name match
    # Strip the over/under and number from selection to isolate player name
    player_text = re.sub(r'\b(over|under|o|u)\b', '', sel_lower, flags=re.IGNORECASE)
    player_text = re.sub(r'\d+\.?\d*', '', player_text)
    # Remove stat keywords
    for keyword in list(_PROP_STAT_TO_ESPN.keys()) + ["pts", "reb", "ast", "stl", "blk", "3pm", "pra"]:
        player_text = player_text.replace(keyword, "")
    player_text = re.sub(r'[()+-]', '', player_text).strip()
    player_words = [w for w in player_text.split() if len(w) > 1]

    if not player_words:
        return None

    # Try to match player: exact match first, then last name
    matched_player = None
    for pname in player_stats:
        # Exact name match
        if all(w in pname for w in player_words):
            matched_player = pname
            break
    if not matched_player:
        # Last name match (last word of player_words)
        last_name = player_words[-1]
        candidates = [p for p in player_stats if last_name in p]
        if len(candidates) == 1:
            matched_player = candidates[0]
        elif len(candidates) > 1:
            # Multiple matches — try first + last
            if len(player_words) >= 2:
                first = player_words[0]
                better = [p for p in candidates if first in p]
                if len(better) == 1:
                    matched_player = better[0]

    if not matched_player:
        logger.info(f"Prop grade: player not found in box score — words={player_words}")
        return None

    stats = player_stats[matched_player]
    actual_value = stats.get(espn_stat)
    if actual_value is None or not isinstance(actual_value, (int, float)):
        logger.info(f"Prop grade: stat {espn_stat} not found for {matched_player}")
        return None

    # Reverse stat label: ESPN key -> human readable
    _ESPN_TO_LABEL = {"PTS": "Points", "REB": "Rebounds", "AST": "Assists", "3PM": "Threes",
                      "STL": "Steals", "BLK": "Blocks", "TO": "Turnovers",
                      "G": "Goals", "SV": "Saves", "SOG": "Shots",
                      "K": "Strikeouts", "H": "Hits", "R": "Runs", "RBI": "RBI",
                      "TB": "Total Bases", "HR": "Home Runs", "BB": "Walks",
                      "YDS": "Yards", "TD": "Touchdowns", "CMP": "Completions", "REC": "Receptions"}

    # Grade
    if actual_value == line:
        result_str = "P"
    elif over_under == "over":
        result_str = "W" if actual_value > line else "L"
    else:  # under
        result_str = "W" if actual_value < line else "L"

    # Compute pct_over: how far actual exceeded the line (for overs) or fell below (for unders)
    if line > 0:
        if over_under == "over":
            pct_over = round((actual_value - line) / line * 100, 1)
        else:
            pct_over = round((line - actual_value) / line * 100, 1)
    else:
        pct_over = 0.0

    return {
        "result": result_str,
        "actual_value": float(actual_value),
        "line": line,
        "stat": _ESPN_TO_LABEL.get(espn_stat, espn_stat),
        "stat_key": espn_stat,
        "player": matched_player,
        "over_under": over_under,
        "pct_over": pct_over,
    }


# ============================================================
# AI PROP GRADING — Fallback when mechanical parsing fails
# ============================================================

async def _ai_grade_props(ungraded_props: list[dict], completed_games: list[dict]) -> list[dict]:
    """AI-powered prop grading fallback.

    When mechanical _grade_prop_pick() can't parse a prop (weird formatting,
    combo stats, etc.), send the props + box scores to Azure OpenAI and let
    AI figure out W/L/P.

    Returns list of {pick_id, result, player, stat, actual_value, line, over_under, pct_over}
    """
    if not AZURE_ENDPOINT or not AZURE_KEY:
        return []

    # Collect props that need AI grading + their matched box scores
    props_for_ai = []
    for pick in ungraded_props:
        if not _is_prop(pick):
            continue
        pick_sport = pick.get("sport", "").lower()
        if pick_sport not in _ESPN_SPORT_PATHS:
            continue

        # Find the matching completed game
        matchup = pick.get("matchup", "")
        selection = pick.get("selection", "")
        matched_game = None
        for game in completed_games:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            if _teams_match(matchup, home, away, pick_sport) or _teams_match(selection, home, away, pick_sport):
                if _game_date_matches_pick(pick, game):
                    matched_game = game
                    break

        if not matched_game:
            continue

        # Fetch ESPN box score for this game
        home = matched_game.get("home_team", "")
        away = matched_game.get("away_team", "")
        pick_date = pick.get("date", datetime.now(PST).strftime("%Y-%m-%d"))
        try:
            box_score = await _fetch_espn_box_score(pick_sport, home, away, pick_date)
        except Exception:
            box_score = None

        if not box_score:
            continue

        # Build a compact stat summary for the AI (top players only)
        stat_summary = {}
        for player_name, stats in box_score.items():
            # Only include players with meaningful stats
            pts = stats.get("PTS", stats.get("G", 0))
            if isinstance(pts, (int, float)) and pts > 0:
                stat_summary[player_name] = {k: v for k, v in stats.items()
                                              if isinstance(v, (int, float))}

        props_for_ai.append({
            "pick_id": pick.get("id", ""),
            "selection": selection,
            "matchup": matchup,
            "sport": pick_sport,
            "box_score": stat_summary,
        })

    if not props_for_ai:
        return []

    # Batch up to 20 props per AI call
    ai_results = []
    for batch_start in range(0, len(props_for_ai), 20):
        batch = props_for_ai[batch_start:batch_start + 20]

        # Build the prompt
        props_text = []
        for i, p in enumerate(batch):
            props_text.append(f"{i+1}. Pick: \"{p['selection']}\" | Game: {p['matchup']}")

        box_scores_text = []
        seen_games = set()
        for p in batch:
            game_key = p["matchup"]
            if game_key in seen_games:
                continue
            seen_games.add(game_key)
            box_scores_text.append(f"\n--- {game_key} ---")
            for player, stats in list(p["box_score"].items())[:30]:
                stat_parts = [f"{k}={v}" for k, v in stats.items() if k not in ("MIN",)]
                box_scores_text.append(f"  {player}: {', '.join(stat_parts)}")

        prompt = f"""Grade each player prop bet as W (win), L (loss), or P (push).

PROPS TO GRADE:
{chr(10).join(props_text)}

PLAYER STATS (from completed games):
{chr(10).join(box_scores_text)}

For each prop, compare the player's actual stat to the line. Over 24.5 points + scored 28 = W. Under 6.5 assists + had 7 = L. Exact match on the line = P.

Return ONLY a JSON array, one object per prop, in order:
[{{"pick_number": 1, "result": "W", "player": "name", "stat": "Points", "actual_value": 28, "line": 24.5, "over_under": "over"}}]

No explanation, just the JSON array."""

        try:
            url = f"{AZURE_BASE}/openai/deployments/{AZURE_MODEL}/chat/completions?api-version=2024-08-01-preview"
            headers = {"api-key": AZURE_KEY, "Content-Type": "application/json"}
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a sports betting grading system. Grade prop bets accurately against box score stats. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 2000,
            }

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, json=payload, headers=headers)

            if resp.status_code != 200:
                logger.warning(f"[AI PROP GRADE] Azure call failed: {resp.status_code}")
                continue

            resp_json = resp.json()
            content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON from response (handle markdown code blocks)
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            grades = json.loads(content)
            if not isinstance(grades, list):
                logger.warning(f"[AI PROP GRADE] Unexpected response format")
                continue

            for grade in grades:
                pick_num = grade.get("pick_number", 0)
                if pick_num < 1 or pick_num > len(batch):
                    continue
                prop = batch[pick_num - 1]
                result = grade.get("result", "").upper()
                if result not in ("W", "L", "P"):
                    continue

                actual = grade.get("actual_value", 0)
                line = grade.get("line", 0)
                over_under = grade.get("over_under", "over").lower()

                # Compute pct_over
                if line and line > 0:
                    if over_under == "over":
                        pct_over = round((actual - line) / line * 100, 1)
                    else:
                        pct_over = round((line - actual) / line * 100, 1)
                else:
                    pct_over = 0.0

                ai_results.append({
                    "pick_id": prop["pick_id"],
                    "result": result,
                    "player": grade.get("player", ""),
                    "stat": grade.get("stat", ""),
                    "actual_value": float(actual) if actual else 0.0,
                    "line": float(line) if line else 0.0,
                    "over_under": over_under,
                    "pct_over": pct_over,
                    "graded_by": "ai",
                })

            logger.info(f"[AI PROP GRADE] Graded {len(ai_results)} props via AI")

        except json.JSONDecodeError as e:
            logger.warning(f"[AI PROP GRADE] JSON parse failed: {e}")
        except Exception as e:
            logger.warning(f"[AI PROP GRADE] Error: {e}")

    return ai_results


# ============================================================
# PROP BOARD — Crushed Lines Tracker
# ============================================================

PROP_BOARD_FILE = os.path.join(DATA_DIR, "prop_board.json")


def _read_prop_board() -> dict:
    """Read prop board from persistent JSON file."""
    try:
        if os.path.exists(PROP_BOARD_FILE):
            with open(PROP_BOARD_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error reading prop board: {e}")
    return {"board": []}


def _write_prop_board(board_data: dict):
    """Write prop board to persistent JSON file."""
    try:
        os.makedirs(os.path.dirname(PROP_BOARD_FILE), exist_ok=True)
        with open(PROP_BOARD_FILE, "w") as f:
            json.dump(board_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error writing prop board: {e}")


def _maybe_add_to_prop_board(pick: dict, prop_data: dict):
    """Add to prop board if the prop crushed the line by 25%+.

    For overs: actual_value >= line * 1.25
    For unders: line - actual_value >= line * 0.25
    """
    actual = prop_data.get("actual_value", 0)
    line = prop_data.get("line", 0)
    over_under = prop_data.get("over_under", "over")
    pct_over = prop_data.get("pct_over", 0)

    if line <= 0:
        return

    # Check if it crushed by 25%+
    if over_under == "over":
        crushed = actual >= line * 1.25
    else:
        crushed = (line - actual) >= line * 0.25

    if not crushed:
        return

    pick_id = pick.get("id", "")
    board = _read_prop_board()

    # Dedup by pick_id
    if any(e.get("pick_id") == pick_id for e in board["board"]):
        return

    entry = {
        "id": f"pb_{uuid.uuid4().hex[:12]}",
        "pick_id": pick_id,
        "player": prop_data.get("player", ""),
        "sport": pick.get("sport", "").upper(),
        "stat": prop_data.get("stat", ""),
        "stat_key": prop_data.get("stat_key", ""),
        "line": line,
        "actual_value": actual,
        "pct_over": pct_over,
        "over_under": over_under,
        "matchup": pick.get("matchup", ""),
        "picked_by": pick.get("name", ""),
        "graded_date": datetime.now(PST).strftime("%Y-%m-%d"),
        "relock_count": 0,
        "last_relock": None,
    }
    board["board"].append(entry)
    _write_prop_board(board)
    logger.info(f"[PROP BOARD] Added: {entry['player']} {entry['stat']} {entry['line']} → {entry['actual_value']} (+{pct_over}%)")


# Reverse stat mapping for matching board entries to Odds API props
_STAT_KEY_TO_ODDS_API = {
    "PTS": "Points", "REB": "Rebounds", "AST": "Assists", "3PM": "Threes",
    "STL": "Steals", "BLK": "Blocks", "TO": "Turnovers",
    "G": "Goals", "SV": "Saves", "SOG": "Shots On Goal",
    "K": "Strikeouts", "H": "Hits", "R": "Runs", "RBI": "Rbi",
    "TB": "Total Bases", "HR": "Home Runs", "BB": "Walks",
    "YDS": "Passing Yards", "TD": "Passing Touchdowns", "CMP": "Completions", "REC": "Receptions",
}


def _match_board_to_fresh_odds(entry: dict, props_data: dict) -> dict | None:
    """Try to find today's fresh odds for a board entry from the props cache.

    Matches on last name + stat type. Returns {today_line, today_odds, today_book, today_matchup} or None.
    """
    board_player = entry.get("player", "").lower()
    board_stat = entry.get("stat", "")
    board_stat_key = entry.get("stat_key", "")

    # Get last name from board player (ESPN format: "J. Giddey" or "Josh Giddey")
    board_last = board_player.split()[-1] if board_player else ""
    if not board_last:
        return None

    # Map stat key to what Odds API calls it
    odds_stat_labels = set()
    if board_stat:
        odds_stat_labels.add(board_stat.lower())
    if board_stat_key and board_stat_key in _STAT_KEY_TO_ODDS_API:
        odds_stat_labels.add(_STAT_KEY_TO_ODDS_API[board_stat_key].lower())

    all_props = props_data.get("props", [])
    for prop in all_props:
        prop_player = prop.get("player", "").lower()
        prop_stat = prop.get("stat", "").lower()
        prop_side = prop.get("side", "").lower()

        # Match last name
        if board_last not in prop_player:
            continue

        # Match stat
        if not any(sl in prop_stat for sl in odds_stat_labels):
            continue

        # Match over/under direction
        if prop_side != entry.get("over_under", "over"):
            continue

        return {
            "today_line": prop.get("line"),
            "today_odds": prop.get("best_odds"),
            "today_book": prop.get("best_book"),
            "today_matchup": prop.get("matchup"),
        }

    return None


def _teams_match(pick_text: str, home: str, away: str, sport: str = "") -> bool:
    """Check if a pick's matchup references this game."""
    pt = _expand_abbrevs(_normalize_team(pick_text), sport)
    h = _expand_abbrevs(_normalize_team(home), sport)
    a = _expand_abbrevs(_normalize_team(away), sport)
    h_parts = h.split()
    a_parts = a.split()
    pt_parts = pt.split()
    h_match = any(p in pt for p in h_parts if len(p) > 2) or h in pt
    a_match = any(p in pt for p in a_parts if len(p) > 2) or a in pt
    # Reverse matching: check pick text parts against team names
    if not h_match:
        h_match = any(p in h for p in pt_parts if len(p) > 2)
    if not a_match:
        a_match = any(p in a for p in pt_parts if len(p) > 2)
    return h_match and a_match


_PROP_STATS = {"points", "assists", "rebounds", "steals", "blocks", "threes",
               "turnovers", "pts", "ast", "reb", "stl", "blk", "3pm", "pra",
               "goals", "saves", "shots", "hits", "faceoffs", "toi",
               "strikeouts", "runs", "rbi", "hrs", "walks",
               "yards", "touchdowns", "completions", "receptions"}


def _game_date_matches_pick(pick: dict, game: dict) -> bool:
    """Check that a game's commence_time is on or after the pick's date (PST).

    Prevents grading today's picks against yesterday's completed games.
    A pick made on 2026-03-06 should only match games that start on 2026-03-06 or later.
    """
    pick_date_str = pick.get("date", "")
    commence = game.get("commence_time", "")
    if not pick_date_str or not commence:
        return True  # can't verify, allow match (backward compat)
    try:
        game_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        game_date_pst = game_dt.astimezone(PST).strftime("%Y-%m-%d")
        return game_date_pst >= pick_date_str
    except (ValueError, TypeError):
        return True  # can't parse, allow match


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

    # Safety: don't grade if game hasn't actually started yet (commence_time in the future)
    commence = game.get("commence_time", "")
    if commence:
        try:
            game_start = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            if game_start > datetime.now(PST):
                logger.warning(f"Skipping grade: game {game.get('home_team')} vs {game.get('away_team')} hasn't started (commence={commence})")
                return None
        except (ValueError, TypeError):
            pass

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
            if not _teams_match(leg["selection"], home, away):
                continue
            # Don't grade today's picks against yesterday's games
            if not _game_date_matches_pick(pick, game):
                continue
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
    else:
        data = _read_picks()
        ungraded = [p for p in data.get("picks", []) if not p.get("result")]

    if not ungraded:
        return JSONResponse({"status": "ok", "graded": 0, "message": "No ungraded picks"})

    # Deduplicate picks by key fields (name + date + selection + matchup)
    ungraded = _deduplicate_picks(ungraded)

    # ── Pass 1: Grade from local scores archive (FREE — no API calls) ──
    archive = _read_scores_archive()
    archived_completed = [g for g in archive.values() if g.get("completed")]
    graded_from_cache = 0

    for game in archived_completed:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        for pick in ungraded:
            if pick.get("result") or pick.get("type", "").lower() == "parlay":
                continue
            pick_sport = pick.get("sport", "").lower()
            if not _teams_match(pick.get("matchup", ""), home, away, pick_sport):
                if not _teams_match(pick.get("selection", ""), home, away, pick_sport):
                    continue
            if not _game_date_matches_pick(pick, game):
                continue
            grade = _grade_pick_against_score(pick, game)
            if grade:
                await _update_pick_result(pick.get("id", ""), grade)
                pick["result"] = grade
                graded_from_cache += 1

    # ── Pass 2: Only fetch scores for sports with remaining ungraded picks ──
    still_ungraded = [p for p in ungraded if not p.get("result") and p.get("type", "").lower() != "parlay"]
    days_needed = _days_from_oldest_pick(ungraded)
    sports_needed = set()
    for p in still_ungraded:
        sport = p.get("sport", "").lower()
        if sport in SPORT_KEYS:
            sports_needed.add(sport)

    if not sports_needed and still_ungraded:
        sports_needed = {"nba", "wnba", "nhl"}  # fallback only if zero sports detected

    all_scores = []
    fetch_errors = []
    if sports_needed:
        logger.info(f"[AUTOGRADE] Fetching scores for: {', '.join(sorted(sports_needed))}")
        # Sequential fetches with 2s spacing to avoid rate limits
        for sport in sorted(sports_needed):
            for key in SPORT_KEYS.get(sport, []):
                label = f"{sport}/{key}"
                try:
                    result = await _fetch_scores(key, days_from=days_needed)
                    if result:
                        all_scores.extend(result)
                        logger.info(f"[AUTOGRADE] {label}: {len(result)} games")
                except Exception as e:
                    fetch_errors.append(f"{label}: {e}")
                    logger.warning(f"[AUTOGRADE] {label} failed: {e}")
                await asyncio.sleep(2)  # Space out API calls

    api_completed = [g for g in all_scores if g.get("completed")]

    # Merge archived + API completed games, dedupe by ID
    all_completed_map = {}
    for g in archived_completed:
        gid = g.get("id")
        if gid:
            all_completed_map[gid] = g
    for g in api_completed:
        gid = g.get("id")
        if gid:
            all_completed_map[gid] = g  # API wins on conflicts
    completed = list(all_completed_map.values())

    # Debug info
    debug_parts = [
        f"{len(ungraded)} ungraded",
        f"{graded_from_cache} graded from cache",
        f"{len(sports_needed)} sports ({', '.join(sports_needed) if sports_needed else 'none'})",
        f"{len(all_scores)} games fetched from API",
        f"{len(completed)} total completed",
    ]
    if fetch_errors:
        debug_parts.append(f"fetch errors: {', '.join(fetch_errors)}")

    if not completed:
        return JSONResponse({
            "status": "ok", "graded": graded_from_cache,
            "message": f"Graded {graded_from_cache} from cache, no additional completed games found",
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
        # Only route to parlay if explicitly typed as parlay (not spreads like "BOS +1")
        if pick_type == "parlay":
            result = _grade_parlay(pick, completed)
            if result:
                pick_id = pick.get("id", "")
                await _update_pick_result(pick_id, result)
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
        pick_sport = pick.get("sport", "").lower()
        for game in completed:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            if not _teams_match(matchup, home, away, pick_sport):
                # Also try matching on selection text
                if not _teams_match(selection, home, away, pick_sport):
                    continue

            # Don't grade today's picks against yesterday's games
            if not _game_date_matches_pick(pick, game):
                continue

            result = _grade_pick_against_score(pick, game)

            # --- PROP GRADING via ESPN Box Score ---
            prop_grade_data = None
            if result is None and _is_prop(pick) and game.get("completed"):
                pick_sport = pick.get("sport", "").lower()
                if pick_sport in _ESPN_SPORT_PATHS:
                    pick_date = pick.get("date", datetime.now(PST).strftime("%Y-%m-%d"))
                    try:
                        box_score = await _fetch_espn_box_score(pick_sport, home, away, pick_date)
                        if box_score:
                            prop_grade_data = _grade_prop_pick(pick, box_score)
                            if prop_grade_data:
                                result = prop_grade_data["result"]
                                logger.info(f"Prop graded via ESPN: {selection} = {result} (actual={prop_grade_data['actual_value']})")
                    except Exception as e:
                        logger.warning(f"ESPN prop grade failed: {e}")

            if result:
                pick_id = pick.get("id", "")
                await _update_pick_result(pick_id, result, prop_data=prop_grade_data)

                # Trigger prop board entry if prop crushed the line by 25%+
                if prop_grade_data and result == "W":
                    _maybe_add_to_prop_board(pick, prop_grade_data)

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

    # ── Pass 4: AI prop grading fallback for still-ungraded props ──
    ai_graded = 0
    still_ungraded_props = [p for p in ungraded if not p.get("result") and _is_prop(p)]
    if still_ungraded_props and completed:
        try:
            ai_results = await _ai_grade_props(still_ungraded_props, completed)
            for ai_grade in ai_results:
                pick_id = ai_grade["pick_id"]
                result = ai_grade["result"]
                prop_data = {
                    "result": result,
                    "actual_value": ai_grade.get("actual_value", 0),
                    "line": ai_grade.get("line", 0),
                    "stat": ai_grade.get("stat", ""),
                    "player": ai_grade.get("player", ""),
                    "over_under": ai_grade.get("over_under", "over"),
                    "pct_over": ai_grade.get("pct_over", 0),
                }
                await _update_pick_result(pick_id, result, prop_data=prop_data)
                ai_graded += 1

                # Find the pick to update its result in memory
                for p in ungraded:
                    if p.get("id") == pick_id:
                        p["result"] = result
                        break

                # Prop board entry if crushed
                if result == "W":
                    matching_pick = next((p for p in ungraded if p.get("id") == pick_id), {})
                    _maybe_add_to_prop_board(matching_pick, prop_data)

                results.append({
                    "pick_id": pick_id,
                    "selection": next((p.get("selection", "") for p in ungraded if p.get("id") == pick_id), ""),
                    "matchup": next((p.get("matchup", "") for p in ungraded if p.get("id") == pick_id), ""),
                    "result": result,
                    "final_score": f"AI graded: {ai_grade.get('player', '?')} {ai_grade.get('stat', '?')} = {ai_grade.get('actual_value', '?')}",
                })
            if ai_graded:
                logger.info(f"[AUTOGRADE] AI prop fallback graded {ai_graded} props")
        except Exception as e:
            logger.warning(f"[AUTOGRADE] AI prop grading failed: {e}")

    # Also show sample completed games for debugging
    sample_games = [f"{g['away_team']} @ {g['home_team']}" for g in completed[:5]]
    debug_parts.append(f"sample games: {', '.join(sample_games)}")

    # Score unique picks — contrarian/underdog wins get bonus grades
    unique_picks = _score_unique_picks(results, ungraded)

    total_graded = graded_count + graded_from_cache + ai_graded
    return JSONResponse({
        "status": "ok",
        "graded": total_graded,
        "graded_from_cache": graded_from_cache,
        "graded_from_ai": ai_graded,
        "total_ungraded": len(ungraded),
        "completed_games": len(completed),
        "results": results,
        "unique_picks": unique_picks,
        "debug": " | ".join(debug_parts),
    })


# ============================================================
# PROP BOARD API ENDPOINTS
# ============================================================


@app.get("/api/prop-board")
async def get_prop_board():
    """Return prop board entries enriched with today's fresh odds."""
    board = _read_prop_board()
    entries = board.get("board", [])

    # Try to enrich each entry with fresh odds from cache
    for entry in entries:
        sport = entry.get("sport", "").lower()
        cache_key = f"props:{sport}"
        cached_props = _get_cached(cache_key, ttl=PROPS_CACHE_TTL)
        if cached_props:
            fresh = _match_board_to_fresh_odds(entry, cached_props)
            if fresh:
                entry.update(fresh)

    # Sort by pct_over descending (biggest crushers first)
    entries.sort(key=lambda e: e.get("pct_over", 0), reverse=True)

    return JSONResponse({"board": entries, "count": len(entries)})


@app.post("/api/prop-board/{entry_id}/relock")
async def relock_prop_board(entry_id: str):
    """Increment relock count for a board entry."""
    board = _read_prop_board()
    for entry in board["board"]:
        if entry.get("id") == entry_id:
            entry["relock_count"] = entry.get("relock_count", 0) + 1
            entry["last_relock"] = datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S")
            _write_prop_board(board)
            return JSONResponse({"status": "ok", "entry": entry})
    return JSONResponse({"error": "Entry not found"}, status_code=404)


@app.post("/api/prop-board/manual")
async def manual_prop_board(request: Request):
    """Manually add a crushed line to the prop board."""
    body = await request.json()
    player = body.get("player", "").strip()
    if not player:
        return JSONResponse({"error": "Player name required"}, status_code=400)

    line = body.get("line", 0)
    actual = body.get("actual_value", 0)
    if not line or not actual:
        return JSONResponse({"error": "Line and actual value required"}, status_code=400)

    pct_over = body.get("pct_over", 0)
    if not pct_over and line > 0:
        over_under = body.get("over_under", "over")
        if over_under == "over":
            pct_over = round((actual - line) / line * 100, 1)
        else:
            pct_over = round((line - actual) / line * 100, 1)

    entry = {
        "id": f"pb_{uuid.uuid4().hex[:12]}",
        "pick_id": f"manual_{uuid.uuid4().hex[:8]}",
        "player": player,
        "sport": body.get("sport", "NBA").upper(),
        "stat": body.get("stat", ""),
        "stat_key": "",
        "line": line,
        "actual_value": actual,
        "pct_over": pct_over,
        "over_under": body.get("over_under", "over"),
        "matchup": body.get("matchup", ""),
        "picked_by": body.get("picked_by", "Peter"),
        "graded_date": body.get("graded_date", datetime.now(PST).strftime("%Y-%m-%d")),
        "relock_count": 0,
        "last_relock": None,
    }

    board = _read_prop_board()
    board["board"].append(entry)
    _write_prop_board(board)
    logger.info(f"[PROP BOARD] Manual add: {player} {entry['stat']} {line} → {actual} (+{pct_over}%)")

    return JSONResponse({"status": "ok", "entry": entry})


# ============================================================
# DJ FEEDBACK — Notes from DJ to Peter
# ============================================================

DJ_FEEDBACK_FILE = os.path.join(DATA_DIR, "dj_feedback.json")


def _read_dj_feedback() -> dict:
    try:
        if os.path.exists(DJ_FEEDBACK_FILE):
            with open(DJ_FEEDBACK_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error reading DJ feedback: {e}")
    return {"notes": []}


def _write_dj_feedback(data: dict):
    try:
        os.makedirs(os.path.dirname(DJ_FEEDBACK_FILE), exist_ok=True)
        with open(DJ_FEEDBACK_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error writing DJ feedback: {e}")


@app.get("/api/dj-feedback")
async def get_dj_feedback():
    """Return all DJ feedback notes, newest first."""
    data = _read_dj_feedback()
    notes = data.get("notes", [])
    notes.sort(key=lambda n: n.get("timestamp", ""), reverse=True)
    return JSONResponse({"notes": notes, "count": len(notes)})


@app.post("/api/dj-feedback")
async def post_dj_feedback(request: Request):
    """DJ submits a feedback note to Peter."""
    body = await request.json()
    message = body.get("message", "").strip()
    if not message:
        return JSONResponse({"error": "Message required"}, status_code=400)

    note = {
        "id": f"fb_{uuid.uuid4().hex[:10]}",
        "from": body.get("from", "DJ"),
        "category": body.get("category", "general"),
        "message": message,
        "reply": None,
        "timestamp": datetime.now(PST).strftime("%Y-%m-%d %H:%M"),
    }

    data = _read_dj_feedback()
    data["notes"].append(note)
    _write_dj_feedback(data)
    logger.info(f"[DJ FEEDBACK] {note['category']}: {message[:80]}")

    return JSONResponse({"status": "ok", "note": note})


@app.post("/api/dj-feedback/{note_id}/reply")
async def reply_dj_feedback(note_id: str, request: Request):
    """Peter replies to a DJ feedback note."""
    body = await request.json()
    reply = body.get("reply", "").strip()
    if not reply:
        return JSONResponse({"error": "Reply required"}, status_code=400)

    data = _read_dj_feedback()
    for note in data["notes"]:
        if note.get("id") == note_id:
            note["reply"] = reply
            note["replied_at"] = datetime.now(PST).strftime("%Y-%m-%d %H:%M")
            _write_dj_feedback(data)
            return JSONResponse({"status": "ok", "note": note})
    return JSONResponse({"error": "Note not found"}, status_code=404)


@app.post("/api/prop-board/backfill")
async def backfill_prop_board():
    """One-time: re-scan historical prop wins and populate board for 25%+ crushers."""
    data = _read_picks()
    all_picks = data.get("picks", [])

    # Find props that already have actual_value stored (from enriched grading)
    added = 0
    already_on_board = 0
    board = _read_prop_board()
    existing_pick_ids = {e.get("pick_id") for e in board["board"]}

    for pick in all_picks:
        if pick.get("result") != "W":
            continue
        if not _is_prop(pick):
            continue
        if pick.get("id") in existing_pick_ids:
            already_on_board += 1
            continue

        actual = pick.get("actual_value")
        line = pick.get("prop_line")
        if actual is None or line is None or line <= 0:
            continue

        # Detect over/under from selection text
        sel_lower = pick.get("selection", "").lower()
        if "over" in sel_lower or " o " in f" {sel_lower} ":
            over_under = "over"
        elif "under" in sel_lower or " u " in f" {sel_lower} ":
            over_under = "under"
        else:
            continue

        # Check 25% threshold
        if over_under == "over":
            crushed = actual >= line * 1.25
        else:
            crushed = (line - actual) >= line * 0.25

        if not crushed:
            continue

        pct = pick.get("pct_over", 0)
        if not pct and line > 0:
            if over_under == "over":
                pct = round((actual - line) / line * 100, 1)
            else:
                pct = round((line - actual) / line * 100, 1)

        entry = {
            "id": f"pb_{uuid.uuid4().hex[:12]}",
            "pick_id": pick.get("id", ""),
            "player": pick.get("prop_player", ""),
            "sport": pick.get("sport", "").upper(),
            "stat": pick.get("prop_stat", ""),
            "stat_key": "",
            "line": line,
            "actual_value": actual,
            "pct_over": pct,
            "over_under": over_under,
            "matchup": pick.get("matchup", ""),
            "picked_by": pick.get("name", ""),
            "graded_date": pick.get("graded_at", "")[:10] if pick.get("graded_at") else "",
            "relock_count": 0,
            "last_relock": None,
        }
        board["board"].append(entry)
        added += 1

    if added:
        _write_prop_board(board)

    return JSONResponse({
        "status": "ok",
        "added": added,
        "already_on_board": already_on_board,
        "total_board": len(board["board"]),
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

    # Add picks context (local file storage)
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
    allowed = {"selection", "odds", "units", "matchup", "notes", "confidence", "sport", "type", "date", "name"}
    update_data = {k: _sanitize(str(v)) for k, v in body.items() if k in allowed and v is not None}
    if not update_data:
        return JSONResponse({"error": "No valid fields to update"}, status_code=400)
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
    data = _read_picks()
    data["picks"] = [p for p in data["picks"] if p["id"] != pick_id]
    _write_picks(data)
    return JSONResponse({"status": "deleted", "id": pick_id})


@app.get("/api/upsets")
async def get_upsets(sport: str = "", date: str = ""):
    """Return upset picks, optionally filtered by sport and date."""
    data = _read_upsets()
    picks = data.get("picks", [])
    filter_date = date if date and date != "today" else datetime.now(PST).strftime("%Y-%m-%d")
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
        "date": datetime.now(PST).strftime("%Y-%m-%d"),
        "created_at": datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S"),
    }

    data = _read_upsets()
    data["picks"].insert(0, pick)
    _write_upsets(data)

    return JSONResponse({"status": "ok", "pick": pick})


@app.delete("/api/upsets/{pick_id}")
async def delete_upset(pick_id: str):
    """Delete an upset pick by ID."""
    data = _read_upsets()
    original_len = len(data["picks"])
    data["picks"] = [p for p in data["picks"] if p.get("id") != pick_id]
    if len(data["picks"]) == original_len:
        return JSONResponse({"error": "Pick not found"}, status_code=404)
    _write_upsets(data)
    return JSONResponse({"status": "ok"})


@app.get("/api/bankroll")
async def get_bankroll():
    """Get shared bankroll settings from disk."""
    try:
        with open(BANKROLL_FILE, "r") as f:
            data = json.load(f)
        return JSONResponse({
            "starting_balance": float(data.get("starting_balance", 1000)),
            "unit_size": float(data.get("unit_size", 25)),
            "updated_by": data.get("updated_by", ""),
        })
    except (FileNotFoundError, json.JSONDecodeError):
        return JSONResponse({"starting_balance": 1000, "unit_size": 25})


@app.post("/api/bankroll")
async def save_bankroll(request: Request):
    """Save shared bankroll settings to disk."""
    body = await request.json()
    data = {
        "starting_balance": body.get("starting_balance", 1000),
        "unit_size": body.get("unit_size", 25),
        "updated_at": datetime.now(PST).isoformat(),
        "updated_by": body.get("updated_by", ""),
    }
    try:
        with open(BANKROLL_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Bankroll save failed: {e}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)
    return JSONResponse({"status": "ok", **data})


@app.get("/api/gotcha")
async def get_gotcha():
    """Get gotcha notes (per sport) from disk."""
    try:
        with open(GOTCHA_FILE, "r") as f:
            notes = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        notes = {}
    return JSONResponse({"notes": notes})


@app.post("/api/gotcha")
async def save_gotcha_notes(request: Request):
    """Save gotcha notes for a sport to disk."""
    body = await request.json()
    sport = body.get("sport", "NBA")
    notes_text = body.get("notes", "")
    try:
        with open(GOTCHA_FILE, "r") as f:
            all_notes = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_notes = {}
    all_notes[sport] = {
        "notes": notes_text,
        "updated_by": body.get("updated_by", ""),
        "updated_at": datetime.now(PST).isoformat(),
    }
    try:
        with open(GOTCHA_FILE, "w") as f:
            json.dump(all_notes, f, indent=2)
    except Exception as e:
        logger.warning(f"Gotcha save failed: {e}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)
    return JSONResponse({"status": "ok", "sport": sport, "notes": notes_text})


@app.get("/sw.js")
async def serve_sw():
    return FileResponse("static/sw.js", media_type="application/javascript",
                        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"})

@app.get("/favicon.ico")
async def serve_favicon():
    return FileResponse("static/favicon.ico", media_type="image/x-icon", headers={"Cache-Control": "public, max-age=86400"})

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
    """Fetch current team rosters from ESPN. Cached 1 hour."""
    sport_lower = sport.lower()
    cache_key = f"rosters:{sport_lower}"
    cached = _get_cached(cache_key, ttl=3600)
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
                            "id": a.get("id", ""),
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


# ── ESPN Game Logs (WS1) ──────────────────────────────────────────────────────
async def _fetch_player_game_logs(sport, team_abbrevs):
    """Fetch last 10 game logs for top 8 players per team from ESPN.

    Returns a structured dict: {player_name: {team, gp, l10_pts, l10_reb, l10_ast,
    l10_min, season_pts, season_reb, season_ast, games_missed_last_30, minutes_trend, status}}
    """
    espn_sport = _ESPN_SPORT_PATHS.get(sport.lower())
    if not espn_sport:
        return {}

    # Get roster data with athlete IDs
    roster_cache = _get_cached(f"rosters:{sport.lower()}", ttl=3600)
    if not roster_cache:
        try:
            await get_rosters(sport)
            roster_cache = _get_cached(f"rosters:{sport.lower()}", ttl=3600)
        except Exception:
            pass
    if not roster_cache:
        return {}

    rosters = roster_cache.get("rosters", {})
    # Filter to only teams playing tonight
    target_teams = {}
    for abbr, info in rosters.items():
        team_full = info.get("team", "")
        if abbr.lower() in [t.lower() for t in team_abbrevs] or \
           team_full.lower() in [t.lower() for t in team_abbrevs]:
            target_teams[abbr] = info

    game_log_cache_key = f"gamelogs:{sport.lower()}:{','.join(sorted(t.lower() for t in team_abbrevs))}"
    cached = _get_cached(game_log_cache_key, ttl=7200)  # 2h TTL
    if cached:
        return cached

    player_logs = {}
    async with httpx.AsyncClient(timeout=15.0) as client:
        for abbr, info in target_teams.items():
            top_players = info.get("players", [])[:8]
            for player in top_players:
                pid = player.get("id", "")
                pname = player.get("name", "?")
                if not pid:
                    continue
                try:
                    url = f"https://site.web.api.espn.com/apis/common/v3/sports/{espn_sport}/athletes/{pid}/gamelog"
                    resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code != 200:
                        continue
                    data = resp.json()

                    # Parse labels to find stat indices dynamically
                    labels = [l.upper() for l in data.get("labels", [])]
                    dynamic_indices = {}
                    for i, label in enumerate(labels):
                        if label == "MIN":
                            dynamic_indices["min"] = i
                        elif label == "PTS":
                            dynamic_indices["pts"] = i
                        elif label == "REB":
                            dynamic_indices["reb"] = i
                        elif label == "AST":
                            dynamic_indices["ast"] = i

                    # Parse game log entries — categories are months (most recent first)
                    season_stats = data.get("seasonTypes", [])
                    games = []
                    # Only use regular season (first seasonType)
                    if season_stats:
                        for cat in season_stats[0].get("categories", []):
                            for event in cat.get("events", []):
                                stats_row = event.get("stats", [])
                                if stats_row:
                                    games.append(stats_row)

                    gp = len(games)
                    if gp == 0:
                        player_logs[pname] = {
                            "team": abbr, "gp": 0,
                            "note": "No game log data available"
                        }
                        continue

                    # Use dynamically parsed labels if available, fall back to NBA defaults
                    if dynamic_indices:
                        stat_indices = {
                            "min": dynamic_indices.get("min", 0),
                            "pts": dynamic_indices.get("pts", 13),
                            "reb": dynamic_indices.get("reb", 7),
                            "ast": dynamic_indices.get("ast", 8),
                        }
                    else:
                        # NBA default: MIN(0), FG(1), FG%(2), 3PT(3), 3P%(4), FT(5), FT%(6), REB(7), AST(8), BLK(9), STL(10), PF(11), TO(12), PTS(13)
                        stat_indices = {"min": 0, "pts": 13, "reb": 7, "ast": 8}

                    def _safe_float(row, idx):
                        try:
                            if idx < 0 or idx >= len(row):
                                return 0.0
                            val = row[idx]
                            if isinstance(val, str):
                                val = val.replace("--", "0").replace("-", "0")
                            return float(val)
                        except (ValueError, IndexError):
                            return 0.0

                    l10 = games[:10]
                    l10_pts = sum(_safe_float(g, stat_indices.get("pts", -1)) for g in l10) / len(l10) if l10 else 0
                    l10_reb = sum(_safe_float(g, stat_indices.get("reb", 10)) for g in l10) / len(l10) if l10 else 0
                    l10_ast = sum(_safe_float(g, stat_indices.get("ast", 11)) for g in l10) / len(l10) if l10 else 0
                    l10_min = sum(_safe_float(g, stat_indices.get("min", 0)) for g in l10) / len(l10) if l10 else 0

                    # L5 averages — used by Prop Gap Analyzer for absorption detection
                    l5 = games[:5]
                    l5_pts = sum(_safe_float(g, stat_indices.get("pts", -1)) for g in l5) / len(l5) if l5 else 0
                    l5_reb = sum(_safe_float(g, stat_indices.get("reb", 10)) for g in l5) / len(l5) if l5 else 0
                    l5_ast = sum(_safe_float(g, stat_indices.get("ast", 11)) for g in l5) / len(l5) if l5 else 0

                    season_pts = sum(_safe_float(g, stat_indices.get("pts", -1)) for g in games) / gp if gp else 0
                    season_reb = sum(_safe_float(g, stat_indices.get("reb", 10)) for g in games) / gp if gp else 0
                    season_ast = sum(_safe_float(g, stat_indices.get("ast", 11)) for g in games) / gp if gp else 0

                    # Minutes trend: compare L5 avg to L10 avg
                    l5_min = sum(_safe_float(g, stat_indices.get("min", 0)) for g in games[:5]) / min(5, len(games)) if games else 0
                    min_trend = "stable"
                    if l10_min > 0:
                        diff = (l5_min - l10_min) / l10_min
                        if diff > 0.1:
                            min_trend = "UP"
                        elif diff < -0.1:
                            min_trend = "DOWN"

                    # Build raw stat arrays for screener (floor/ceiling)
                    label_indices = {}
                    for i, label in enumerate(labels):
                        label_indices[label] = i
                    raw_stats = {}
                    for stat_name, idx in label_indices.items():
                        l5_vals = [_safe_float(g, idx) for g in games[:5]]
                        l10_vals = [_safe_float(g, idx) for g in games[:10]]
                        raw_stats[stat_name] = {
                            "l5_raw": l5_vals,
                            "l10_raw": l10_vals,
                            "l5_avg": round(sum(l5_vals) / len(l5_vals), 1) if l5_vals else 0,
                            "l10_avg": round(sum(l10_vals) / len(l10_vals), 1) if l10_vals else 0,
                            "l5_floor": min(l5_vals) if l5_vals else 0,
                            "l5_ceiling": max(l5_vals) if l5_vals else 0,
                        }

                    player_logs[pname] = {
                        "team": abbr,
                        "gp": gp,
                        "l10_pts": round(l10_pts, 1),
                        "l10_reb": round(l10_reb, 1),
                        "l10_ast": round(l10_ast, 1),
                        "l10_min": round(l10_min, 1),
                        "l5_pts": round(l5_pts, 1),
                        "l5_reb": round(l5_reb, 1),
                        "l5_ast": round(l5_ast, 1),
                        "l5_min": round(l5_min, 1),
                        "season_pts": round(season_pts, 1),
                        "season_reb": round(season_reb, 1),
                        "season_ast": round(season_ast, 1),
                        "minutes_trend": min_trend,
                        "raw_stats": raw_stats,
                    }
                except Exception as e:
                    logger.debug(f"Game log fetch failed for {pname} ({pid}): {e}")
                    continue

    if player_logs:
        _set_cache(game_log_cache_key, player_logs)
    return player_logs


def _format_game_logs_for_prompt(player_logs):
    """Format game log data as a structured table for the analysis prompt."""
    if not player_logs:
        return ""
    lines = ["\n=== PLAYER GAME LOGS (ESPN - verified, current season) ==="]
    lines.append(f"{'PLAYER':<22} {'TEAM':<5} {'GP':<4} {'L10 PTS':<9} {'L10 REB':<9} {'L10 AST':<9} {'L10 MIN':<9} {'SZN PTS':<9} {'MIN TREND':<10}")
    lines.append("-" * 95)
    # Sort by team then by L10 pts descending
    sorted_players = sorted(player_logs.items(), key=lambda x: (-x[1].get("l10_pts", 0),))
    for name, stats in sorted_players:
        if stats.get("note"):
            lines.append(f"{name:<22} {stats.get('team', '?'):<5} {'N/A':<4} {stats['note']}")
            continue
        lines.append(
            f"{name:<22} {stats.get('team', '?'):<5} {stats.get('gp', 0):<4} "
            f"{stats.get('l10_pts', 0):<9} {stats.get('l10_reb', 0):<9} "
            f"{stats.get('l10_ast', 0):<9} {stats.get('l10_min', 0):<9} "
            f"{stats.get('season_pts', 0):<9} {stats.get('minutes_trend', '?'):<10}"
        )
    return "\n".join(lines)


@app.get("/api/gamelogs/{sport}")
async def get_game_logs(sport: str):
    """Fetch player game logs for tonight's slate teams."""
    sport_lower = sport.lower()
    # Get today's teams from odds
    odds_cache_key = f"{sport_lower}:h2h,spreads,totals"
    odds_data = _get_cached(odds_cache_key)
    if not odds_data:
        odds_resp = await get_odds(sport)
        if hasattr(odds_resp, 'body'):
            odds_data = json.loads(odds_resp.body)
        else:
            odds_data = {"games": []}

    team_names = set()
    for g in odds_data.get("games", []):
        team_names.add(g.get("away", ""))
        team_names.add(g.get("home", ""))

    if not team_names:
        return JSONResponse({"sport": sport.upper(), "players": {}, "message": "No games on slate"})

    logs = await _fetch_player_game_logs(sport, list(team_names))
    return JSONResponse({
        "sport": sport.upper(),
        "players": logs,
        "count": len(logs),
        "teams": list(team_names),
        "fetched_at": _now_ts(),
    })


# ── ESPN Injuries API (WS2) ──────────────────────────────────────────────────
async def _fetch_espn_injuries(sport):
    """Fetch injuries from ESPN's official API. Returns structured text for prompt injection."""
    espn_sport = _ESPN_SPORT_PATHS.get(sport.lower())
    if not espn_sport:
        return ""

    cache_key = f"espn_injuries:{sport.lower()}"
    cached = _get_cached(cache_key, ttl=1800)  # 30min TTL
    if cached:
        return cached

    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_sport}/injuries"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                return ""
            data = resp.json()

        injury_lines = [f"\nESPN INJURY REPORT ({sport.upper()}):\n"]
        injury_count = 0
        for team_data in data.get("injuries", data.get("items", [])):
            team_name = team_data.get("displayName", team_data.get("team", {}).get("displayName", "?"))
            team_abbr = team_data.get("abbreviation", team_data.get("team", {}).get("abbreviation", "?"))
            team_injuries = []
            for injury in team_data.get("injuries", []):
                athlete = injury.get("athlete", {})
                pname = athlete.get("displayName", "?") if isinstance(athlete, dict) else "?"
                status = injury.get("status", "Unknown")
                # Details may be nested or flat
                details = injury.get("details", {})
                if isinstance(details, dict):
                    injury_type = details.get("type", "?")
                    detail = details.get("detail", "")
                    side = details.get("side", "")
                else:
                    injury_type = "?"
                    detail = str(details) if details else ""
                    side = ""
                desc = f"{injury_type}"
                if side:
                    desc = f"{side} {desc}"
                if detail:
                    desc += f" — {detail}"
                # Also grab short comment for context
                short = injury.get("shortComment", "")
                if short and len(short) < 100:
                    desc += f" ({short})"
                team_injuries.append(f"    {pname} | {status} | {desc}")
                injury_count += 1
            if team_injuries:
                injury_lines.append(f"  {team_abbr} ({team_name}):")
                injury_lines.extend(team_injuries)

        if injury_count == 0:
            return ""

        result = "\n".join(injury_lines)
        _set_cache(cache_key, result)
        return result
    except Exception as e:
        logger.warning(f"ESPN injuries fetch failed for {sport}: {e}")
        return ""


# ── Cross-Validation Gate (WS4) ──────────────────────────────────────────────
def _validate_analysis_players(analysis_games, roster_cache, injury_text=""):
    """Post-GPT validation: ensure every player mentioned exists in roster data.
    Exclude OUT players from prop suggestions. Returns cleaned games + validation log."""
    if not roster_cache:
        return analysis_games, ["WARN: No roster data available for validation"]

    rosters = roster_cache.get("rosters", {})
    # Build lookup: player_name (lowered) -> {team_abbr, team_name}
    all_players = {}
    for abbr, info in rosters.items():
        team_name = info.get("team", "")
        for p in info.get("players", []):
            all_players[p["name"].lower()] = {"abbr": abbr, "team": team_name}

    # Build OUT player set from injury text
    out_players = set()
    if injury_text:
        for line in injury_text.split("\n"):
            line_lower = line.lower()
            if "| out" in line_lower or "|out" in line_lower:
                # Extract player name (first part before |)
                parts = line.strip().split("|")
                if parts:
                    name = parts[0].strip().lstrip("- ").strip()
                    if len(name) > 3:
                        out_players.add(name.lower())

    validation_log = []
    for game in analysis_games:
        game["_player_validation"] = {"valid": [], "invalid": [], "out_excluded": []}
        props = game.get("player_props", [])
        valid_props = []
        for prop in props:
            player_name = prop.get("player", "")
            pname_lower = player_name.lower()

            # Check if player exists in any roster
            if pname_lower not in all_players:
                game["_player_validation"]["invalid"].append(player_name)
                validation_log.append(f"EXCLUDED prop: {player_name} — not found in any roster")
                continue

            # Check if player is OUT
            if pname_lower in out_players:
                game["_player_validation"]["out_excluded"].append(player_name)
                validation_log.append(f"EXCLUDED prop: {player_name} — listed as OUT")
                continue

            # Check player's team matches one of the two game teams
            matchup = game.get("matchup", "").lower()
            player_team = all_players[pname_lower]["team"].lower()
            player_abbr = all_players[pname_lower]["abbr"].lower()
            if player_abbr not in matchup and player_team not in matchup:
                # Try partial match (e.g. "76ers" in "New York Knicks @ Philadelphia 76ers")
                team_words = player_team.split()
                if not any(w.lower() in matchup for w in team_words if len(w) > 3):
                    game["_player_validation"]["invalid"].append(
                        f"{player_name} (on {all_players[pname_lower]['team']}, not in {game.get('matchup', '?')})"
                    )
                    validation_log.append(
                        f"EXCLUDED prop: {player_name} — plays for {all_players[pname_lower]['team']}, "
                        f"not in matchup {game.get('matchup', '?')}"
                    )
                    continue

            game["_player_validation"]["valid"].append(player_name)
            valid_props.append(prop)

        game["player_props"] = valid_props

    return analysis_games, validation_log


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
async def voice_token(request: Request):
    """Get ephemeral WebRTC token from Azure Realtime API.
    Accepts optional user_id in body for personalized agent sessions."""
    if not AZURE_BASE or not AZURE_KEY:
        return JSONResponse({"error": "Azure not configured"}, status_code=500)

    # Parse optional user_id from request body
    user_id = None
    try:
        body = await request.json()
        user_id = body.get("user_id")
    except Exception:
        pass

    # Build session config — personalized if user_id provided
    instructions = EV_AGENT_VOICE["prompt"]
    voice = EV_AGENT_VOICE["voice"]
    tools = []

    if user_id:
        profile = _load_user_profile(user_id)
        if profile:
            instructions = _build_agent_voice_prompt(profile)
            voice = profile.get("agent_personality", {}).get("voice", "ash")
            tools = AGENT_VOICE_TOOLS

    url = f"{AZURE_BASE}/openai/v1/realtime/client_secrets"
    headers = {"api-key": AZURE_KEY, "Content-Type": "application/json"}
    session = {
        "type": "realtime",
        "model": REALTIME_DEPLOYMENT,
        "instructions": instructions,
        "audio": {"output": {"voice": voice}}
    }
    if tools:
        session["tools"] = tools
        session["tool_choice"] = "auto"
    payload = {"session": session}

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        result = {"token": data["value"], "expires_at": data.get("expires_at"), "voice": voice}
        if user_id and tools:
            result["has_tools"] = True
        return result


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


# ── User Agent Endpoints ─────────────────────────────────────────────────────

@app.get("/agent/{user_id}")
async def agent_page(user_id: str):
    """Serve the agent page for a user."""
    return FileResponse("static/agent.html", headers=NO_CACHE_HEADERS)


@app.get("/api/agent/{user_id}/profile")
async def get_agent_profile(user_id: str):
    """Return a user's agent profile."""
    profile = _load_user_profile(user_id)
    if not profile:
        return JSONResponse({"error": "Profile not found"}, status_code=404)
    return JSONResponse(profile)


@app.put("/api/agent/{user_id}/profile")
async def update_agent_profile(user_id: str, request: Request):
    """Partial merge update of a user's agent profile."""
    profile = _load_user_profile(user_id)
    if not profile:
        return JSONResponse({"error": "Profile not found"}, status_code=404)
    updates = await request.json()
    # Merge top-level keys (shallow merge for safety)
    for key in ("display_name", "sport_preferences", "betting_style", "agent_personality"):
        if key in updates:
            profile[key] = updates[key]
    if "conversation_context" in updates:
        existing = profile.get("conversation_context", {})
        incoming = updates["conversation_context"]
        if "learned_preferences" in incoming:
            existing_prefs = existing.get("learned_preferences", [])
            for pref in incoming["learned_preferences"]:
                if pref not in existing_prefs:
                    existing_prefs.append(pref)
            existing["learned_preferences"] = existing_prefs
        profile["conversation_context"] = existing
    _save_user_profile(profile)
    return JSONResponse(profile)


@app.post("/api/agent/{user_id}/profile")
async def create_agent_profile(user_id: str, request: Request):
    """Create a new user agent profile."""
    existing = _load_user_profile(user_id)
    if existing:
        return JSONResponse({"error": "Profile already exists", "profile": existing}, status_code=409)
    body = await request.json()
    profile = {
        "user_id": user_id,
        "display_name": body.get("display_name", user_id.upper()),
        "created_at": _now_ts(),
        "updated_at": _now_ts(),
        "sport_preferences": body.get("sport_preferences", ["ncaab", "nba", "mlb"]),
        "betting_style": body.get("betting_style", {"periods": ["full"], "strategy": [], "notes": ""}),
        "matrices": body.get("matrices", {}),
        "conversation_context": body.get("conversation_context", {"learned_preferences": []}),
        "agent_personality": body.get("agent_personality", {"voice": "ash", "style": "direct"}),
    }
    _save_user_profile(profile)
    return JSONResponse(profile, status_code=201)


@app.post("/api/agent/{user_id}/action")
async def agent_action(user_id: str, request: Request):
    """Process a voice agent tool call."""
    body = await request.json()
    action_name = body.get("action") or body.get("name", "")
    args = body.get("args", body.get("arguments", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    result = _process_agent_action(user_id, action_name, args)
    return JSONResponse(result)


@app.get("/api/agent/{user_id}/matrix/{sport}")
async def get_agent_matrix(user_id: str, sport: str):
    """Return a user's resolved matrix for a sport."""
    profile = _load_user_profile(user_id)
    if not profile:
        return JSONResponse({"error": "Profile not found"}, status_code=404)
    resolved = _resolve_user_matrix(sport.lower(), profile)
    default_dict = {k: w for k, w, _ in SPORT_MATRICES.get(sport.lower(), [])}
    sport_cfg = profile.get("matrices", {}).get(sport.lower(), {})
    overrides = sport_cfg.get("overrides", {})
    excluded = sport_cfg.get("excluded", [])
    custom_keys = {cv["key"] for cv in sport_cfg.get("custom_vars", [])}

    matrix_out = []
    for key, weight, desc in resolved:
        entry = {"key": key, "weight": weight, "description": desc, "type": "default"}
        if key in custom_keys:
            entry["type"] = "custom"
        elif key in overrides:
            entry["type"] = "override"
            entry["default_weight"] = default_dict.get(key, weight)
        matrix_out.append(entry)

    return JSONResponse({
        "sport": sport.upper(),
        "user_id": user_id,
        "matrix": matrix_out,
        "excluded": excluded,
        "total_weight": sum(w for _, w, _ in resolved),
        "var_count": len(resolved),
    })


@app.get("/dj")
async def dj_page():
    return FileResponse("static/dj.html", headers=NO_CACHE_HEADERS)


@app.get("/dj/ncaab-system")
async def ncaab_system_page():
    return FileResponse("static/ncaab-system.html", headers=NO_CACHE_HEADERS)


@app.get("/dj/nba-system")
async def nba_system_page():
    return FileResponse("static/nba-system.html", headers=NO_CACHE_HEADERS)


@app.get("/api/arb-scan")
async def arb_scan():
    """Scan all active sports for arbitrage opportunities."""
    all_arbs = []
    for sport in _in_season_sports():
        odds_data = _get_cached(f"{sport}:h2h,spreads,totals")
        if not odds_data or not odds_data.get("games"):
            continue
        for g in odds_data["games"]:
            if g.get("arbs"):
                for arb in g["arbs"]:
                    all_arbs.append({
                        "sport": sport.upper(),
                        "matchup": f"{g.get('away', '?')} @ {g.get('home', '?')}",
                        "time": g.get("time", ""),
                        **arb,
                    })
    all_arbs.sort(key=lambda a: a.get("profit_pct", 0), reverse=True)
    return JSONResponse({"arbs": all_arbs, "count": len(all_arbs), "scanned_at": _now_ts()})


# ── Data Card Endpoints ────────────────────────────────────────────────────
_ESPN_SPORT_PATHS_CARDS = {
    "nba": "basketball/nba",
    "wnba": "basketball/wnba",
    "nhl": "hockey/nhl",
    "soccer": None,
}

def _parse_matchup_teams(matchup: str):
    """Parse 'AWAY @ HOME' matchup string into (away, home) tuple."""
    parts = matchup.split("@")
    if len(parts) != 2:
        parts = matchup.split(" at ")
    if len(parts) != 2:
        return None, None
    return parts[0].strip(), parts[1].strip()


@app.get("/api/card/injuries/{sport}/{matchup}")
async def card_injuries(sport: str, matchup: str):
    """Return structured injury data for a specific matchup."""
    sport_lower = sport.lower()
    away_abbr, home_abbr = _parse_matchup_teams(matchup)
    if not away_abbr or not home_abbr:
        return JSONResponse({"error": "Invalid matchup format. Use 'AWAY @ HOME'."}, status_code=400)

    away_name = _expand_abbrevs(away_abbr, sport_lower)
    home_name = _expand_abbrevs(home_abbr, sport_lower)

    # Try ESPN injuries API first
    espn_sport = _ESPN_SPORT_PATHS.get(sport_lower)
    away_injuries = []
    home_injuries = []

    if espn_sport:
        cache_key = f"espn_injuries_raw:{sport_lower}"
        cached = _get_cached(cache_key, ttl=1800)
        if not cached:
            try:
                url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_sport}/injuries"
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code == 200:
                        cached = resp.json()
                        _set_cache(cache_key, cached)
            except Exception as e:
                logger.warning(f"ESPN injuries fetch for card failed: {e}")

        if cached:
            for team_data in cached.get("injuries", cached.get("items", [])):
                team_a = (team_data.get("abbreviation") or
                          team_data.get("team", {}).get("abbreviation", "")).lower()
                team_n = (team_data.get("displayName") or
                          team_data.get("team", {}).get("displayName", "")).lower()
                is_away = (team_a == away_abbr.lower() or
                           away_name in team_n or away_abbr.lower() in team_n)
                is_home = (team_a == home_abbr.lower() or
                           home_name in team_n or home_abbr.lower() in team_n)
                if not is_away and not is_home:
                    continue
                target = away_injuries if is_away else home_injuries
                for inj in team_data.get("injuries", []):
                    athlete = inj.get("athlete", {})
                    pname = athlete.get("displayName", "?") if isinstance(athlete, dict) else "?"
                    status = inj.get("status", "Unknown")
                    details = inj.get("details", {})
                    injury_type = details.get("type", "?") if isinstance(details, dict) else "?"
                    target.append({
                        "player": pname,
                        "status": status.upper(),
                        "injury": injury_type,
                    })

    return JSONResponse({
        "away_team": away_injuries,
        "home_team": home_injuries,
        "away_name": away_abbr,
        "home_name": home_abbr,
        "source": "ESPN",
    })


@app.get("/api/card/records/{sport}/{matchup}")
async def card_records(sport: str, matchup: str):
    """Return team records for a matchup via ESPN team API."""
    sport_lower = sport.lower()
    away_abbr, home_abbr = _parse_matchup_teams(matchup)
    if not away_abbr or not home_abbr:
        return JSONResponse({"error": "Invalid matchup format."}, status_code=400)

    espn_sport = _ESPN_SPORT_PATHS.get(sport_lower)
    if not espn_sport:
        return JSONResponse({
            "away": {"name": away_abbr, "record": "N/A"},
            "home": {"name": home_abbr, "record": "N/A"},
            "note": f"No ESPN source for {sport}",
        })

    async def _fetch_team_record(abbr_str):
        """Fetch record for a single team from ESPN."""
        tid = abbr_str.lower()
        cache_key = f"espn_team_record:{sport_lower}:{tid}"
        cached = _get_cached(cache_key, ttl=3600)
        if cached:
            return cached

        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_sport}/teams/{tid}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code == 200:
                    data = resp.json()
                    team = data.get("team", {})
                    abbr_out = team.get("abbreviation", tid.upper())
                    record_items = team.get("record", {}).get("items", [])
                    overall = {}
                    home_rec = ""
                    away_rec = ""
                    streak_str = ""
                    last10 = ""
                    for item in record_items:
                        if item.get("type") == "total":
                            overall = item
                        stats = {s.get("name", ""): s.get("displayValue", s.get("value", ""))
                                 for s in item.get("stats", [])}
                        if item.get("type") == "total":
                            streak_str = stats.get("streak", "")
                            # Try to find home/away/last10 in stats
                        if item.get("type") == "home":
                            home_rec = item.get("summary", "")
                        if item.get("type") == "road" or item.get("type") == "away":
                            away_rec = item.get("summary", "")

                    result = {
                        "name": abbr_out,
                        "record": overall.get("summary", "?"),
                        "home": home_rec,
                        "away": away_rec,
                        "streak": streak_str,
                    }
                    _set_cache(cache_key, result)
                    return result
        except Exception as e:
            logger.warning(f"ESPN team record fetch failed for {tid}: {e}")

        return {"name": tid.upper(), "record": "N/A"}

    away_rec, home_rec = await asyncio.gather(
        _fetch_team_record(away_abbr),
        _fetch_team_record(home_abbr),
    )
    return JSONResponse({"away": away_rec, "home": home_rec})


# ── Team Profile + H2H + Alt Grade helpers ────────────────────────────────────

def _team_in_game(team: str, game: dict, sport: str = "") -> str:
    """Check if a team appears in a game. Returns 'home', 'away', or '' if no match."""
    home = game.get("home_team", "")
    away = game.get("away_team", "")
    t_norm = _expand_abbrevs(_normalize_team(team), sport)
    h_norm = _expand_abbrevs(_normalize_team(home), sport)
    a_norm = _expand_abbrevs(_normalize_team(away), sport)
    # Check if team matches home
    t_parts = t_norm.split()
    h_parts = h_norm.split()
    a_parts = a_norm.split()
    h_match = (t_norm == h_norm or
               any(p in t_norm for p in h_parts if len(p) > 2) or
               any(p in h_norm for p in t_parts if len(p) > 2))
    a_match = (t_norm == a_norm or
               any(p in t_norm for p in a_parts if len(p) > 2) or
               any(p in a_norm for p in t_parts if len(p) > 2))
    if h_match:
        return "home"
    if a_match:
        return "away"
    return ""


def _get_game_scores(game: dict):
    """Extract (home_score, away_score) from a game object. Returns (None, None) if unavailable."""
    scores = game.get("scores")
    if not scores or not isinstance(scores, list):
        return None, None
    home_team = game.get("home_team", "")
    away_team = game.get("away_team", "")
    home_score = None
    away_score = None
    for s in scores:
        name = s.get("name", "")
        score = s.get("score")
        if score is None:
            continue
        try:
            score_val = int(score) if isinstance(score, str) else score
        except (ValueError, TypeError):
            continue
        if name == home_team:
            home_score = score_val
        elif name == away_team:
            away_score = score_val
    return home_score, away_score


def _build_team_profile(team: str, sport: str) -> dict:
    """Build a team profile from the scores archive. Returns last 5 games + summary."""
    cache_key = f"team_profile:{sport}:{_normalize_team(team)}"
    cached = _get_cached(cache_key, ttl=1800)
    if cached is not None:
        return cached

    archive = _read_scores_archive()
    team_games = []
    for gid, game in archive.items():
        if not game.get("completed"):
            continue
        sport_key = game.get("sport_key", "")
        if sport.lower() not in sport_key:
            continue
        side = _team_in_game(team, game, sport)
        if not side:
            continue
        home_score, away_score = _get_game_scores(game)
        if home_score is None or away_score is None:
            continue
        team_games.append({
            "game": game,
            "side": side,
            "home_score": home_score,
            "away_score": away_score,
            "commence_time": game.get("commence_time", ""),
        })

    # Sort by commence_time descending, take last 5
    team_games.sort(key=lambda x: x["commence_time"], reverse=True)
    last_5_raw = team_games[:5]

    last_5 = []
    wins = 0
    losses = 0
    ats_wins = 0
    ats_losses = 0
    ats_pushes = 0
    ou_overs = 0
    ou_unders = 0
    ou_pushes = 0
    total_margin = 0

    for entry in last_5_raw:
        game = entry["game"]
        side = entry["side"]
        home_score = entry["home_score"]
        away_score = entry["away_score"]

        if side == "home":
            team_score = home_score
            opp_score = away_score
            opponent = game.get("away_team", "?")
        else:
            team_score = away_score
            opp_score = home_score
            opponent = game.get("home_team", "?")

        margin = team_score - opp_score
        result = "W" if margin > 0 else "L"
        if margin > 0:
            wins += 1
        else:
            losses += 1
        total_margin += margin

        # ATS result
        closing_spread = game.get("closing_spread")
        ats = None
        if closing_spread is not None:
            try:
                spread_val = float(closing_spread)
                # closing_spread is home_spread; adjust for team side
                if side == "home":
                    adjusted_margin = margin + spread_val
                else:
                    adjusted_margin = margin - spread_val
                if adjusted_margin > 0:
                    ats = "W"
                    ats_wins += 1
                elif adjusted_margin < 0:
                    ats = "L"
                    ats_losses += 1
                else:
                    ats = "P"
                    ats_pushes += 1
            except (ValueError, TypeError):
                pass

        # O/U result
        closing_total = game.get("closing_total")
        ou = None
        if closing_total is not None:
            try:
                total_val = float(closing_total)
                actual_total = home_score + away_score
                if actual_total > total_val:
                    ou = "O"
                    ou_overs += 1
                elif actual_total < total_val:
                    ou = "U"
                    ou_unders += 1
                else:
                    ou = "P"
                    ou_pushes += 1
            except (ValueError, TypeError):
                pass

        # Parse date from commence_time
        ct = game.get("commence_time", "")
        game_date = ct[:10] if len(ct) >= 10 else "?"

        last_5.append({
            "opponent": opponent,
            "result": result,
            "margin": margin,
            "score": f"{team_score}-{opp_score}",
            "ats": ats,
            "ou": ou,
            "date": game_date,
        })

    # Summary
    count = len(last_5)
    avg_margin = round(total_margin / count, 1) if count > 0 else 0
    # Trend: "up" if last 2 wins, "down" if last 2 losses, else "flat"
    trend = "flat"
    if len(last_5) >= 2:
        if last_5[0].get("result") == "W" and last_5[1].get("result") == "W":
            trend = "up"
        elif last_5[0].get("result") == "L" and last_5[1].get("result") == "L":
            trend = "down"

    profile = {
        "team": team,
        "sport": sport.lower(),
        "last_5": last_5,
        "summary": {
            "record": f"{wins}-{losses}",
            "ats_record": f"{ats_wins}-{ats_losses}" + (f"-{ats_pushes}" if ats_pushes else ""),
            "ou_record": f"{ou_overs}-{ou_unders}-{ou_pushes}",
            "avg_margin": avg_margin,
            "trend": trend,
        },
    }
    _set_cache(cache_key, profile)
    return profile


def _build_h2h(away_team: str, home_team: str, sport: str) -> dict:
    """Build head-to-head history from scores archive."""
    cache_key = f"h2h:{sport}:{_normalize_team(away_team)}:{_normalize_team(home_team)}"
    cached = _get_cached(cache_key, ttl=1800)
    if cached is not None:
        return cached

    archive = _read_scores_archive()
    h2h_games = []
    for gid, game in archive.items():
        if not game.get("completed"):
            continue
        sport_key = game.get("sport_key", "")
        if sport.lower() not in sport_key:
            continue
        away_side = _team_in_game(away_team, game, sport)
        home_side = _team_in_game(home_team, game, sport)
        if not away_side or not home_side:
            continue
        # Both teams in this game
        home_score, away_score = _get_game_scores(game)
        if home_score is None or away_score is None:
            continue
        h2h_games.append({
            "game": game,
            "home_score": home_score,
            "away_score": away_score,
            "commence_time": game.get("commence_time", ""),
        })

    h2h_games.sort(key=lambda x: x["commence_time"], reverse=True)
    last_5_raw = h2h_games[:5]

    meetings = []
    away_wins = 0
    home_wins = 0
    total_margin = 0

    for entry in last_5_raw:
        game = entry["game"]
        home_score = entry["home_score"]
        away_score = entry["away_score"]
        margin = away_score - home_score  # From away_team perspective

        # Determine which side our query away_team was actually on
        actual_away_side = _team_in_game(away_team, game, sport)
        if actual_away_side == "home":
            # Our "away_team" was actually the home team in this game
            team_score = home_score
            opp_score = away_score
        else:
            team_score = away_score
            opp_score = home_score
        team_margin = team_score - opp_score

        if team_margin > 0:
            away_wins += 1
            result = "W"
        else:
            home_wins += 1
            result = "L"
        total_margin += team_margin

        # ATS
        closing_spread = game.get("closing_spread")
        ats = None
        if closing_spread is not None:
            try:
                spread_val = float(closing_spread)
                if actual_away_side == "home":
                    adjusted = team_margin + spread_val
                else:
                    adjusted = team_margin - spread_val
                ats = "W" if adjusted > 0 else ("L" if adjusted < 0 else "P")
            except (ValueError, TypeError):
                pass

        # O/U
        closing_total = game.get("closing_total")
        ou = None
        if closing_total is not None:
            try:
                total_val = float(closing_total)
                actual_total = home_score + away_score
                ou = "O" if actual_total > total_val else ("U" if actual_total < total_val else "P")
            except (ValueError, TypeError):
                pass

        ct = game.get("commence_time", "")
        game_date = ct[:10] if len(ct) >= 10 else "?"

        meetings.append({
            "away": game.get("away_team", "?"),
            "home": game.get("home_team", "?"),
            "result": result,
            "margin": team_margin,
            "score": f"{away_score}-{home_score}",
            "ats": ats,
            "ou": ou,
            "date": game_date,
        })

    count = len(meetings)
    avg_margin = round(total_margin / count, 1) if count > 0 else 0

    h2h = {
        "away_team": away_team,
        "home_team": home_team,
        "meetings": meetings,
        "count": count,
        "summary": {
            "away_record": f"{away_wins}-{home_wins}",
            "home_record": f"{home_wins}-{away_wins}",
            "avg_margin": avg_margin,
        },
    }
    _set_cache(cache_key, h2h)
    return h2h


def _calculate_alt_grade(away_profile: dict, home_profile: dict, h2h: dict) -> dict:
    """Calculate alternative grade based on recent form, H2H, and momentum.

    Weights: Recent form 50%, H2H 30%, Momentum 20%.
    Returns: {"grade": "B+", "score": 7.2, "trend": "up"}
    """
    # --- Form score (50%) ---
    def _form_score(profile):
        summary = profile.get("summary", {})
        record = summary.get("record", "0-0")
        parts = record.split("-")
        try:
            wins = int(parts[0])
        except (ValueError, IndexError):
            wins = 0
        avg_margin = summary.get("avg_margin", 0)
        # Base: wins/5 * 10, adjusted by avg margin (capped effect)
        base = (wins / 5) * 10
        margin_adj = max(-2, min(2, avg_margin / 5))
        return max(1, min(10, base + margin_adj))

    away_form = _form_score(away_profile)
    home_form = _form_score(home_profile)
    # Use the higher form as the "matchup quality" signal
    form_score = (away_form + home_form) / 2

    # --- H2H score (30%) ---
    h2h_score = 5.0  # Default neutral if no H2H data
    h2h_summary = h2h.get("summary", {})
    h2h_count = h2h.get("count", 0)
    if h2h_count > 0:
        away_rec = h2h_summary.get("away_record", "0-0").split("-")
        try:
            h2h_away_wins = int(away_rec[0])
        except (ValueError, IndexError):
            h2h_away_wins = 0
        h2h_avg_margin = abs(h2h_summary.get("avg_margin", 0))
        # Competitive H2H (close games) = higher quality matchup
        competitiveness = max(0, 10 - h2h_avg_margin)
        # Split decisions also indicate quality
        split_factor = min(h2h_count, 5) / 5 * 2
        h2h_score = max(1, min(10, competitiveness + split_factor))

    # --- Momentum (20%) ---
    away_trend = away_profile.get("summary", {}).get("trend", "flat")
    home_trend = home_profile.get("summary", {}).get("trend", "flat")
    momentum_score = 5.0
    for trend in [away_trend, home_trend]:
        if trend == "up":
            momentum_score += 1.0
        elif trend == "down":
            momentum_score -= 1.0
    momentum_score = max(1, min(10, momentum_score))

    # --- Composite ---
    composite = round(form_score * 0.5 + h2h_score * 0.3 + momentum_score * 0.2, 1)
    composite = max(1.0, min(10.0, composite))

    # --- Grade using same thresholds as _recalculate_grade ---
    if composite >= 9.0:
        grade = "A+"
    elif composite >= 8.5:
        grade = "A"
    elif composite >= 7.5:
        grade = "A-"
    elif composite >= 7.0:
        grade = "B+"
    elif composite >= 6.5:
        grade = "B"
    elif composite >= 6.0:
        grade = "B-"
    elif composite >= 5.5:
        grade = "C+"
    elif composite >= 4.5:
        grade = "C"
    elif composite >= 3.0:
        grade = "D"
    else:
        grade = "F"

    # --- Trend: dominant momentum direction ---
    if away_trend == "up" and home_trend == "up":
        trend = "up"
    elif away_trend == "down" and home_trend == "down":
        trend = "down"
    elif away_trend == "up" or home_trend == "up":
        trend = "up"
    elif away_trend == "down" or home_trend == "down":
        trend = "down"
    else:
        trend = "flat"

    return {"grade": grade, "score": composite, "trend": trend}


@app.get("/api/team-profile/{sport}/{team}")
async def team_profile_endpoint(sport: str, team: str):
    """Get team profile with last 5 games, ATS/OU records, and trend."""
    sport_lower = sport.lower()
    team_name = _expand_abbrevs(team, sport_lower)
    profile = _build_team_profile(team_name, sport_lower)
    return JSONResponse(profile)


@app.get("/api/card/h2h/{sport}/{matchup}")
async def card_h2h(sport: str, matchup: str):
    """Head-to-head data for a matchup from scores archive."""
    sport_lower = sport.lower()
    away_abbr, home_abbr = _parse_matchup_teams(matchup)
    if not away_abbr or not home_abbr:
        return JSONResponse({"error": "Invalid matchup format. Use 'AWAY @ HOME'."}, status_code=400)

    away_name = _expand_abbrevs(away_abbr, sport_lower)
    home_name = _expand_abbrevs(home_abbr, sport_lower)

    away_profile = _build_team_profile(away_name, sport_lower)
    home_profile = _build_team_profile(home_name, sport_lower)
    h2h = _build_h2h(away_name, home_name, sport_lower)

    return JSONResponse({
        "matchup": matchup,
        "away_profile": away_profile,
        "home_profile": home_profile,
        "h2h": h2h,
        "source": "Scores Archive",
    })


@app.get("/api/alt-grade/{sport}/{matchup}")
async def alt_grade_endpoint(sport: str, matchup: str):
    """Calculate alternative grade based on team profiles and H2H."""
    sport_lower = sport.lower()
    away_abbr, home_abbr = _parse_matchup_teams(matchup)
    if not away_abbr or not home_abbr:
        return JSONResponse({"error": "Invalid matchup format. Use 'AWAY @ HOME'."}, status_code=400)

    cache_key = f"alt_grade:{sport_lower}:{_normalize_team(away_abbr)}:{_normalize_team(home_abbr)}"
    cached = _get_cached(cache_key, ttl=1800)
    if cached is not None:
        return JSONResponse(cached)

    away_name = _expand_abbrevs(away_abbr, sport_lower)
    home_name = _expand_abbrevs(home_abbr, sport_lower)

    away_profile = _build_team_profile(away_name, sport_lower)
    home_profile = _build_team_profile(home_name, sport_lower)
    h2h = _build_h2h(away_name, home_name, sport_lower)

    # Don't calculate alt grade if we have no game data for either team
    away_games = len(away_profile.get("last_5", []))
    home_games = len(home_profile.get("last_5", []))
    if away_games == 0 and home_games == 0:
        alt_grade = None
    else:
        alt_grade = _calculate_alt_grade(away_profile, home_profile, h2h)

    result = {
        "away_profile": away_profile,
        "home_profile": home_profile,
        "h2h": h2h,
        "alt_grade": alt_grade,
    }
    _set_cache(cache_key, result)
    return JSONResponse(result)


@app.get("/api/card/lineup/{sport}/{matchup}")
async def card_lineup(sport: str, matchup: str):
    """Return lineup data for a specific matchup."""
    sport_lower = sport.lower()
    away_abbr, home_abbr = _parse_matchup_teams(matchup)
    if not away_abbr or not home_abbr:
        return JSONResponse({"error": "Invalid matchup format."}, status_code=400)

    away_name = _expand_abbrevs(away_abbr, sport_lower)
    home_name = _expand_abbrevs(home_abbr, sport_lower)
    away_lineup = []
    home_lineup = []
    confirmed = False

    espn_sport = _ESPN_SPORT_PATHS.get(sport_lower)
    if espn_sport:
        # Try ESPN roster/depth chart as a proxy for lineups
        for tid, target, name in [(away_abbr.lower(), away_lineup, away_name),
                                   (home_abbr.lower(), home_lineup, home_name)]:
            cache_key = f"espn_roster_card:{sport_lower}:{tid}"
            cached = _get_cached(cache_key, ttl=1800)
            if cached:
                target.extend(cached)
                continue
            try:
                url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_sport}/teams/{tid}/roster"
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code == 200:
                        data = resp.json()
                        players = []
                        for a in data.get("athletes", []):
                            players.append({
                                "name": a.get("displayName", "?"),
                                "pos": a.get("position", {}).get("abbreviation", "?"),
                            })
                        _set_cache(cache_key, players)
                        target.extend(players)
            except Exception as e:
                logger.warning(f"ESPN roster card fetch failed for {tid}: {e}")

    return JSONResponse({
        "away_lineup": away_lineup[:15],
        "home_lineup": home_lineup[:15],
        "away_name": away_abbr,
        "home_name": home_abbr,
        "confirmed": confirmed,
    })


# ══════════════════════════════════════════════════════════════════════════════
# ── PROP GAP ANALYZER — Injury-Driven Edge Detection (Pure Math, No AI) ─────
# ══════════════════════════════════════════════════════════════════════════════
#
# Pipeline: detect injury gaps → find absorbers → match to props → score → grade
# Zero AI/GPT cost. Separate from main analysis pipeline. Read-only consumer of
# existing ESPN injuries, game logs, and Odds API prop caches.
# ─────────────────────────────────────────────────────────────────────────────

GAP_PROPS_CACHE_TTL = 900  # 15 minutes
GAP_PROPS_SUPPORTED = {"nba", "nhl"}

# Stat mapping per sport for gap analysis
_GAP_STAT_KEYS = {
    "nba": {"pts": "Points", "reb": "Rebounds", "ast": "Assists"},
    "nhl": {"pts": "Points", "ast": "Assists", "goals": "Goals"},
}

_GRADE_THRESHOLDS = [
    (8.5, "A"), (7.5, "A-"), (7.0, "B+"), (6.5, "B"), (6.0, "B-"),
    (5.5, "C+"), (5.0, "C"), (0.0, "D"),
]


def _grade_gap_prop(combined_score):
    """Map combined score (1-10) to letter grade."""
    for threshold, grade in _GRADE_THRESHOLDS:
        if combined_score >= threshold:
            return grade
    return "D"


async def _detect_injury_gaps(sport):
    """Core gap detector: find OUT/LIMITED players on tonight's slate,
    cross-reference against game logs to quantify production lost.

    Returns list of gaps: {player_out, team, status, days_out, ppg_lost, rpg_lost, apg_lost, mpg_lost}
    """
    sport_lower = sport.lower()
    if sport_lower not in GAP_PROPS_SUPPORTED:
        return []

    # Fetch raw ESPN injury JSON (not the formatted text)
    espn_sport = _ESPN_SPORT_PATHS.get(sport_lower)
    if not espn_sport:
        return []

    injury_cache_key = f"espn_injuries_raw:{sport_lower}"
    raw_injuries = _get_cached(injury_cache_key, ttl=1800)
    if not raw_injuries:
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_sport}/injuries"
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code != 200:
                    logger.warning(f"[GAP] ESPN injuries fetch failed: {resp.status_code}")
                    return []
                raw_injuries = resp.json()
                _set_cache(injury_cache_key, raw_injuries)
        except Exception as e:
            logger.warning(f"[GAP] ESPN injuries fetch error: {e}")
            return []

    # Parse injuries — collect OUT/Limited players with team context
    out_players = []
    now = datetime.now(PST)
    for team_data in raw_injuries.get("injuries", raw_injuries.get("items", [])):
        team_name = team_data.get("displayName", team_data.get("team", {}).get("displayName", "?"))
        team_abbr = team_data.get("abbreviation", team_data.get("team", {}).get("abbreviation", "?"))

        for injury in team_data.get("injuries", []):
            status = injury.get("status", "").strip()
            if status.lower() not in ("out", "out for season"):
                continue  # Only OUT players — GTD too uncertain

            athlete = injury.get("athlete", {})
            pname = athlete.get("displayName", "?") if isinstance(athlete, dict) else "?"
            if pname == "?":
                continue

            # Freshness: estimate days out from injury date
            days_out = 0
            details = injury.get("details", {})
            if isinstance(details, dict):
                date_str = details.get("fantasyStatus", {}).get("date", "") if isinstance(details.get("fantasyStatus"), dict) else ""
                if not date_str:
                    date_str = details.get("date", "")
            else:
                date_str = ""

            if date_str:
                try:
                    injury_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    days_out = (now - injury_date.replace(tzinfo=None if injury_date.tzinfo else None)).days
                    if injury_date.tzinfo:
                        days_out = (now - injury_date.astimezone(PST).replace(tzinfo=None)).days
                except Exception:
                    days_out = 0

            # 3-day freshness filter: skip if market has had 3+ days to adjust
            if days_out >= 3:
                continue

            out_players.append({
                "player_out": pname,
                "team": team_abbr,
                "team_name": team_name,
                "status": status,
                "days_out": days_out,
            })

    if not out_players:
        return []

    # Get tonight's slate teams to filter injuries to only tonight's games
    slate_teams = set()
    try:
        sport_keys = SPORT_KEYS.get(sport_lower, [])
        for sport_key in sport_keys:
            events_url = f"{ODDS_API_BASE}/{sport_key}/events/"
            params = {"apiKey": ODDS_API_KEY}
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(events_url, params=params)
                if resp.status_code == 200:
                    for event in resp.json():
                        home = event.get("home_team", "")
                        away = event.get("away_team", "")
                        if home:
                            slate_teams.add(home.lower())
                        if away:
                            slate_teams.add(away.lower())
    except Exception as e:
        logger.debug(f"[GAP] Slate fetch for team filter: {e}")

    # Filter to tonight's slate only (if we have slate data)
    if slate_teams:
        slate_filtered = []
        for gap in out_players:
            team_full = gap["team_name"].lower()
            if team_full in slate_teams or any(t in team_full or team_full in t for t in slate_teams):
                slate_filtered.append(gap)
        out_players = slate_filtered

    if not out_players:
        return []

    # Cross-reference against game logs to get production stats for OUT players
    all_team_abbrevs = list(set(g["team"] for g in out_players))
    game_logs = await _fetch_player_game_logs(sport, all_team_abbrevs)

    gaps = []
    for gap in out_players:
        pname = gap["player_out"]
        # Find this player's stats in game logs (case-insensitive match)
        player_stats = None
        for log_name, log_data in game_logs.items():
            if log_name.lower() == pname.lower() and log_data.get("team", "").lower() == gap["team"].lower():
                player_stats = log_data
                break
        # Fuzzy match: last name
        if not player_stats:
            last_name = pname.split()[-1].lower() if pname else ""
            for log_name, log_data in game_logs.items():
                if last_name and last_name in log_name.lower() and log_data.get("team", "").lower() == gap["team"].lower():
                    player_stats = log_data
                    break

        if not player_stats or player_stats.get("gp", 0) == 0:
            continue

        ppg = player_stats.get("season_pts", 0)
        # 10+ PPG gate: skip if OUT player isn't significant enough
        if ppg < 10:
            continue

        gap["ppg_lost"] = round(ppg, 1)
        gap["rpg_lost"] = round(player_stats.get("season_reb", 0), 1)
        gap["apg_lost"] = round(player_stats.get("season_ast", 0), 1)
        gap["mpg_lost"] = round(player_stats.get("l10_min", 0), 1)
        gaps.append(gap)

    logger.info(f"[GAP] Detected {len(gaps)} injury gaps for {sport.upper()}")
    return gaps


def _find_absorbers(sport, team_abbr, out_player_stats, game_logs):
    """Find role players likely to absorb production from the OUT player.

    Filters: season_avg < 20 PPG, l10_min < 32, games_played >= 20.
    Scores by L5 delta from season avg + minutes trend.
    Returns top 3 candidates sorted by absorption signal.
    """
    candidates = []
    sport_lower = sport.lower()

    for player_name, stats in game_logs.items():
        if stats.get("team", "").lower() != team_abbr.lower():
            continue
        if stats.get("gp", 0) < 20:
            continue  # sample too small
        if stats.get("l10_min", 0) >= 32:
            continue  # already a full-time starter
        if stats.get("season_pts", 0) >= 20:
            continue  # already a star — not a role player

        l5_pts = stats.get("l5_pts", stats.get("l10_pts", 0))
        l5_reb = stats.get("l5_reb", stats.get("l10_reb", 0))
        l5_ast = stats.get("l5_ast", stats.get("l10_ast", 0))
        l5_min = stats.get("l5_min", stats.get("l10_min", 0))
        season_pts = stats.get("season_pts", 0)
        l10_min = stats.get("l10_min", 0)

        # L5 delta: how much have they spiked recently?
        pts_delta = l5_pts - season_pts
        min_delta = l5_min - l10_min if l10_min > 0 else 0
        min_trend = stats.get("minutes_trend", "stable")

        absorption_signal = (
            pts_delta * 2.0 +              # Recent scoring spike
            min_delta * 0.5 +              # Minutes increasing
            (3 if min_trend == "UP" else 0)  # Trend bonus
        )

        candidates.append({
            "player": player_name,
            "team": team_abbr,
            "score": round(absorption_signal, 2),
            "l5_pts": round(l5_pts, 1),
            "l5_reb": round(l5_reb, 1),
            "l5_ast": round(l5_ast, 1),
            "l5_min": round(l5_min, 1),
            "l10_pts": round(stats.get("l10_pts", 0), 1),
            "l10_reb": round(stats.get("l10_reb", 0), 1),
            "l10_ast": round(stats.get("l10_ast", 0), 1),
            "l10_min": round(l10_min, 1),
            "season_pts": round(season_pts, 1),
            "season_reb": round(stats.get("season_reb", 0), 1),
            "season_ast": round(stats.get("season_ast", 0), 1),
            "gp": stats.get("gp", 0),
            "minutes_trend": min_trend,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:3]


def _match_absorbers_to_props(absorbers, prop_data, sport):
    """Cross-reference absorbers against live Odds API props.
    Filters to Over side only. Returns matched props with book/odds/line info.
    """
    if not prop_data or not absorbers:
        return []

    props_list = prop_data.get("props", [])
    if not props_list:
        return []

    # Build absorber name lookup (case-insensitive)
    absorber_map = {}
    for ab in absorbers:
        name_lower = ab["player"].lower().strip()
        absorber_map[name_lower] = ab
        # Also index by last name for fuzzy match
        parts = name_lower.split()
        if len(parts) >= 2:
            absorber_map[parts[-1]] = ab

    # Stat types we care about per sport
    stat_labels = _GAP_STAT_KEYS.get(sport.lower(), {})
    target_stats = set(stat_labels.values())

    matched = []
    for prop in props_list:
        if prop.get("side", "").lower() != "over":
            continue

        stat = prop.get("stat", "")
        if stat not in target_stats:
            continue

        player_name = prop.get("player", "").lower().strip()
        absorber = absorber_map.get(player_name)
        # Fuzzy: try last name
        if not absorber:
            parts = player_name.split()
            if len(parts) >= 2:
                absorber = absorber_map.get(parts[-1])

        if not absorber:
            continue

        matched.append({
            "absorber": absorber,
            "prop": prop,
        })

    return matched


def _score_gap_prop(gap_info, absorption_info, prop_info):
    """Score a gap prop using the 3-component weighted algorithm.

    - Gap Size Score (40%): based on ppg_lost magnitude
    - Absorption Score (35%): based on L5 delta from season avg
    - Line Lag Score (25%): based on L5 avg vs prop line gap
    Returns: {gap_size, absorption, line_lag, combined, grade, edge_pct, edge_summary}
    """
    # ── Component 1: Gap Size Score (40%) ──
    ppg_lost = gap_info.get("ppg_lost", 0)
    rpg_lost = gap_info.get("rpg_lost", 0)
    apg_lost = gap_info.get("apg_lost", 0)
    gap_magnitude = ppg_lost + (rpg_lost * 0.5) + (apg_lost * 0.3)

    if gap_magnitude >= 25:
        gap_score = 10
    elif gap_magnitude >= 20:
        gap_score = 9
    elif gap_magnitude >= 15:
        gap_score = 8
    elif gap_magnitude >= 12:
        gap_score = 7
    elif gap_magnitude >= 9:
        gap_score = 6
    elif gap_magnitude >= 6:
        gap_score = 5
    else:
        gap_score = 4

    # ── Component 2: Absorption Score (35%) ──
    prop = prop_info.get("prop", {})
    stat_label = prop.get("stat", "Points")
    stat_key_map = {"Points": "pts", "Rebounds": "reb", "Assists": "ast", "Goals": "goals"}
    stat_key = stat_key_map.get(stat_label, "pts")

    absorber = absorption_info
    l5_avg = absorber.get(f"l5_{stat_key}", absorber.get("l5_pts", 0))
    season_avg = absorber.get(f"season_{stat_key}", absorber.get("season_pts", 0))

    if season_avg > 0:
        l5_delta_pct = ((l5_avg - season_avg) / season_avg) * 100
    else:
        l5_delta_pct = 0

    if l5_delta_pct >= 30:
        abs_score = 10
    elif l5_delta_pct >= 20:
        abs_score = 9
    elif l5_delta_pct >= 15:
        abs_score = 8
    elif l5_delta_pct >= 10:
        abs_score = 7
    elif l5_delta_pct >= 5:
        abs_score = 6
    elif l5_delta_pct >= 0:
        abs_score = 5
    else:
        abs_score = 3

    # Bonus modifiers
    if absorber.get("minutes_trend") == "UP":
        abs_score = min(10, abs_score + 1)
    if absorber.get("gp", 0) >= 50:
        abs_score = min(10, abs_score + 1)
    if absorber.get("gp", 0) < 20:
        abs_score = max(1, abs_score - 1)

    # ── Component 3: Line Lag Score (25%) ──
    prop_line = float(prop.get("line", 0))
    if prop_line > 0:
        line_gap = l5_avg - prop_line
        line_gap_pct = (line_gap / prop_line) * 100
    else:
        line_gap = 0
        line_gap_pct = 0

    if line_gap_pct >= 25:
        lag_score = 10
    elif line_gap_pct >= 20:
        lag_score = 9
    elif line_gap_pct >= 15:
        lag_score = 8
    elif line_gap_pct >= 10:
        lag_score = 7
    elif line_gap_pct >= 5:
        lag_score = 6
    elif line_gap_pct >= 0:
        lag_score = 5
    else:
        lag_score = 2

    # ── Combined ──
    combined = (gap_score * 0.40) + (abs_score * 0.35) + (lag_score * 0.25)
    combined = round(combined, 1)
    grade = _grade_gap_prop(combined)
    edge_pct = round(line_gap_pct, 1) if line_gap_pct > 0 else 0

    # Build edge summary
    player_out = gap_info.get("player_out", "?")
    days = gap_info.get("days_out", 0)
    player_name = absorber.get("player", "?")
    trend = absorber.get("minutes_trend", "stable")
    summary = (
        f"{player_out} OUT {days}d. {player_name} L5 avg {l5_avg} {stat_label.upper()} "
        f"vs {prop_line} line. {'+' if line_gap >= 0 else ''}{round(line_gap, 1)} pts over book. "
        f"Minutes trending {trend}."
    )

    return {
        "gap_size": round(gap_score, 1),
        "absorption": round(abs_score, 1),
        "line_lag": round(lag_score, 1),
        "combined": combined,
        "grade": grade,
        "edge_pct": edge_pct,
        "edge_summary": summary,
        "l5_avg": round(l5_avg, 1),
        "season_avg": round(season_avg, 1),
        "l5_delta_pct": round(l5_delta_pct, 1),
    }


# ── Mathurin Screener — L5 floor vs book line ─────────────────────────────────

_SCREENER_STAT_MAP = {
    "Points": "PTS", "Rebounds": "REB", "Assists": "AST",
    "Threes": "3PM", "Three Pointers": "3PM",
    "Goals": "G", "Shots": "SOG", "Strikeouts": "K",
    "Total Bases": "TB", "Hits": "H",
    "Pass Yds": "YDS", "Rush Yds": "YDS", "Reception Yds": "YDS",
    "Pass Tds": "TD",
}


@app.get("/api/screener/{sport}")
async def run_screener(sport: str):
    """Mathurin Screener: for every prop on tonight's slate, pull L5 game logs,
    compute floor (min of L5), compare to book's line.
    Floor > line = MISPRICED = guaranteed edge."""
    sport_lower = sport.lower()

    cache_key = f"screener:{sport_lower}"
    cached = _get_cached(cache_key, ttl=300)
    if cached:
        cached["cached"] = True
        return JSONResponse(cached)

    # Step 1: Get props
    if sport_lower not in PROP_MARKETS:
        return JSONResponse({"sport": sport.upper(), "screener": [], "count": 0,
                             "message": f"No prop markets for {sport}"})

    props_result = await get_player_props(sport)
    if hasattr(props_result, 'body'):
        props_data = json.loads(props_result.body)
    else:
        props_data = _get_cached(f"props:{sport_lower}", ttl=PROPS_CACHE_TTL) or {"props": []}

    all_props = props_data.get("props", [])
    if not all_props:
        return JSONResponse({"sport": sport.upper(), "screener": [], "count": 0,
                             "message": "No props on slate"})

    # Step 2: Get teams playing
    odds_cache_key = f"{sport_lower}:h2h,spreads,totals"
    odds_data = _get_cached(odds_cache_key)
    if not odds_data:
        try:
            odds_resp = await get_odds(sport)
            if hasattr(odds_resp, 'body'):
                odds_data = json.loads(odds_resp.body)
            else:
                odds_data = {"games": []}
        except Exception:
            odds_data = {"games": []}

    team_names = set()
    for g in odds_data.get("games", []):
        team_names.add(g.get("away", ""))
        team_names.add(g.get("home", ""))

    # Step 3: Get game logs with raw L5 arrays
    game_logs = await _fetch_player_game_logs(sport, list(team_names)) if team_names else {}

    # Step 4: Screen each OVER prop
    screener_results = []
    for prop in all_props:
        if prop.get("side", "").lower() != "over":
            continue
        player = prop.get("player", "")
        stat = prop.get("stat", "")
        line = prop.get("line", 0)
        if not line:
            continue

        player_data = game_logs.get(player)
        if not player_data or player_data.get("note"):
            continue

        espn_stat = _SCREENER_STAT_MAP.get(stat)
        if not espn_stat:
            continue

        # Get raw L5 values from game logs
        raw_stats = player_data.get("raw_stats", {})
        raw = raw_stats.get(espn_stat) if raw_stats else None

        # Fallback: build from l5_ fields if raw_stats not available
        if not raw:
            stat_key = espn_stat.lower()
            l5_val = player_data.get(f"l5_{stat_key}", 0)
            l10_val = player_data.get(f"l10_{stat_key}", 0)
            if l5_val > 0:
                # Can't do floor test without raw array, skip
                continue
            continue

        l5_raw = raw.get("l5_raw", [])
        if not l5_raw:
            continue

        l5_floor = min(l5_raw)
        l5_ceiling = max(l5_raw)
        l5_avg = raw.get("l5_avg", 0)
        l10_avg = raw.get("l10_avg", 0)

        gap = round(l5_floor - line, 1)
        mispriced = l5_floor > line
        hit_count = sum(1 for v in l5_raw if v > line)
        hit_rate = round(hit_count / len(l5_raw) * 100, 0)

        screener_results.append({
            "player": player,
            "stat": stat,
            "team": player_data.get("team", "?"),
            "matchup": prop.get("matchup", ""),
            "commence": prop.get("commence", ""),
            "line": line,
            "l5_raw": l5_raw,
            "l5_floor": l5_floor,
            "l5_ceiling": l5_ceiling,
            "l5_avg": l5_avg,
            "l10_avg": l10_avg,
            "gap": gap,
            "mispriced": mispriced,
            "hit_rate": hit_rate,
            "best_odds": prop.get("best_odds"),
            "best_book": prop.get("best_book"),
            "best_prob": prop.get("best_prob"),
            "consensus_prob": prop.get("consensus_prob"),
            "edge": prop.get("edge", 0),
            "book_count": prop.get("book_count", 0),
            "books": prop.get("books", []),
            "verdict": "MISPRICED" if mispriced else ("CLOSE" if gap >= -1 else "PASS"),
        })

    screener_results.sort(key=lambda x: -x["gap"])

    result = {
        "sport": sport.upper(),
        "screener": screener_results,
        "mispriced_count": sum(1 for s in screener_results if s["mispriced"]),
        "total_screened": len(screener_results),
        "games_on_slate": len(team_names),
        "players_with_logs": len(game_logs),
        "cached": False,
        "generated_at": _now_ts(),
    }

    _set_cache(cache_key, result)
    return JSONResponse(result)


# ── Book Discrepancy Engine ───────────────────────────────────────────────────

@app.get("/api/discrepancies/{sport}")
async def find_discrepancies(sport: str):
    """Compare same prop across all books. Flag where they disagree."""
    sport_lower = sport.lower()

    cache_key = f"discrepancies:{sport_lower}"
    cached = _get_cached(cache_key, ttl=300)
    if cached:
        cached["cached"] = True
        return JSONResponse(cached)

    if sport_lower not in PROP_MARKETS:
        return JSONResponse({"sport": sport.upper(), "discrepancies": [], "count": 0})

    props_result = await get_player_props(sport)
    if hasattr(props_result, 'body'):
        props_data = json.loads(props_result.body)
    else:
        props_data = _get_cached(f"props:{sport_lower}", ttl=PROPS_CACHE_TTL) or {"props": []}

    all_props = props_data.get("props", [])
    discrepancies = []

    for prop in all_props:
        books = prop.get("books", [])
        if len(books) < 2:
            continue
        sorted_books = sorted(books, key=lambda b: b["odds"], reverse=True)
        best = sorted_books[0]
        worst = sorted_books[-1]
        best_prob = best.get("implied_prob", 0) or 0
        worst_prob = worst.get("implied_prob", 0) or 0
        prob_gap = round(abs(worst_prob - best_prob), 1)
        odds_gap = best["odds"] - worst["odds"]

        if prob_gap >= 1.0:
            discrepancies.append({
                "player": prop.get("player", ""),
                "stat": prop.get("stat", ""),
                "side": prop.get("side", ""),
                "line": prop.get("line", 0),
                "matchup": prop.get("matchup", ""),
                "commence": prop.get("commence", ""),
                "best_book": best["book"],
                "best_odds": best["odds"],
                "best_prob": best_prob,
                "worst_book": worst["book"],
                "worst_odds": worst["odds"],
                "worst_prob": worst_prob,
                "prob_gap": prob_gap,
                "odds_gap": odds_gap,
                "book_count": len(books),
                "all_books": sorted_books,
            })

    discrepancies.sort(key=lambda x: -x["prob_gap"])
    result = {
        "sport": sport.upper(),
        "discrepancies": discrepancies,
        "count": len(discrepancies),
        "cached": False,
    }
    _set_cache(cache_key, result)
    return JSONResponse(result)


@app.get("/api/props/gaps/{sport}")
async def get_gap_props(sport: str, min_gap_score: float = 5.0, force: bool = False):
    """Prop Gap Analyzer — find mispriced role player props when starters are OUT.
    Pure math engine: injury detection → absorber ID → prop matching → scoring.
    No AI/GPT calls. Zero additional API cost beyond existing caches."""
    sport_lower = sport.lower()

    if sport_lower not in GAP_PROPS_SUPPORTED:
        return JSONResponse({
            "sport": sport.upper(),
            "gap_props": [],
            "message": f"Gap analysis not supported for {sport.upper()}. Supported: {', '.join(s.upper() for s in GAP_PROPS_SUPPORTED)}",
        }, status_code=400)

    # Check cache
    cache_key = f"gap_props:{sport_lower}"
    if not force:
        cached = _get_cached(cache_key, ttl=GAP_PROPS_CACHE_TTL)
        if cached:
            cached["cached"] = True
            return JSONResponse(cached)

    # Also check file cache (same-day persistence)
    today_str = datetime.now(PST).strftime("%Y-%m-%d")
    data_dir = "/data/gap_props"
    file_path = f"{data_dir}/gap_{sport_lower}_{today_str}.json"
    if not force:
        try:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    file_data = json.load(f)
                # Check if file is fresh enough (within TTL)
                gen_ts = file_data.get("_generated_ts", 0)
                if time.time() - gen_ts < GAP_PROPS_CACHE_TTL:
                    file_data["cached"] = True
                    _set_cache(cache_key, file_data)
                    return JSONResponse(file_data)
        except Exception:
            pass

    # ── PIPELINE: Detect → Absorb → Match → Score → Filter ──

    # Step 1: Detect injury gaps
    gaps = await _detect_injury_gaps(sport)
    if not gaps:
        result = {
            "sport": sport.upper(),
            "gap_props": [],
            "injuries_scanned": 0,
            "absorbers_found": 0,
            "props_matched": 0,
            "message": "No impactful injuries detected on tonight's slate",
            "generated_at": _now_ts(),
            "cached": False,
            "ttl_seconds": GAP_PROPS_CACHE_TTL,
        }
        _set_cache(cache_key, result)
        return JSONResponse(result)

    # Step 2: Get game logs for all affected teams
    affected_teams = list(set(g["team"] for g in gaps))
    game_logs = await _fetch_player_game_logs(sport, affected_teams)

    # Step 3: Get prop data (reuse existing cache)
    prop_cache_key = f"props:{sport_lower}"
    prop_data = _get_cached(prop_cache_key, ttl=PROPS_CACHE_TTL)
    if not prop_data:
        try:
            prop_result = await get_player_props(sport)
            # get_player_props returns JSONResponse, extract body
            if hasattr(prop_result, 'body'):
                prop_data = json.loads(prop_result.body)
            else:
                prop_data = _get_cached(prop_cache_key, ttl=PROPS_CACHE_TTL)
        except Exception as e:
            logger.warning(f"[GAP] Props fetch failed: {e}")
            prop_data = None

    if not prop_data or not prop_data.get("props"):
        result = {
            "sport": sport.upper(),
            "gap_props": [],
            "injuries_scanned": len(gaps),
            "absorbers_found": 0,
            "props_matched": 0,
            "message": "No prop lines available from Odds API",
            "generated_at": _now_ts(),
            "cached": False,
            "ttl_seconds": GAP_PROPS_CACHE_TTL,
        }
        _set_cache(cache_key, result)
        return JSONResponse(result)

    # Step 4: For each gap, find absorbers and match to props
    all_gap_props = []
    total_absorbers = 0

    for gap in gaps:
        out_stats = {
            "ppg": gap.get("ppg_lost", 0),
            "rpg": gap.get("rpg_lost", 0),
            "apg": gap.get("apg_lost", 0),
            "mpg": gap.get("mpg_lost", 0),
        }

        absorbers = _find_absorbers(sport, gap["team"], out_stats, game_logs)
        total_absorbers += len(absorbers)

        if not absorbers:
            continue

        # Match absorbers to props
        matched = _match_absorbers_to_props(absorbers, prop_data, sport)

        for match in matched:
            absorber = match["absorber"]
            prop = match["prop"]

            # Score it
            scoring = _score_gap_prop(gap, absorber, match)

            # Filter by minimum score
            if scoring["combined"] < min_gap_score:
                continue
            # Hard gate: minimum B- (6.0) regardless of min_gap_score param
            if scoring["combined"] < 6.0:
                continue

            gap_prop = {
                "player": absorber["player"],
                "team": absorber["team"],
                "matchup": prop.get("matchup", ""),
                "commence": prop.get("commence", ""),
                "prop_type": prop.get("stat", "Points"),
                "prop_line": prop.get("line", 0),
                "prop_side": "Over",
                "best_odds": prop.get("best_odds", 0),
                "best_book": prop.get("best_book", ""),
                "book_count": prop.get("book_count", 0),
                "gap_reason": {
                    "player_out": gap["player_out"],
                    "out_status": gap["status"],
                    "days_out": gap["days_out"],
                    "ppg_lost": gap.get("ppg_lost", 0),
                    "rpg_lost": gap.get("rpg_lost", 0),
                    "apg_lost": gap.get("apg_lost", 0),
                    "mpg_lost": gap.get("mpg_lost", 0),
                },
                "absorption": {
                    "l5_avg": scoring["l5_avg"],
                    "l10_avg": absorber.get("l10_pts", 0),
                    "season_avg": scoring["season_avg"],
                    "l5_min": absorber.get("l5_min", 0),
                    "l10_min": absorber.get("l10_min", 0),
                    "minutes_trend": absorber.get("minutes_trend", "stable"),
                    "games_played": absorber.get("gp", 0),
                    "recent_delta": f"+{round(scoring['l5_avg'] - scoring['season_avg'], 1)} {prop.get('stat', 'PTS').upper()} vs season avg in L5",
                },
                "scores": {
                    "gap_size": scoring["gap_size"],
                    "absorption": scoring["absorption"],
                    "line_lag": scoring["line_lag"],
                    "combined": scoring["combined"],
                },
                "grade": scoring["grade"],
                "edge_pct": scoring["edge_pct"],
                "edge_summary": scoring["edge_summary"],
            }
            all_gap_props.append(gap_prop)

    # Sort by combined score descending
    all_gap_props.sort(key=lambda x: x["scores"]["combined"], reverse=True)

    result = {
        "sport": sport.upper(),
        "gap_props": all_gap_props,
        "injuries_scanned": len(gaps),
        "absorbers_found": total_absorbers,
        "props_matched": len(all_gap_props),
        "generated_at": _now_ts(),
        "cached": False,
        "ttl_seconds": GAP_PROPS_CACHE_TTL,
    }

    # Cache in-memory
    result["_generated_ts"] = time.time()
    _set_cache(cache_key, result)

    # Persist to file
    try:
        os.makedirs(data_dir, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(result, f, indent=2)
        # Prune old files (older than 3 days)
        for fname in os.listdir(data_dir):
            fpath = os.path.join(data_dir, fname)
            if os.path.isfile(fpath):
                try:
                    file_age = time.time() - os.path.getmtime(fpath)
                    if file_age > 3 * 86400:
                        os.remove(fpath)
                except Exception:
                    pass
    except Exception as e:
        logger.debug(f"[GAP] File persistence failed: {e}")

    # Remove internal timestamp before response
    result.pop("_generated_ts", None)
    logger.info(f"[GAP] {sport.upper()}: {len(all_gap_props)} gap props found (scanned {len(gaps)} injuries, {total_absorbers} absorbers)")
    return JSONResponse(result)


# ── End Prop Gap Analyzer ────────────────────────────────────────────────────


# ── Card EV Analysis Engine ──────────────────────────────────────────────────

@app.post("/api/ev/card")
async def ev_card(request: Request):
    """Run full EV analysis on a single game card — screener + discrepancies + gaps,
    filtered to the two teams on the card, then sent through AI for structured output."""
    body = await request.json()
    sport = body.get("sport", "").lower()
    away_team = body.get("away_team", "")
    home_team = body.get("home_team", "")

    if not sport or not away_team or not home_team:
        return JSONResponse({"error": "sport, away_team, and home_team are required"}, status_code=400)

    team_names = {away_team.lower(), home_team.lower()}
    card_data = body  # pass full payload to AI prompt

    # ── Run all three engines in parallel ──
    async def safe_call(coro, label):
        try:
            resp = await coro
            if hasattr(resp, "body"):
                return json.loads(resp.body)
            return resp
        except Exception as e:
            logger.warning(f"[EV CARD] {label} failed: {e}")
            return {}

    screener_data, disc_data, gap_data = await asyncio.gather(
        safe_call(run_screener(sport), "screener"),
        safe_call(find_discrepancies(sport), "discrepancies"),
        safe_call(get_gap_props(sport), "gaps"),
    )

    # ── Filter results to this game's teams ──
    def team_match(name):
        return name.lower() in team_names or any(t in name.lower() for t in team_names)

    filtered_screener = [
        s for s in screener_data.get("screener", [])
        if team_match(s.get("team", "")) or team_match(s.get("matchup", ""))
    ]
    filtered_disc = [
        d for d in disc_data.get("discrepancies", [])
        if team_match(d.get("matchup", ""))
    ]
    filtered_gaps = [
        g for g in gap_data.get("gap_props", [])
        if team_match(g.get("team", "")) or team_match(g.get("matchup", ""))
    ]

    # ── Build AI prompt ──
    ev_prompt = f"""You are the EV Agent. Analyze this game for real edge using the math below.
DO NOT grade the teams. DO NOT give letter grades.
Find where the book is WRONG. Show the number that proves it.
Run the Mathurin Test on any props flagged (L5 floor vs book line).
Output ONLY valid JSON with this exact structure:
{{
  "spread_call": {{ "pick": "TEAM +/-X", "reasoning": "1-2 sentences with numbers" }},
  "top_props": [
    {{ "player": "Name", "prop": "O/U X.5 STAT", "l5_floor": 0, "line": 0, "gap": 0, "verdict": "MISPRICED/CLOSE/PASS" }}
  ],
  "book_discrepancy": "1-2 sentences or null if none",
  "trap_check": "1-2 sentences",
  "lock_suggestions": [
    {{ "selection": "TEAM +/-X or PLAYER O/U X.5", "matchup": "{away_team} @ {home_team}", "sport": "{sport.upper()}", "odds": "-110", "units": "1", "confidence": "Strong/Moderate/Lean" }}
  ]
}}
Be direct. Numbers only. No fluff. No markdown. Just the JSON object.

CARD DATA: {json.dumps(card_data)}
SCREENER HITS ({len(filtered_screener)}): {json.dumps(filtered_screener[:10])}
BOOK DISCREPANCIES ({len(filtered_disc)}): {json.dumps(filtered_disc[:10])}
GAP PROPS ({len(filtered_gaps)}): {json.dumps(filtered_gaps[:5])}"""

    # ── Call Azure OpenAI ──
    if not AZURE_ENDPOINT or not AZURE_KEY:
        # No AI available — return raw engine data
        return JSONResponse({
            "raw": True,
            "screener": filtered_screener[:5],
            "discrepancies": filtered_disc[:5],
            "gap_props": filtered_gaps[:5],
        })

    try:
        url = f"{AZURE_BASE}/openai/deployments/{AZURE_MODEL}/chat/completions?api-version=2024-08-01-preview"
        headers = {"api-key": AZURE_KEY, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json={
                "messages": [
                    {"role": "system", "content": "You are a sharp sports betting EV analyst. Return ONLY valid JSON. No markdown fences."},
                    {"role": "user", "content": ev_prompt},
                ],
                "max_tokens": 800,
                "temperature": 0.3,
            }, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            raw_reply = data["choices"][0]["message"]["content"].strip()

            # Parse JSON from reply (handle markdown fences if model adds them)
            clean = raw_reply
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()

            try:
                ev_result = json.loads(clean)
            except json.JSONDecodeError:
                ev_result = {"raw_text": raw_reply}

            ev_result["_engine_data"] = {
                "screener_count": len(filtered_screener),
                "disc_count": len(filtered_disc),
                "gap_count": len(filtered_gaps),
            }
            return JSONResponse(ev_result)

    except Exception as e:
        logger.error(f"[EV CARD] AI call failed: {e}")
        return JSONResponse({
            "error": "EV analysis timed out. Try again.",
            "raw": True,
            "screener": filtered_screener[:5],
            "discrepancies": filtered_disc[:5],
            "gap_props": filtered_gaps[:5],
        })


# ── End Card EV Analysis ────────────────────────────────────────────────────


# ── Simple Slate API ─────────────────────────────────────────────────────────

@app.get("/api/simple/{sport}")
async def get_simple_slate(sport: str):
    """Combined endpoint for /simple page — returns everything in ONE call.
    Merges analysis + edge data + gap props into a single response.
    Games with grade B+ or better AND composite > 7.0 go into games_with_edges.
    Everything else listed as no_edge_games (matchup names only)."""
    sport_lower = sport.lower()
    if sport_lower not in ALL_SPORTS:
        return JSONResponse({"error": f"Unknown sport: {sport}"}, status_code=400)

    # 1. Get cached analysis (never trigger GPT)
    analysis_games = []
    try:
        analysis_resp = await get_analysis(sport_lower, cached_only=True)
        if hasattr(analysis_resp, 'body'):
            analysis_data = json.loads(analysis_resp.body)
            analysis_games = analysis_data.get("games", [])
    except Exception as e:
        print(f"[SIMPLE] Analysis fetch error for {sport_lower}: {e}")

    # 2. Get edge data
    edge_games = []
    try:
        edge_resp = await get_edge(sport_lower)
        if hasattr(edge_resp, 'body'):
            edge_data = json.loads(edge_resp.body)
            edge_games = edge_data.get("games", [])
    except Exception as e:
        print(f"[SIMPLE] Edge fetch error for {sport_lower}: {e}")

    # 3. Get gap props (only for supported sports)
    gap_props = []
    if sport_lower in GAP_PROPS_SUPPORTED:
        try:
            gap_resp = await get_gap_props(sport_lower)
            if hasattr(gap_resp, 'body'):
                gap_data = json.loads(gap_resp.body)
                gap_props = gap_data.get("gap_props", [])
        except Exception as e:
            print(f"[SIMPLE] Gap props fetch error for {sport_lower}: {e}")

    # Build edge lookup by matchup for merging
    edge_by_matchup = {}
    for eg in edge_games:
        key = eg.get("matchup", eg.get("game", "")).lower()
        edge_by_matchup[key] = eg

    # Grade thresholds: B+ or better
    PASSING_GRADES = {"A+", "A", "A-", "B+"}

    games_with_edges = []
    no_edge_games = []

    for g in analysis_games:
        grade = g.get("grade", "")
        composite = g.get("composite_score", 0)
        matchup = g.get("matchup", g.get("game", ""))

        if grade in PASSING_GRADES and composite > 7.0:
            # Merge edge data if available
            edge_info = edge_by_matchup.get(matchup.lower(), {})
            games_with_edges.append({
                "matchup": matchup,
                "grade": grade,
                "composite_score": composite,
                "pick": g.get("pick", g.get("recommendation", "")),
                "line": g.get("line", ""),
                "confidence": g.get("confidence", ""),
                "edge_thesis": g.get("edge_thesis", g.get("why_market_wrong", "")),
                "upset_score": edge_info.get("upset_score", None),
                "public_fade": edge_info.get("public_fade", False),
                "sharp_signal": edge_info.get("sharp_signal", False),
                "commence_time": g.get("commence_time", ""),
            })
        else:
            no_edge_games.append({"matchup": matchup, "grade": grade})

    # Sort edges by composite score descending
    games_with_edges.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

    return JSONResponse({
        "sport": sport.upper(),
        "games_with_edges": games_with_edges,
        "no_edge_games": no_edge_games,
        "gap_props": gap_props[:10] if gap_props else [],
        "total_games": len(analysis_games),
        "edge_count": len(games_with_edges),
        "updated_at": _now_ts(),
    })


@app.get("/simple")
async def simple_page():
    return FileResponse("static/simple.html", headers=NO_CACHE_HEADERS)


@app.get("/")
async def root():
    return FileResponse("static/index.html", headers=NO_CACHE_HEADERS)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# Edge Crew V2 — Full System Flow Chart

**Last updated:** 2026-03-23 by Pietra
**Server:** `C:\Claude\Repos\edge-crew\server.py` (14,465 lines)
**Live:** `edge-crew-v2.onrender.com`

---

## 1. THE FULL PROCESS — User Clicks "Analyze" to Card on Screen

### Overview

```
USER CLICKS ANALYZE
       |
   [1] ODDS FETCH (SGO > Odds API > SharpAPI)
       |
   [2] DATA ASSEMBLY (injuries, rosters, form, game logs, props, gap props)
       |
   [3] CROWDSOURCE FIRST — 4 lightweight models grade ALL games
       |
   [4] B+ FILTER — only majority-B+ games pass to deep thinker
       |
   [5] DEEP THINKER (DeepSeek-R1) — full matrix analysis on B+ games only
       |
   [6] FORMATTER (DeepSeek-V3.2) — raw analysis -> structured JSON
       |
   [7] POST-PROCESSING — grade recalc, chains, injury clamps, validator
       |
   [8] CHALLENGER MODEL — independent grade from Claude Sonnet 4.6
       |
   [9] PROP EDGE INJECTION — discrepancies + gap props attached to cards
       |
   [10] CACHE + SERVE — memory cache + disk file + Postgres dual-write
       |
   CARD RENDERS ON SCREEN
```

### Step-by-Step Detail

#### STEP 1: Odds Fetch (`get_odds()`)
| Priority | Source | Config Key | Sports |
|----------|--------|-----------|--------|
| 1 (primary) | SportsGameOdds (SGO) | `SPORTSGAMEODDS_KEY` | All via `SGO_LEAGUE_MAP` |
| 2 (fallback) | The Odds API | `ODDS_API_KEY` | All via `SPORT_KEYS` |
| 3 (fallback) | SharpAPI | `SHARPAPI_KEY` | NBA, NHL, MLB, Soccer, MMA, Boxing |

- Filters to future games within 24h window
- Deduplicates by matchup (away + home)
- Tracks opening lines for shift detection
- Detects arbitrage across bookmakers
- Stores ALL bookmaker data for multi-book comparison
- Saves to daily slate file on disk
- Book preference: HardRockBet > DraftKings > FanDuel > BetMGM > Bovada

#### STEP 2: Data Assembly (parallel fetches)

| Data Layer | Source(s) | Cache TTL | Function |
|-----------|-----------|-----------|----------|
| Injuries (primary) | ESPN API | 30 min | `_fetch_espn_injuries()` |
| Injuries (fallback) | CBS Sports scrape | 30 min | `_get_lineup_and_injury_context()` |
| Injuries (supplement) | RotoWire scrape | 30 min | `_fetch_rotowire_page()` |
| Injuries (enrichment) | Tank01 (RapidAPI) | 30 min | `_build_tank01_injury_context()` |
| Lineups (soccer) | API-Sports | 30 min | `_fetch_api_sports_lineups()` |
| Lineups (NHL goalies) | API-Sports | 30 min | `_fetch_api_sports_lineups()` |
| Rosters | ESPN API | 1 hour | `get_rosters()` |
| Team Form (L5/L10) | Scores archive (local) | sync | `_build_team_profile()` |
| Player Game Logs | ESPN/BDL | 1 hour | `_fetch_player_game_logs()` |
| Player Props | SGO > Odds API | 5 min | `get_player_props()` |
| Gap Props (cascade) | Injury + game logs | 5 min | `get_gap_props()` |
| Discrepancies | Multi-book comparison | 5 min | `find_discrepancies()` |

- Step 2.5 overnight batch (2 AM PST) pre-computes all data layers — no AI calls
- Warm caches from overnight batch before analysis to avoid redundant fetches

#### STEP 3: Crowdsource-First Pipeline (default mode)

4 lightweight models grade ALL games with a shortened prompt:
1. **Mistral Large 3** (AI Services)
2. **Llama 4 Maverick** (AI Services)
3. **Grok 3** (AI Services)
4. **DeepSeek V3.2** (AI Services)

Run **sequentially** with 3s delay between calls. Each returns grade + composite score + edge pick.

#### STEP 4: B+ Consensus Filter

- Count models giving B+ or higher per game
- **MAJORITY_BP** (50%+ of models = B+): send to deep thinker
- **BELOW_BP**: crowdsource grades stand alone, no deep analysis
- **SPLIT**: marked but not deep-analyzed

#### STEP 5: Deep Thinker (only on B+ games)

| Sport | Thinker Model | Timeout |
|-------|--------------|---------|
| All sports | DeepSeek-R1 | 180-300s |

- Receives full prompt: odds + injuries + rosters + form + game logs + props + matrix
- Produces free-form reasoning (no JSON pressure)
- Validated for team/injury references before formatting
- Raw output saved to disk as backup

#### STEP 6: Formatter

- **Model:** DeepSeek-V3.2 (AI Services)
- Converts raw thinker output into exact JSON schema
- Maps analyst's variable scores to correct matrix keys
- Retries once on failure; falls back to single-model if both fail
- Fallback chain: Grok 3 -> Kimi K2.5 -> DeepSeek V3.2

#### STEP 7: Post-Processing Pipeline

| Stage | What It Does |
|-------|-------------|
| **Matchup Normalization** | Resolve full team names to abbreviations matching odds data |
| **Matrix Score Normalization** | Map model's variable names to canonical matrix keys via fuzzy matching |
| **Server-Side Grade Recalc** | Recalculate composite from matrix_scores weights. Override GPT's grade with our math |
| **Injury Freshness Clamp** | Cap `star_player_status` based on injury age: SEASON=3-5, ESTABLISHED=5, RECENT=8, FRESH=10 |
| **Chain Detection** | Detect compound signal patterns (e.g., THE MISPRICING, FATIGUE FADE). Add bonus to composite (capped at +3.0) |
| **Auto-Edge Flag** | If 3+ FRESH injuries on one side, boost composite by 15% |
| **Edge Grade** | Calculate subset grade from thesis_edge + line_vs_model + sharp_vs_public |
| **Picking-Into-Injury Cap** | If picking team with FRESH injuries and no "despite" explanation, cap grade to C+ |
| **Grade Validation Gate** | Final recalculation pass — override any GPT math that's off by >0.2 |
| **Grade Validator (engine)** | `engines/grade_validator.py` — post-analysis checks and balances |
| **Player Validation (WS4)** | Cross-check all mentioned players against ESPN rosters. Flag/remove phantoms |
| **Alt Grade** | Team-profile + H2H secondary grade for comparison |

#### STEP 8: Challenger Model

| Current Challenger | Rotation |
|-------------------|----------|
| Claude Sonnet 4.6 (fixed override) | Weekly rotation: Claude > Grok 3 > Llama 4 > DeepSeek V3.2 > Grok 4 Fast |

- Runs on batch 0 only, 60s hard timeout
- Result attached as `challenger_grade` / `challenger_score` on each card
- Never blocks the main pipeline

#### STEP 9: Prop Edge Injection

- Fetches book discrepancies (`find_discrepancies()`)
- Fetches gap props (`get_gap_props()`)
- Attaches mispriced props to each game card
- `prop_edge_count` on each card for frontend rendering

#### STEP 10: Cache + Serve

| Layer | TTL | Key Pattern |
|-------|-----|-------------|
| Memory (dict) | 3 hours | `analysis:{sport}` |
| Disk (JSON file) | Today + yesterday fallback | `data/analysis_cache/analysis_{sport}_{date}_latest.json` |
| Postgres | Dual-write (fire-and-forget) | `picks` and `scores` tables |

---

## 2. EVERY ENDPOINT

### Health & System

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/health` | Health check for Render | — | No (Render only) |
| GET | `/api/version` | Build version, mode, thinker, formatter | — | YES |
| GET | `/api/errors` | Last 100 error log entries | In-memory deque | No (admin) |
| POST | `/api/errors` | Accept error reports from frontend JS | — | YES |
| GET | `/api/credits` | Check Odds API + SharpAPI remaining credits | Odds API | No (admin) |
| POST | `/api/cache/clear` | Clear in-memory cache | — | YES (admin) |
| GET | `/api/schedule` | Smart analysis schedule (tip times, fired status) | Daily slates | No (admin) |
| GET | `/api/sgo-usage` | SGO entity budget tracker | `data/sgo_usage.json` | No (admin) |

### Auth

| Method | Path | What It Does | Frontend Wired? |
|--------|------|-------------|-----------------|
| POST | `/api/auth/register` | Register new crew member (name + 4-digit PIN) | YES |
| POST | `/api/auth/login` | Login with name + PIN, get session token | YES |
| GET | `/api/auth/me` | Check current session | YES |
| POST | `/api/auth/logout` | Destroy session token | YES |
| GET | `/api/auth/profiles` | List all crew names + colors (no PINs) | YES |

### Odds & Slate

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/api/odds/{sport}` | Live odds — SGO primary, Odds API fallback | SGO/OddsAPI/SharpAPI | YES |
| GET | `/api/slate` | Full slate — all sports combined | Cached slates or live | YES |
| GET | `/api/slate/cached` | Cached-only full slate (no API calls) | Disk files | YES |
| GET | `/api/slate/cached/{sport}` | Cached sport slate, live fallback | Disk/live | YES |

### Soccer Hub

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/api/soccer/leagues` | Full league registry with live game counts | SOCCER_LEAGUES registry | YES |
| GET | `/api/soccer/high-edge` | B+ or above soccer games from analysis | Analysis cache | YES |
| GET | `/api/odds/soccer/league/{league_key}` | On-demand Tier 3 league odds | The Odds API | YES |

### Analysis & Edge

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/api/analysis/{sport}` | **MAIN ENGINE** — AI analysis with full pipeline | Azure AI (multi-model) | YES |
| GET | `/api/edge/{sport}` | Math-only edge analysis (no AI) — upset scores, injury impact | Odds + injuries | YES |
| GET | `/api/hurdle/{sport}` | 8-stage hurdle pipeline (Phase 1: Stage 0 filter only) | Odds + profiles | No (not wired yet) |
| GET | `/api/simple/{sport}` | Simplified analysis view | Analysis cache | YES |
| GET | `/api/profedge/{sport}` | Professional edge analysis | Multi-source | YES |
| POST | `/api/profedge/regrade/{game_id}` | Regrade a specific game | Azure AI | YES |

### Step 2.5 (Overnight Batch)

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/api/step25` | Check batch status (all or one sport) | Disk files | No (admin) |
| POST | `/api/step25/run` | Manually trigger batch | Multiple APIs | No (admin) |

### Player Props & Screener

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/api/props/{sport}` | Player prop lines with edge analysis | SGO/Odds API | YES |
| GET | `/api/screener/{sport}` | Prop screener — L5 floor vs book line | Game logs + props | YES |
| GET | `/api/discrepancies/{sport}` | Multi-book line discrepancies | Odds data (all books) | YES |
| GET | `/api/props/gaps/{sport}` | Gap/cascade props (injury -> usage boost) | Injuries + game logs | YES |
| GET | `/api/props/edge-board/{sport}` | Edge board for props | Props + analysis | YES |
| GET | `/api/prop-board` | Prop board — winning prop history | Disk file | YES |
| POST | `/api/prop-board/manual` | Manually add prop board entry | — | YES |
| POST | `/api/prop-board/{entry_id}/relock` | Relock a prop board entry | — | YES |
| POST | `/api/prop-board/backfill` | Backfill prop board from graded picks | Picks history | No (admin) |

### Picks (Bet Tracker)

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/api/picks` | Get all picks, optional date filter | `data/picks.json` + PG | YES |
| POST | `/api/picks` | Save new pick from bet slip | — | YES |
| PUT | `/api/picks/{pick_id}` | Update a pick | — | YES |
| DELETE | `/api/picks/{pick_id}` | Delete a pick | — | YES |
| POST | `/api/picks/place` | Mark pick as placed on book | — | YES |
| POST | `/api/picks/grade` | Grade pick W/L/P | — | YES |
| POST | `/api/picks/autograde` | Trigger autograde for ungraded picks | Odds API scores | YES |
| POST | `/api/picks/seed` | Seed picks from Supabase export | — | No (migration) |
| POST | `/api/picks/normalize` | Normalize pick types/sports | — | No (migration) |
| GET | `/api/picks/unique` | Contrarian/unique winning picks | Picks history | No (not wired) |
| DELETE | `/api/picks/crew/{member_name}` | Delete all picks for a crew member | — | No (admin) |

### Scores

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/api/scores` | Live scoreboard — today + yesterday completed | Odds API scores | YES |

### Card Detail Endpoints

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/api/card/injuries/{sport}/{matchup}` | Injury data for one matchup | ESPN + CBS | YES |
| GET | `/api/card/records/{sport}/{matchup}` | Team records for one matchup | Scores archive | YES |
| GET | `/api/card/h2h/{sport}/{matchup}` | Head-to-head history | Scores archive | YES |
| GET | `/api/card/lineup/{sport}/{matchup}` | Lineup data for one matchup | API-Sports + ESPN | YES |
| GET | `/api/alt-grade/{sport}/{matchup}` | Alt grade for one matchup | Team profiles + H2H | YES |
| GET | `/api/team-profile/{sport}/{team}` | Full team profile | Scores archive | YES |

### Lineups & Rosters

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/api/lineups/{sport}` | Structured injury/lineup data for frontend | ESPN + CBS + RotoWire | YES |
| GET | `/api/rosters/{sport}` | ESPN rosters for all teams | ESPN API | YES (internal) |
| GET | `/api/gamelogs/{sport}` | Player game logs | ESPN/BDL | No (internal) |

### Crew & Social

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/api/crew/inactive` | List inactive crew members | Picks + profiles | No (admin) |
| GET | `/api/crew/{member_id}/homepage` | Crew member homepage data | Picks + profiles | YES |
| DELETE | `/api/crew/{member_name}` | Remove crew member | Profiles | No (admin) |
| GET | `/api/crew-board/{member}` | Crew member message board | Disk file | YES |
| POST | `/api/crew-board/{member}` | Post to crew board | — | YES |
| POST | `/api/crew-board/{member}/{post_id}/reply` | Reply to board post | — | YES |
| DELETE | `/api/crew-board/{member}/{post_id}` | Delete board post | — | YES |
| PATCH | `/api/crew-board/{member}/{post_id}/pin` | Pin/unpin board post | — | YES |
| GET | `/api/dj-feedback` | DJ feedback/notes | Disk file | YES |
| POST | `/api/dj-feedback` | Submit DJ feedback | — | YES |
| POST | `/api/dj-feedback/{note_id}/reply` | Reply to DJ feedback | — | YES |

### Kalshi & Market Data

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/api/kalshi` | Market-implied probabilities from Kalshi | Kalshi public API | No (not wired) |
| GET | `/api/arb-scan` | Cross-book arbitrage scanner | Odds data | YES |

### Challenger & Crowdsource

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/api/challenger` | Current challenger model + rotation | Config | YES |

### EV / Chat / Voice

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| POST | `/api/ev/card` | Generate EV analysis card | Azure AI | YES |
| POST | `/api/chat` | Chat with Edge Crew AI | Azure AI | YES |
| POST | `/api/voice/token` | Get OpenAI Realtime API ephemeral token | Azure | YES |
| POST | `/api/voice/sdp` | WebRTC SDP exchange for voice | Azure | YES |

### User Agent System

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/api/agent/{user_id}/profile` | Get user agent profile | Disk file | YES |
| PUT | `/api/agent/{user_id}/profile` | Update agent profile | — | YES |
| POST | `/api/agent/{user_id}/profile` | Create agent profile | — | YES |
| POST | `/api/agent/{user_id}/action` | Execute agent action (voice tool) | — | YES |
| GET | `/api/agent/{user_id}/matrix/{sport}` | Get user's resolved matrix | Profile + defaults | YES |

### Misc Data

| Method | Path | What It Does | Data Source | Frontend Wired? |
|--------|------|-------------|-------------|-----------------|
| GET | `/api/upsets` | Get upset picks history | `data/upsets.json` | YES |
| POST | `/api/upsets` | Save upset pick | — | YES |
| DELETE | `/api/upsets/{pick_id}` | Delete upset pick | — | YES |
| GET | `/api/bankroll` | Get bankroll tracker | `data/bankroll.json` | YES |
| POST | `/api/bankroll` | Update bankroll | — | YES |
| GET | `/api/gotcha` | Get gotcha notes | `data/gotcha_notes.json` | YES |
| POST | `/api/gotcha` | Save gotcha note | — | YES |
| GET | `/api/pre-analysis/{sport}` | Pre-analysis data (overnight) | Disk file | YES |
| GET | `/api/debug/injuries/{sport}` | Debug injury sources | Multiple | No (admin) |
| POST | `/api/autopicker/run` | Manually trigger autopicker | Analysis cache | No (admin) |

### Page Routes

| Method | Path | What It Does |
|--------|------|-------------|
| GET | `/` | Main app (index.html) |
| GET | `/simple` | Simple view |
| GET | `/bets` | Bets page |
| GET | `/ev` | EV page |
| GET | `/picks` | Picks page |
| GET | `/dj` | DJ's page |
| GET | `/dj/ncaab-system` | DJ NCAAB system |
| GET | `/dj/nba-system` | DJ NBA system |
| GET | `/peter` | Peter's page |
| GET | `/chinny` | Chinny's page |
| GET | `/chinny/nhl-system` | Chinny NHL system |
| GET | `/pietra` | Pietra's page |
| GET | `/jimmy` | Jimmy's page |
| GET | `/crew/{member_slug}` | Generic crew page |
| GET | `/home/{user_id}` | User home page |
| GET | `/agent/{user_id}` | User agent page |
| GET | `/sw.js` | Service worker |
| GET | `/favicon.ico` | Favicon |
| GET | `/manifest.json` | PWA manifest |

---

## 3. BACKGROUND LOOPS (startup tasks)

| Loop | Interval | What It Does |
|------|----------|-------------|
| `_autograde_loop` | Every 2 hours | Auto-grade ungraded picks: local archive first, then Odds API, then AI prop fallback |
| `_daily_slate_pull` | Every 5 min (check) | Pull odds at hours 0,6,7,12,16,23. Run analysis at 7AM/Noon/4PM. Smart T-2h/T-30m before tips. Autopicker after analysis. NCAAB morning retries. |
| `_warm_analysis_cache` | Once (2 min after boot) | Load today's analysis cache from disk into memory |
| `_prefetch_loop` | Every 10 min | Pre-warm odds/props/discrepancy caches for all sports |

---

## 4. THE GAPS

### Built But NOT Wired to Frontend

| Feature | Endpoint | Status |
|---------|----------|--------|
| **Kalshi market probabilities** | `/api/kalshi` | Backend complete, not called from frontend |
| **8-Stage Hurdle Pipeline** | `/api/hurdle/{sport}` | Phase 1 only (Stage 0 filter). Stages 1-8 not implemented. Not wired to frontend. |
| **Unique/Contrarian Picks** | `/api/picks/unique` | Backend complete, not surfaced |
| **SGO Usage Monitor** | `/api/sgo-usage` | Admin-only, no frontend UI |
| **Analysis Schedule** | `/api/schedule` | Admin-only, no frontend UI |
| **Step 2.5 Status** | `/api/step25` | Admin-only, no frontend UI |
| **Debug Injuries** | `/api/debug/injuries/{sport}` | Admin-only |
| **Game Logs API** | `/api/gamelogs/{sport}` | Internal use only, no frontend surface |
| **Autopicker Manual** | `/api/autopicker/run` | Admin-only |

### Missing / Incomplete Integrations

| Gap | Detail |
|-----|--------|
| **Hurdle Pipeline Stages 1-8** | Only Stage 0 (morning filter) is implemented. `ai_caller` passed as `None`. Architecture exists in `engines/stage1_fixture.py` through `stage8_final.py` but they're not wired. |
| **NCAAB Injury Data** | Explicitly skipped — no reliable college basketball injury feeds. Returns neutral fallback. |
| **Soccer Injury Data** | Best-effort only. CBS doesn't cover soccer injuries. RotoWire parse is fragile. `star_player_status` defaults to 5.0 (neutral) for soccer. |
| **BallDontLie API** | Key hardcoded (`BALLDONTLIE_API_KEY`). Used for NBA game logs but unclear if still active. |
| **Tank01 (RapidAPI)** | Key hardcoded (`RAPIDAPI_KEY`). Used for injury enrichment for NBA/NHL/MLB only. |

### Dead Code / Retired Features

| Item | Detail |
|------|--------|
| **Supabase** | Completely removed. Comment says "all storage on Render persistent disk." |
| **Retired crew members** | Startup cleanup removes picks from: Sinton.ia, Midday Edge, Padrino, Los, Sportsbook.ag |
| **Legacy crowdsource block** | Duplicated CROWDSOURCE_MODELS list at line ~7422 — skipped when CROWDSOURCE_FIRST=true (default). Dead code path. |
| **EV_AGENT_VOICE config** | Voice prompt defined but the voice system uses the full agent profile builder instead. |
| **WBC (World Baseball Classic)** | ESPN fetch exists but WBC is seasonal — returns empty outside tournament windows. |

### Frontend Calls Endpoints That Don't Exist

| Frontend Call | Status |
|--------------|--------|
| `/api/injuries/{sport}` | Frontend calls this but the actual endpoint is `/api/lineups/{sport}` — likely a dead reference or alias |
| `/api/beat-report` | Called from frontend but NO corresponding endpoint in server.py |
| `/api/social/{sport}` | Called from frontend but NO corresponding endpoint in server.py |

### Data Source Budget Concerns

| Source | Budget | Status |
|--------|--------|--------|
| SGO (SportsGameOdds) | 100K entities/month (Rookie tier) | Budget gate at 90K. Falls back to Odds API above threshold. |
| The Odds API | Credit-based | `/api/credits` endpoint checks remaining |
| API-Sports | Rate-limited by key | Used for soccer fixtures/predictions/standings/lineups + NHL goalies |
| Tank01 (RapidAPI) | Unknown tier | Hardcoded key — no budget tracking |

---

## 5. SCORING MATRICES (10 sports)

| Sport | Variables | Top Weight | Key Differentiators |
|-------|----------|------------|-------------------|
| NBA | 24 vars | thesis_edge (10) | star_player, rest, line_movement, depth, momentum, second_half |
| NHL | 17 vars | thesis_edge + goalie_confirmed (10) | goalie-centric, process_quality (Corsi/xG), puck_luck (PDO) |
| NCAAB | 24 vars | thesis_edge (10) | regression_markers, public_market_bias, bench_depth, travel_rest |
| Soccer | 17 vars | thesis_edge + line_movement (10) | xg_trend, fixture_congestion, press_intensity, set_pieces |
| MLB | 21 vars | thesis_edge (10) | starting_pitcher, bullpen, park_factor, umpire |
| WNBA | 21 vars | thesis_edge (10) | mirrors NBA with WNBA-specific adjustments |
| NCAAF | 21 vars | thesis_edge (10) | recruiting, transfer_portal, rivalry_factor |
| Tennis | 21 vars | thesis_edge (10) | surface_advantage, h2h_record, serve/return stats |
| MMA | 21 vars | thesis_edge (10) | style_matchup, age_differential, chin_durability |
| Boxing | 17 vars | thesis_edge (10) | style_matchup, age_differential, punch_output |

Each sport also has **chain detection** — compound signal patterns that trigger bonus points (capped at +3.0):
- Examples: THE MISPRICING, FATIGUE FADE, FORM WAVE, TRAP GAME, GOALIE EDGE, REGRESSION HAMMER

---

## 6. KEY FILE PATHS

| Purpose | Path |
|---------|------|
| Server | `C:\Claude\Repos\edge-crew\server.py` |
| Frontend | `C:\Claude\Repos\edge-crew\static\index.html` |
| Engines | `C:\Claude\Repos\edge-crew\engines\` |
| Grade Validator | `C:\Claude\Repos\edge-crew\engines\grade_validator.py` |
| Hurdle Pipeline | `C:\Claude\Repos\edge-crew\engines\hurdle_pipeline.py` |
| Stage 0 Filter | `C:\Claude\Repos\edge-crew\engines\stage0_filter.py` |
| Screener | `C:\Claude\Repos\edge-crew\engines\screener.py` |
| Discrepancy Engine | `C:\Claude\Repos\edge-crew\engines\discrepancy.py` |
| Gap Props | `C:\Claude\Repos\edge-crew\engines\gaps.py` |
| EV Calculator | `C:\Claude\Repos\edge-crew\engines\ev_calc.py` |
| AI Caller | `C:\Claude\Repos\edge-crew\engines\ai_caller.py` |
| Cache Engine | `C:\Claude\Repos\edge-crew\engines\cache.py` |
| DB Module | `C:\Claude\Repos\edge-crew\db.py` |
| Data Directory | `/data/` (Render) or `./data/` (local) |

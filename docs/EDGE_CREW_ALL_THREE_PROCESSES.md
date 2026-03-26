# Edge Crew — All Three Processes (Detailed)

**Date:** March 23, 2026 | **Session:** S223

---

## PROCESS 1: CURRENT LIVE (Crowdsource-First)

`CROWDSOURCE_FIRST=true` on Render. This is what runs when you hit Analyze right now.

---

### Step 1: Data Grab
**Trigger:** `GET /api/analysis/{sport}` called (button click or schedule)

| What happens | Input | Output | Endpoint/Function |
|-------------|-------|--------|-------------------|
| Fetch odds | Sport key | Games with spreads, totals, ML, bookmaker | `get_odds()` → SGO primary, Odds API fallback, SharpAPI fallback |
| Filter started games | Raw games list | Future games only (24h window) | `_game_not_started()` + `_game_within_cutoff()` |
| Separate complete vs incomplete | All games | `complete_games[]` (has spread+total+ML) + `incomplete_games[]` | Lines check in analysis fn |
| Fetch ESPN injuries | Sport | Injury text with freshness tiers [FRESH/RECENT/ESTABLISHED/SEASON] | `_fetch_espn_injuries()` |
| Fetch CBS/RotoWire injuries | Sport | Supplemental injury text | `_get_lineup_and_injury_context()` |
| Fetch Tank01 injury enrichment | Sport + injury text | Enriched injury details | `_build_tank01_injury_context()` |
| Fetch rosters (ESPN) | Sport | Current roster per team (top 15 players) | `get_rosters()` |
| Build team form | Each team name | L5/L10 record, ATS, streak, avg margin, SOS, season record, home/away splits | `_build_team_profile()` |
| Fetch player game logs | Sport + team list | L10 PTS/REB/AST per player | `_fetch_player_game_logs()` (NBA only via BDL) |
| Fetch lineups (soccer/NHL) | Sport | Confirmed starting XI / starting goalie | `_fetch_api_sports_lineups()` |
| Fetch player props | Sport | All props with edge calcs, grouped by matchup | `get_player_props()` → SGO primary, Odds API fallback |
| Fetch gap/cascade props | Sport | Injury absorption props (star OUT → role player absorbs) | `get_gap_props()` |
| Load Step 2.5 overnight batch | Sport | Pre-cached rosters + game logs if available | `_load_step25_batch()` |
| Build matrix section | Sport | 24-26 weighted variables for prompt injection | `_build_matrix_section()` |

**OUT from Step 1:** `games_text`, `injury_context`, `roster_context`, `form_context`, `game_log_context`, `prop_lines_text`, `lineup_context`, `matrix_section`, `incomplete_note`

---

### Step 2: Crowdsource — 4 Models Grade ALL Games

**IN:** `games_text` + `injury_context` + `form_context` + `matrix_section` (combined into `cs_prompt`)

| What happens | Input | Output |
|-------------|-------|--------|
| Build lightweight prompt | All game lines + injuries + form + matrix | Short prompt asking for grade + pick per game |
| Call Mistral Large 3 | cs_prompt | JSON: `{games: [{matchup, grade, composite_score, edge_pick}]}` |
| Wait 3 seconds | — | — |
| Call Llama 4 Maverick | cs_prompt | Same JSON format |
| Wait 3 seconds | — | — |
| Call Grok 3 | cs_prompt | Same JSON format |
| Wait 3 seconds | — | — |
| Call DeepSeek V3.2 | cs_prompt | Same JSON format |

**Models run SEQUENTIALLY** (3s delay between calls to avoid Azure rate limits)

**OUT:** `crowdsource_grades{}` — per matchup: `[{model, grade, score, pick}]`

---

### Step 3: B+ Filter

**IN:** `crowdsource_grades{}` per game

| Rule | Logic | Result |
|------|-------|--------|
| 50%+ models grade B+ or higher | `bp_count >= total * 0.5` | `MAJORITY_BP` → send to deep thinker |
| ALL models grade B+ or higher | `bp_count == total` | `UNANIMOUS_BP` → send to deep thinker |
| Some B+ but < 50% | `bp_count > 0 and bp_count < total * 0.5` | `SPLIT` → crowdsource grade stands, NO deep thinker |
| Zero B+ | `bp_count == 0` | `BELOW_BP` → crowdsource grade stands, NO deep thinker |

**OUT:** `bp_games[]` (games that passed filter) + `all_cs_games[]` (all games with crowdsource data attached)

---

### Step 4: Deep Thinker (ONLY on B+ games)

**IN:** `bp_games[]` (just the game lines that passed the filter) + full context (injuries, form, props, matrix)

| What happens | Input | Output |
|-------------|-------|--------|
| Build thinker prompt | B+ game lines + ALL context (injuries, form, game logs, props, matrix, hard rules) | Full ~6K token thinker prompt |
| Call DeepSeek-R1 (thinker) | Thinker prompt | Freeform reasoning analysis |
| Call DeepSeek-V3.2 (formatter) | Raw analysis from thinker | Structured JSON: matrix_scores, grade, edge_pick, peter_zone, player_props, etc. |
| Merge into crowdsource results | Formatted deep analysis | Deep analysis OVERRIDES crowdsource grade for B+ games |

**Fallback chain:** DeepSeek-R1 fails → GPT-4.1 thinker. DeepSeek-V3.2 formatter fails → GPT-4.1 formatter.

**OUT:** `all_cs_games[]` now has deep analysis merged into B+ games. Non-B+ games keep their crowdsource-only grades.

---

### Step 5: Post-Processing Chain

**IN:** `all_analyzed_games[]`

Runs in this exact order:

| Step | What it does | IN | OUT |
|------|-------------|-----|-----|
| 5a. Matchup normalization | Fix model's team names to match odds format | `"New York Knicks @ Indiana Pacers"` | `"NYK @ IND"` |
| 5b. Matrix retry | Re-analyze games where model forgot matrix_scores | Games missing `matrix_scores` | Filled matrix scores |
| 5c. Server-side grade recalc | Recalculate composite from matrix math, override GPT's grade | `matrix_scores{}` | Corrected `composite_score` + `grade` |
| 5d. Soccer lineup tag | Tag games without confirmed lineups | Lineup data | `LINEUPS NOT CONFIRMED` flag |
| 5e. Alt grade | Secondary grade from team profiles + H2H | Team profiles, H2H data | `alt_grade` field on each game |
| 5f. Player validation (WS4) | Remove props for players not on roster or marked OUT | Roster cache + injury text | Cleaned `player_props[]` |
| 5g. Incomplete games | Add INCOMPLETE-graded games to response | `incomplete_games[]` | Games with `grade: "INCOMPLETE"` |
| 5h. Grade validation gate | Server recalculates composite from matrix, maps to correct grade | `matrix_scores{}` | `_score_override`, `_grade_override` if GPT math was wrong |
| 5i. Challenger model | Weekly rotating model grades batch 0 for comparison | Batch 0 prompt | `challenger_grade`, `challenger_score`, `challenger_model` |
| 5j. Edge score + chains | Detect multi-variable chain patterns (e.g., CINDERELLA SIGNAL) | Matrix scores | `edge_score`, `edge_grade`, `chains[]`, `chain_bonus` |
| 5k. Dual grade | Combine edge_grade + composite grade | Both scores | `dual_grade` field (e.g., "A / B+") |
| 5l. Stale injury clamp | Downgrade if team winning without injured star | L5 record + injury freshness | Grade capped (SEASON → max B-, ESTABLISHED → max B+) |
| 5m. Grade validator | Post-analysis rule checks (ML dodge, momentum, etc.) | All games | Validated grades |
| 5n. Prop edge injection | Attach book discrepancies + gap props to each card | Discrepancy data + gap data | `prop_edges[]` on each game (top 5) |

**OUT:** Final `analysis{}` object with all games fully processed

---

### Step 6: Cache + Return

| What happens | Output |
|-------------|--------|
| Set memory cache | `_set_cache(cache_key, analysis)` |
| Save to disk | `_save_analysis_cache(sport, analysis)` |
| Return JSON response | Full analysis with gotcha, games, metadata |

**Card displays:** grade, composite, edge_pick, peter_zone, matrix_scores, tags, crowdsource consensus, dual grade, prop_edges, challenger grade, flags

---

---

## PROCESS 2: LEGACY (Thinker-First)

`CROWDSOURCE_FIRST=false` or `?model=` override. This was the process BEFORE commit `cccab61`.

---

### Step 1: Data Grab
**Identical to Process 1 Step 1.** Same data, same sources, same outputs.

---

### Step 2: Batch Games

**IN:** `complete_games[]` (game line strings)

| What happens | Input | Output |
|-------------|-------|--------|
| Split into groups of 4 | All complete game lines | `batch_prompts[]` — max 6 batches (24 games) |
| Build full prompt per batch | Game lines + ALL context | One prompt per batch with gotcha (batch 0 only) |

**OUT:** `batch_prompts[]`, `batch_game_lists[]`, `batch_prop_texts[]`

---

### Step 3: Thinker + Formatter (parallel batches)

**IN:** `batch_prompts[]` — all batches run in parallel via `asyncio.gather`

| What happens per batch | Input | Output |
|------------------------|-------|--------|
| Build thinker prompt | Batch game lines + context | Freeform reasoning prompt (no JSON pressure) |
| Call thinker (Grok-3 default) | Thinker prompt | Raw analysis text |
| Validate thinker output | Raw text | Check length ≥ 200 chars, check game coverage |
| Call formatter (DeepSeek-V3.2) | Raw analysis + JSON schema | Structured JSON with matrix_scores, grades, picks |
| Formatter retry (1x) | Same input | Retry if first attempt fails |
| Fallback to GPT-4.1 single-model | Full JSON prompt | Structured JSON (if both thinker+formatter fail) |

**Fallback chain per batch:**
```
Grok-3 (think) → DeepSeek-V3.2 (format)
  ↓ fail
Grok-3 (think) → DeepSeek-V3.2 (format retry)
  ↓ fail
GPT-4.1 single-model (think + format)
```

**Failed batches get 1 retry.** If retry also fails, those games are lost.

**OUT:** `all_analyzed_games[]` — every game with grade, matrix_scores, edge_pick, peter_zone, etc. + `gotcha_html` from batch 0

---

### Step 4: Post-Processing

**Identical to Process 1 Steps 5a through 5n.** Same chain, same order.

---

### Step 5: Crowdsource (DIFFERENT Models)

**IN:** `batch_prompts[0]` (first batch only) — sent to 6 models in parallel

| Model | Status |
|-------|--------|
| Grok 3 | In pool |
| Kimi K2.5 | In pool |
| Mistral Large 3 | In pool |
| Qwen 3 32B | In pool |
| Llama 4 Maverick | In pool |
| Cohere Command A | In pool |

Each model gets the same prompt, grades independently. Results merged per matchup.

**Consensus rules applied:**
- B+ or higher from majority → `CONSENSUS` tag
- Split grades → `BELOW_BP` (no consensus)

**OUT:** `crowdsource_grades{}` attached to each game. Consensus tag applied.

---

### Step 6: Cache + Return
**Identical to Process 1 Step 6.**

---

---

## PROCESS 3: PETER'S 6-STEP DESIGN (The Target)

This is the intended architecture. Not fully implemented yet.

---

### Step 1: Overnight Data Grab

**WHEN:** Automated overnight (before Peter wakes up)
**TRIGGER:** Step 2.5 batch (`POST /api/step25/run`) or scheduled cron

| What gets pulled | Source | Stored where |
|-----------------|--------|-------------|
| Odds for all sports | SGO + Odds API | Daily slate cache (disk) |
| Rosters (all teams playing) | ESPN | Memory cache + disk |
| Game logs (L10 per player) | BallDontLie (NBA) | Memory cache + disk |
| Injuries (all sports) | ESPN + CBS + RotoWire + Tank01 | Memory cache |
| Lineups (soccer/NHL) | API-Sports | Memory cache |
| Props (all sports) | SGO + Odds API | Memory cache |
| Team form (L5/L10/season) | Scores archive | Computed on demand |

**Cards go up on site** with raw data — odds, injuries, form — before any AI grading.

**GAP:** No automated overnight trigger. Must be manually kicked or scheduled via `/schedule`. Step 2.5 exists but isn't on a cron.

---

### Step 2: Data Quality Filter

**WHEN:** Before any AI model touches the game
**RULE:** 80%+ data completeness = proceed. Below 80% = don't grade.

| Data point | Weight | How to check |
|-----------|--------|-------------|
| Spread available | 20% | `home_spread is not None` |
| Total available | 15% | `total is not None` |
| ML available | 15% | `away_ml is not None` |
| Injury data present | 20% | At least 1 injury source returned real data |
| Roster confirmed | 15% | Team appears in roster cache |
| Form data available | 15% | `_build_team_profile()` returns L5 data |

**GAP:** This gate does NOT exist in code. Server grades everything with lines, even with 0% injury data. Need to build a `_data_quality_score()` function and block grading below 80%.

---

### Step 3: Initial Grading — Sinton.ia

**WHO:** Sinton.ia (the OG grader)
**MODELS:** Grok-3 (thinker) + DeepSeek-V3.2 (formatter) — ONLY these two
**THESE MODELS ARE EXCLUDED FROM STEP 4.**

| What happens | Input | Output |
|-------------|-------|--------|
| Sinton.ia receives full context | Odds + injuries + form + game logs + props + matrix | Freeform analysis from Grok |
| Grok reasons (no JSON pressure) | Thinker prompt with hard rules | Raw thinking: edge question, matchup analysis, injury impact |
| DeepSeek formats | Grok's raw analysis + JSON schema | Structured card: matrix_scores (24-26 vars), grade, composite, edge_pick, peter_zone, player_props |
| Server recalculates | matrix_scores | Verified composite_score + correct grade (override GPT if math wrong) |

**OUT:** First batch of graded cards. Every game has a grade, a pick, a peter_zone.

**CURRENT STATUS:** This IS the legacy thinker-first path (Process 2 Step 3). It works. But it's not labeled as "Sinton.ia" — it's just "the analysis engine."

---

### Step 4: Crowdsource — DIFFERENT Models

**WHO:** AI profile (the Engine)
**RULE:** Do NOT use Grok or DeepSeek. Fresh eyes from different architectures.

| Model | Architecture | Why it's different |
|-------|-------------|-------------------|
| Kimi K2.5 | Moonshot (Chinese) | Different training data, different reasoning style |
| Llama 4 Maverick | Meta | Open-source, different optimization |
| Mistral Large 3 | Mistral | European training, different priors |
| Phi-4-reasoning | Microsoft | Small but reasoning-focused |
| Cohere Command A | Cohere | Enterprise-tuned, different grounding |
| Qwen 3 32B | Alibaba | Chinese architecture, different pattern recognition |
| gpt-oss-120b | Open-source | Large open model, different biases |

Each model grades independently against the SAME matrix. No model sees another model's grade.

**GAP:** In the current live system (Process 1), Grok and DeepSeek ARE in the crowdsource pool. Peter's design says they should be EXCLUDED. The legacy crowdsource (Process 2 Step 5) also includes Grok in the pool.

---

### Step 5: Consensus Rules

**IN:** All grades from Steps 3 + 4

| Scenario | Rule | Action |
|----------|------|--------|
| All models B+ or higher | UNANIMOUS consensus | **Best bet candidate** — green flag |
| Majority B+ or higher | MAJORITY consensus | **Strong candidate** — worth a look |
| Split (some A, some C) | DISAGREEMENT | **Dig deeper** — run more models, investigate WHY they disagree |
| All below B+ | No edge | **PASS** — no bet |

**Renzo's role:** Between Steps 5 and 6. After consensus, Renzo scans for prop edges the engine missed — book discrepancies, injury absorption plays, line lag. These get attached to the card as `prop_edges`.

**GAP:** Disagreement doesn't auto-trigger "run more models." It just logs. Should expand the crowdsource pool when splits are detected.

---

### Step 6: Peter Decides

**WHAT:** All data is on the card. Peter makes the final call.

| What Peter sees | Source |
|----------------|--------|
| Grade (server-verified) | Step 3 + server recalc |
| Dual grade (edge vs composite) | Chain detection engine |
| Crowdsource consensus | Step 4-5 |
| Challenger model grade | Weekly rotating model |
| Alt grade (team profiles + H2H) | Post-processing |
| Prop edges (book discrepancies) | Renzo / prop edge engine |
| Matrix scores (all 24-26 vars) | Step 3 thinker+formatter |
| Peter Zone (sizing suggestion) | AI analysis |
| Edge pick with reasoning | AI analysis |
| Player props with grades | AI + roster validation |
| Injury impact (specific players) | ESPN + CBS + RotoWire |
| Team form (L5/L10/ATS/streak) | Scores archive |
| Public betting splits | Odds source |
| Flags (traps, B2B, lineup unconfirmed) | Post-processing |

Peter locks the pick → stored in `/api/picks`. Prop board tracks crushed lines.

---

---

## KEY DIFFERENCES SUMMARY

| Aspect | Process 1 (Current Live) | Process 2 (Legacy) | Process 3 (Peter's Design) |
|--------|------------------------|--------------------|-----------------------------|
| **Order** | Crowdsource FIRST → deep thinker on B+ only | Thinker FIRST → crowdsource validates | Thinker FIRST → crowdsource validates |
| **Who thinks first** | 4 lightweight models | Grok-3 | Sinton.ia (Grok-3) |
| **Who formats** | Consensus (no formatting) → DeepSeek on B+ | DeepSeek-V3.2 | DeepSeek-V3.2 |
| **Crowdsource pool** | Mistral, Llama, Grok, DeepSeek | Grok, Kimi, Mistral, Qwen, Llama, Cohere | Kimi, Llama, Mistral, Phi, Cohere, Qwen, gpt-oss (NO Grok, NO DeepSeek) |
| **Grok in crowdsource?** | YES (shouldn't be) | YES (shouldn't be) | NO — excluded |
| **DeepSeek in crowdsource?** | YES (shouldn't be) | NO | NO — excluded |
| **Data quality gate** | No gate | No gate | 80% threshold required |
| **Renzo's role** | Not in pipeline | Not in pipeline | Prop edge finder between consensus and Peter |
| **Sinton.ia's role** | Not labeled | Not labeled | IS the initial grader |
| **Overnight automation** | Manual | Manual | Automated cron |
| **Non-B+ games** | Get crowdsource-only grade (no deep analysis) | Get full thinker+formatter analysis | Get full analysis from Sinton.ia |
| **Disagreement handling** | Tag only | Tag only | Auto-expand crowdsource pool |

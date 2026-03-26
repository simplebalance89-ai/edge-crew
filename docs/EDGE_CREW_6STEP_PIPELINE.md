# Edge Crew — Peter's 6-Step Pipeline

**Owner:** Peter | **Architect:** Pietra | **Date:** March 23, 2026

---

## The Process

```
STEP 1          STEP 2           STEP 3              STEP 4              STEP 5            STEP 6
Overnight  -->  Data Quality -->  Initial Grade  -->  Crowdsource    -->  Consensus    -->  Peter
Data Grab       Filter (80%+)    Grok + DeepSeek     DIFFERENT models    B+ rule           Decides
                                 (thinker+formatter)  (Kimi, Llama,      splits = dig
                                                      Mistral, etc.)     deeper
```

---

## Step 1: Overnight Data Grab

**What:** Pull everything — lineups, injuries, odds, form, stats. Cards go up on site.

| Data Source | Endpoint | Status |
|------------|----------|--------|
| Odds (SGO primary) | `GET /api/odds/{sport}` | WORKING |
| Odds (The Odds API fallback) | same endpoint, fallback path | WORKING |
| ESPN Injuries | `_fetch_espn_injuries()` | WORKING (NBA/NHL/NFL/MLB) |
| CBS/RotoWire Injuries | `_get_lineup_and_injury_context()` | WORKING |
| Tank01 Injury Enrichment | `_build_tank01_injury_context()` | WORKING (free tier, 1K/mo) |
| ESPN Rosters | `GET /api/rosters/{sport}` | WORKING |
| Player Game Logs (BDL) | `_fetch_player_game_logs()` | WORKING (NBA only) |
| Team Form (L5/L10) | `_build_team_profile()` | WORKING |
| Lineups (API-Sports) | `_fetch_api_sports_lineups()` | WORKING (soccer/NHL) |
| Player Props (SGO) | `GET /api/props/{sport}` | WORKING |
| Step 2.5 Overnight Batch | `POST /api/step25/run` | WORKING |

**GAPS:**
- NCAAB injuries — NO source configured. ESPN/CBS don't cover college injuries well.
- Soccer injuries — partial (API-Sports only, no ESPN)
- Game logs only for NBA (BDL). NHL/MLB/soccer/NCAAB = no player game logs.
- No automated overnight trigger — Step 2.5 batch must be manually kicked or scheduled.

---

## Step 2: Data Quality Filter

**What:** 80%+ data completeness = proceed. Below 80% = don't grade.

| Check | Endpoint | Status |
|-------|----------|--------|
| Lines complete (spread + total + ML) | Built into `get_analysis()` | WORKING |
| Incomplete games tagged | `INCOMPLETE` grade assigned | WORKING |
| Injury data present | Warning injected if all sources fail | WORKING |
| Roster data present | Validation post-AI | WORKING |

**GAPS:**
- No explicit 80% threshold gate. Server grades everything with lines, even if injury/roster data is empty.
- No data quality score shown on the card — Peter can't see "this card was graded with 60% data."
- Should block grading when injury sources all fail, currently just warns in the prompt.

---

## Step 3: Initial Grading (Thinker + Formatter)

**What:** Grok reasons freely, DeepSeek structures into JSON with matrix scores.

| Component | Model | Endpoint | Status |
|-----------|-------|----------|--------|
| Thinker | Grok-3 (Azure) | `get_analysis()` | WORKING |
| Thinker fallback | GPT-4.1 | same | WORKING (auto-fallback) |
| Formatter | DeepSeek-V3.2 | same | PARTIAL — rate limited on free tier |
| Formatter fallback | GPT-4.1 | same | WORKING |
| Matrix scoring | 24-26 variables per sport | built into prompt | WORKING |
| Grade gate | Server recalculates composite from matrix | post-processing | WORKING |
| Score override logging | `_score_override` / `_grade_override` | post-processing | WORKING |

**GAPS:**
- DeepSeek-R1 rate limited (1 req/0s on free tier). Falls back to GPT-4.1 frequently. Need paid deployment or different model.
- Thinker prompt is massive (~6K tokens of rules). Could be trimmed.
- No separation between "Grok's raw thinking" and "structured output" on the card — Peter can't see what Grok actually thought vs what DeepSeek formatted.

---

## Step 4: Crowdsource (DIFFERENT Models)

**What:** Fresh models that DIDN'T do initial grading re-grade against the same matrix. No Grok, no DeepSeek.

| Component | Models | Status |
|-----------|--------|--------|
| Crowdsource pool | Configured in `CROWDSOURCE_MODELS` | WORKING |
| Exclusion of thinker/formatter | Grok + DeepSeek excluded from pool | WORKING |
| Independent grading | Each model grades same games | WORKING |
| Per-model grade tracking | Stored on each game card | WORKING |

**Current crowdsource models (verify on server):** Should be Kimi K2.5, Llama 4, Mistral Large 3, Phi-4-reasoning, Cohere, Qwen, gpt-oss-120b — but need to verify which are actually deployed and working on Azure.

**GAPS:**
- Azure quota limits may block some models. Need to verify which models in the pool are actually callable.
- Crowdsource runs sequentially — could be parallelized for speed.
- No "which models agreed vs disagreed" breakdown visible on the card in a clean way.

---

## Step 5: Consensus Rules

**What:** B+ or higher across models = best bets. Splits = investigate. Below B+ = pass.

| Rule | Implementation | Status |
|------|---------------|--------|
| B+ consensus filter | `crowdsource_consensus` field on each game | WORKING |
| Count B+ or higher | Tracks `bp_or_higher / total_models` | WORKING |
| CONSENSUS tag | Applied when majority B+ | WORKING |
| BELOW_BP tag | Applied when consensus fails | WORKING |
| Split/disagreement flag | Grade spread tracked | PARTIAL |

**GAPS:**
- "Disagreement IS the signal" — when models disagree by 1+ full grade, that should trigger a visible WARNING on the card. Currently just logged, not prominent.
- No "run more models" auto-trigger when disagreement detected. Should auto-expand crowdsource pool on splits.
- No historical tracking of "models agreed, did the bet hit?" — would improve model selection over time.

---

## Step 6: Peter Decides

**What:** All grades, all model outputs, all data on the card. Peter makes the final call.

| Feature | Status |
|---------|--------|
| Full card with composite + grade | WORKING |
| Matrix scores visible | WORKING |
| Edge pick with reasoning | WORKING |
| Peter Zone (sizing suggestion) | WORKING |
| Player props on card | WORKING (from AI) |
| Prop edges (book discrepancies) | JUST DEPLOYED — `prop_edges` on each card |
| Dual grade (edge score vs composite) | WORKING |
| Crowdsource consensus visible | WORKING |
| Lock/pick tracking | WORKING (`/api/picks`) |

**GAPS:**
- Prop edges just deployed — need to verify they render on the frontend card.
- No "one-click lock" from card to bet slip.
- No P&L tracking tied to picks — Peter tracks manually.
- Card doesn't show data quality % — how much data was available when this was graded.

---

## Summary: What Works vs What's Missing

| Step | Status | Critical Gap |
|------|--------|-------------|
| 1. Data Grab | WORKING | NCAAB injuries missing, game logs NBA-only |
| 2. Quality Filter | PARTIAL | No 80% threshold gate, no quality score on card |
| 3. Initial Grade | WORKING | DeepSeek rate limited, falls back to GPT-4.1 |
| 4. Crowdsource | WORKING | Azure quota may block models, verify pool |
| 5. Consensus | PARTIAL | Disagreement not prominent enough on card |
| 6. Peter Decides | WORKING | Prop edges need frontend rendering |

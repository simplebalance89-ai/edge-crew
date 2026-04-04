# Edge Crew — Pro Edge V4 Architecture

**Deployed:** April 4, 2026 | **17 commits** | **Live at:** edge-crew-v2.onrender.com

---

## Pipeline Overview

```
S0 (Team Data) → S1 (Schedule) → S2 (Players)
                       |
           +-----------+-----------+
           |                       |
      OUR PROCESS             AI PROCESS
      Deterministic           Independent
      Matrix Grading          Model Grading
           |                       |
           +-----------+-----------+
                       |
                 CONVERGENCE
              LOCK / ALIGNED /
             CLOSE / DIVERGENT
                       |
               KIMI GATEKEEPER
            CONFIRM / CHALLENGE / BOOST
                       |
                  FINAL GRADE
                  Pick + EV + Sizing
```

---

## Stage Breakdown

### S0 — Team Data (Shared Foundation)
- Spread, O/U, ML from The Odds API (primary)
- Per-team L5 record, streak, ATS record
- Engine snippets (pace, goalie, pitchers, form)

### S1 — Schedule (Shared Foundation)
- Rest days per team (color-coded: 0d=red, 2+=green)
- B2B detection with badge
- Games in last 7 days, road trip length
- Matches in 10 days (soccer congestion)
- Sport-specific: goalie (NHL), starting pitcher (MLB)

### S2 — Players (Shared Foundation)
- PPG L5 / Opp PPG (NBA/NCAAB), GF/GA (NHL), ERA/WHIP (MLB)
- Injury counts: OUT / GTD with top impact players by PPG
- KenPom AdjO/AdjD (NCAAB)

---

## Left Lane: OUR PROCESS

### Main Matrix Analysis
- 15-24 variables per sport, each scored 1-10
- Variables include: star_player_status, rest_advantage, off_ranking, def_ranking, pace_matchup, recent_form, road_trip, h2h, ATS trend, depth_injuries, and sport-specific vars
- Weights defined in `profiles/sintonia.json` (weight 4-10 per variable)
- Composite = weighted average scaled to 1-10

### Spread Amplifier (NEW)
- Top-3 highest-impact variables get 2x influence on composite
- If top 3 are all 8+: composite pulls UP (real edge detected)
- If bottom 3 are all 3-: composite pulls DOWN (garbage game)
- Floor gate: any variable 9.5+ forces minimum B+ grade
- Ceiling gate: any variable 1.5- caps at C grade

### Chain Detection System (30 chains)

**Positive Chains (boost grade):**
| Chain | Bonus | Trigger |
|-------|-------|---------|
| MISMATCH_MASSACRE | +2.0 | Elite offense + solid defense |
| FORM_WAVE | +1.5 | Hot team + elite offense + ATS + H2H |
| INJURY_GOLDMINE | +1.5 | Star OUT + line flat + good form |
| SHARPS_LOVE | +1.5 | Sharp money + line moving + form |
| BLOWOUT_INCOMING | +1.5 | Elite OFF + DEF + home (NBA/NCAAB) |
| GOALIE_EDGE | +1.5 | NHL: backup goalie + bad metrics |
| ACE_DOMINATION | +1.5 | MLB: elite SP + platoon + bullpen |
| CONGESTION_FADE | +1.5 | Soccer: congested opponent |
| HUNGRY_DOG | +1.5 | Underdog in form + motivated + sharps notice |
| PUBLIC_TRAP | +1.5 | Public hammering one side, line won't move |
| THE_MISPRICING | +1.0 | Star OUT + line flat + no movement |
| FATIGUE_FADE | +1.0 | Rest + road + depth advantage |
| TRAP_GAME | +1.0 | Sharp vs public divergence |
| VALUE_DOG | +1.0 | Sharps on dog + healthy |
| REST_DOMINATION | +1.0 | Rest + home + opponent traveling |
| REVENGE_GAME | +1.0 | Lost H2H + in form + home |
| ROAD_WARRIOR | +1.0 | Away but in form + rested |
| BOUNCE_BACK | +1.0 | Elite team on cold streak at home |
| BENCH_MOB | +1.0 | Deep rotation + healthy + in form |
| CLASS_GAP | +1.0 | Soccer: 10+ league position gap |
| FORTRESS_HOME | +1.0 | Soccer: elite home record + clean sheets |
| DERBY_CHAOS | +1.0 | Soccer: rivalry + form + stakes |
| PITCHING_DUEL | +1.0 | MLB: both aces + pitcher park |
| PLATOON_EDGE | +1.0 | MLB: favorable L/R matchup |
| BLUE_BLOOD_TRAP | +1.0 | NCAAB: public on big name, rankings close |

**Negative Chains (pull grade DOWN):**
| Chain | Penalty | Trigger |
|-------|---------|---------|
| SCHEDULE_LOSS | -2.0 | B2B + road + bad form |
| DUMPSTER_FIRE | -2.0 | Bad form + bad offense + injuries |
| COLD_TAKE | -1.5 | Average score below 4.5 — no edge anywhere |
| COASTING_FAV | -1.5 | Favorite with no motivation |
| THIN_ROSTER | -1.5 | Stars OUT + no bench depth |
| TOURIST_TRAP | -1.5 | Soccer: big club away + congested + tough venue |
| GLASS_CANNON | -1.0 | Great offense but garbage defense |
| LETDOWN_ALERT | -1.0 | On huge streak but no motivation |
| FADE_THE_STREAK | -1.0 | Long win streak + away + tired |
| WRONG_SIDE | -1.0 | MLB: bad platoon vs dominant SP |
| BRICK_WALL | -0.5 | Elite defense but can't score |

**Chain cap:** 3.0 in both directions (positive and negative)

### Grader Profiles
| Profile | Style | Key Focus |
|---------|-------|-----------|
| **Sintonia** | Data-driven weighted matrix | 15-24 variables per sport with chain bonuses |
| **Edge** | Situational/calendar spots | Schedule, travel, letdown, lookahead traps |
| **Renzo** | 5 Questions + Mathurin Test | Independent check — caps weak picks at 5 without thesis |
| **Claude** | Contrarian/fade the public | Line movement, public bias, name inflation |

### Peter's Rules (Kill/Boost overlay)
- KILL (-3.0): Big favorite ATS trap, caps score at 2.9
- DOWNGRADE (-1.0 to -0.5): Blow-lead teams, established injuries priced in
- BOOST (+1.0): Fresh injury within 3 days
- Soccer-specific: congestion KILL, European hangover, draw trap

---

## Right Lane: AI PROCESS

### Crowdsource Models (5 models in parallel)
| Model | Endpoint | Purpose |
|-------|----------|---------|
| Grok 4.1 Fast | Azure AI Services | Deep reasoning, fast |
| GPT 5.4 Nano | Azure (peterwilson) | Latest OpenAI nano model |
| DeepSeek R1 | Azure AI Services | Primary thinker, independent grade |
| Kimi K2.5 | Azure AI Services | Profiler + gatekeeper |
| Claude Sonnet 4.6 | Anthropic API | Contrarian perspective |

Each model independently grades every game and picks a side.
Consensus: UNANIMOUS_BP / MAJORITY_BP / SPLIT / BELOW_BP

### Kimi Scout (4-Dimension Profiler)
- **Tactical DNA** (0-10): style clash, pace mismatch, scheme edge
- **H2H Context** (0-10): historical patterns, rivalry factor
- **Structural Edge** (0-10): rest, travel, B2B, congestion
- **Market Signal** (0-10): grader consensus, chain bonuses, EV, sharp signals
- Profile score = average of 4 dimensions
- Profile tag: narrative label (e.g., "DIVERGENT SHARP", "REST EDGE VALUE")

---

## Convergence

Both lanes merge and are compared:
- **OUR score** (consensus of Sintonia/Edge/Renzo/Claude matrix grades)
- **AI score** (average of crowdsource model scores + Kimi Scout)
- **Agreement badge:**
  - **LOCK** (gap <= 0.8): both lanes agree strongly
  - **ALIGNED** (gap <= 1.5): same direction, minor variance
  - **CLOSE** (gap <= 2.5): similar but notable difference
  - **DIVERGENT** (gap > 2.5): lanes disagree — dig deeper

---

## Kimi Gatekeeper (Post-Convergence)

The FINAL validation layer. Kimi reviews the full pipeline output and adjusts:

- **CONFIRM** (+0): grade looks right, no red flags (should be RARE)
- **CHALLENGE** (-1 to -2): grade is too high — grader divergence, negative EV, trap game, big spread
- **BOOST** (+1 to +2): grade is too low — all graders agree, chains fired, exceptional EV

### Gatekeeper Rules (aggressive, not rubber-stamp):
- MUST CHALLENGE when: grader divergence 2+ points, negative EV with high grade, GLASS_CANNON/DUMPSTER_FIRE chains, spreads 15+
- MUST BOOST when: all 4 graders agree + positive EV, MISMATCH_MASSACRE/SHARPS_LOVE chains, EV exceeds +5%
- CONFIRM only when genuinely no flags — most games should get adjusted
- Must TAKE A SIDE when graders disagree, not hedge

---

## Post-Convergence Display

After the final grade, a horizontal row shows:
- **Peter's Report**: kill/boost flags with reasoning
- **EV Grade**: expected value %, win probability, Kelly sizing
- **Kalshi**: market-implied probability + edge divergence
- **Renzo Independent**: standalone Mathurin test grade (if batch-graded)

---

## Grade Scale

| Grade | Score | Sizing |
|-------|-------|--------|
| A+ | >= 8.5 | 2u |
| A | >= 7.8 | 1.5u |
| A- | >= 7.0 | 1u |
| B+ | >= 6.5 | 1u |
| B | >= 6.0 | PASS |
| B- | >= 5.5 | PASS |
| C+ | >= 5.0 | PASS |
| C | >= 4.0 | PASS |
| D | >= 3.0 | PASS |
| F | < 3.0 | PASS |

---

## Weight Calibration (April 4, 2026)

Key changes to fix grade inflation (everything was B/B-):
- `score_off_ranking`: average offense now scores 5 (was 7)
- `score_def_ranking`: average defense now scores 5 (was 6.5)
- `score_home_away`: home base 5.5 (was 6), away 4.5 (was 5)
- `score_pace_matchup`: aligned pace = 5 (was 6)
- `score_road_trip`: home stand 2 games = 5.5 (was 6)
- `score_recent_form`: 3-2 L5 = 5 (was 5.5)
- Matchup boosts: +0.5 (was +1.0)
- Negative chains pull bad games to C/D/F instead of sitting at B

---

## Data Sources

| Source | Status | Purpose |
|--------|--------|---------|
| The Odds API | Primary (4.5M credits) | Odds for all sports |
| SportsGameOdds | DISABLED (403) | Was primary, dead as of Apr 4 |
| ESPN | Active | Scores, rosters, injuries, box scores |
| API-Sports | Active | Lineups, fixtures, standings |
| DeepSeek R1 | Azure AI Services | Primary analysis thinker |
| DeepSeek V3 | Azure AI Services | Formatter + S4 Bridge |
| Kimi K2.5 | Azure (Sweden Central) | Scout profiler + Gatekeeper |
| Grok 4.1 | Azure AI Services | Crowdsource + S4 Bridge fallback |
| GPT 5.4 Nano | Azure (peterwilson) | Crowdsource model |
| Claude Sonnet 4.6 | Anthropic API | Crowdsource + challenger |

---

## Frontend Location

**Pro Edge is in `static/index.html`** (`panel-profedge`), NOT `peter.html`.
- `peter.html` serves the `/peter` route (secondary, also updated)
- Pro Edge tab in bottom nav bar on the main slate
- Sport selector pills at top (NBA, NCAAB, NHL, NFL, MLB, Soccer, etc.)

---

## Soccer Support

116 games across 14 leagues:
EPL, La Liga, Bundesliga, Serie A, Ligue 1, MLS, Liga MX, UCL, Europa League, FA Cup, EFL Championship, Brasileirao, Argentina, Scottish PL

League filter pills in Pro Edge for soccer (same as main slate).

---

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `server.py` | ~17,500 | Backend: all endpoints, analysis, grading, gatekeeper |
| `grade_engine.py` | ~2,800 | Scoring functions, chain detection, composite calc |
| `profiles/sintonia.json` | ~200 | Variable weights + chain configs per sport |
| `static/index.html` | ~11,900 | Main slate + Pro Edge frontend |
| `static/peter.html` | ~1,100 | Secondary Pro Edge page |
| `engines/` | 25+ files | Sport-specific engine modules |

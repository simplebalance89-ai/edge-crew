# Pro Edge Pipeline Spec
Last Updated: 2026-03-19 (Session 209)

## Pipeline: 7 Steps, Left to Right

### Step 1 — DATA
- Raw odds, lines, spreads, ML, O/U
- Source: SGO primary, Odds API fallback
- Runs: real-time, updates as lines move
- Complete when: all lines are in for the game

### Step 2 — ENRICHMENT
- Team stats: L5, L10, PPG, Opp PPG, streak, rest days
- Injuries: who's OUT, GTD, with impact rating
- ATS record
- Tap to expand: full lineup, player props for that lineup
- Complete when: lineup is confirmed = 100%
- Source: CBS Sports, ESPN, RotoWire, API-Sports

### Step 2.5 — TEAM + PLAYER PROFILES (overnight batch)
- **Runs the NIGHT BEFORE** as batch job
- Team profile card: overall grade based on L5 games, current form, off/def ratings, ATS, injuries
- Player profile card: L10 stats, usage, matchup advantage, prop history
- Driven by: last 5 games + last 3 H2H against opponent + current matchup
- By morning: every team and player profile for tomorrow's slate is pre-built
- This is what feeds Step 3 — variables score off structured profiles, not raw data
- STATUS: Need to crowdsource criteria (same process as Step 3 variables)
- HISTORY: Peter + Renzo crowdsourced this, work was lost. Must redo and save 3x.

### Step 3 — VARIABLES (scored criteria)
- 20-25 variables per sport, each scored 1-10
- Variables are sport-specific (NBA has different vars than NHL)
- Each profile (Sintonia, Renzo, Alyssa, Peter) can weight variables differently
- Crowdsourced from 6 AI models: GPT-4.1, Grok, GPT-5.1, DeepSeek, o4-mini, Kimi K2.5
- Peter's variables confirmed by Kimi independently: 3pt variance, minutes load, lookahead letdown
- Variables pull from Step 2.5 profiles, not raw data
- STATUS: 6-model crowdsource DONE. Master consensus list needs Peter review to lock.

### Step 4 — GRADERS (swim lanes)
- Each profile grades independently using their own variable weights
- Sintonia: 15 variables + 7 chain bonuses (cap 3.0)
- Renzo: 6 questions + Mathurin Test gate (thesis_edge weight 10)
- Alyssa: TBD — needs profile definition
- Peter: TBD — analytical pattern recognition codified
- All profiles visible side by side on one card
- Chains shown as green badges when 2+ variables fire together
- Profiles are LOCKED DOWN — only Peter and Alyssa see the configs

### Step 5 — ENGINE VERDICT
- Takes all grader outputs, runs consensus
- Final grade (A+ through F)
- Pick: team + spread/ML + sizing
- Conviction level
- EV calculation

### Step 6 — RACE (consensus visualization)
- Swim lane race showing which graders agree
- Home vs Away score across all heats
- Visual convergence indicator

### Step 7 — MARKET VALIDATION (Kalshi + Polymarket)
- Compare engine probability against Kalshi game-level spread contracts
- Multiple lines per game = probability CURVE
- Divergence between our model and market = edge signal or warning
- Polymarket for futures/macro validation (championship odds, MVP)
- Kalshi series: KXNCAAMBSPREAD, KXNBA1HSPREAD, KXNFLSPREAD

## Per-Card Percent Complete
- Each step shows its own % complete
- Game card header shows average of all steps
- Slate header shows average across all games
- Step 2 at 100% = lineup locked = game is ready to grade
- Below 100% = some data missing, grades are soft

## Two Products
- **V2 (main site)** — bells and whistles, fun, for the crew, college kids
- **Pro Edge** — professional, pipeline visualization, serious money

## Three Backups
- Render persistent disk (/data/)
- GitHub (committed to repo)
- Local (ThinkStation)

## Three Machines
- ThinkStation: Alyssa + Peter (work, building, Edge Crew)
- Mac Mini: Alyssa Spawn (personal research, async deep dives)
- Razer (Dekan): Chiara (support, testing)

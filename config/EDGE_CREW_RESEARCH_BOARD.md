# Edge Crew Research Board
Last Updated: 2026-03-19

## NEW TOOLS — CONFIRMED AVAILABLE
- **Kimi K2.5** — LIVE on Azure. Tested. Returned 24 NBA variables. Independently validated Peter's three-point variance, minutes load, lookahead letdown. Role: junior/intern in pipeline.
- **Kimi K2-Thinking** — Also on Azure. Reasoning model variant. Not yet tested.
- **Polymarket API** — Prediction market probabilities as VALIDATION LAYER. If our model says 65% and Polymarket says 52%, that divergence IS the edge signal. Kalshi saw $208M in March Madness 2025 trading volume. Minerva Networks integrated Polymarket data as "stats for the future." NEED TO CHECK: MCP server or direct API.

## AI MODELS TO TEST VARIABLES AGAINST
1. **Foretell (China)** — First nationally certified sports prediction AI. Multi-agent system with dedicated "scouts" for data collection, tactical analysis, risk warning. Trained on millions of tactical reports and coaching materials. Maps to our Matchup Exploit Potential and Coaching Tactical Edge variables. CHECK: API availability.
2. **Rithmm** — Consumer AI specializing in personalized betting strategies. Focus on player props and usage patterns. Maps to our Heat 2 (Player Chains) and Usage Cascade when stars are OUT. CHECK: API access.
3. **Sharp Sports / Pikkit** — ML for analyzing betting history and surfacing overlooked edges. Pikkit tracks personal betting patterns to highlight strengths. Maps to our ATS Trend and Sharp Money variables. CHECK: API/data feed.
4. **4C Predictions** — Ran $1M head-to-head against a pro gambler for March Madness. Pattern recognition in tournament settings. Maps to Shooting Variance Regression and Market Overreaction. CHECK: methodology docs.

## CROWDSOURCING APPROACHES
1. **Prediction Markets as Validation Layer** — Polymarket + Kalshi. Treat market probabilities as an additional "model" in consensus layer. When our probability diverges significantly from market consensus, that divergence itself is an edge signal.
2. **Open-Source Syndicates** — SIRE project: community-owned AI vaults, users buy units, share wins/losses. Multi-agent systems with transparent model weights. Compare our variable importance against theirs.

## VARIABLE CROWDSOURCE STATUS (S209)
- 6 models queried: GPT-4.1, Grok, GPT-5.1, DeepSeek, o4-mini, Kimi K2.5
- Combined results: C:\tmp\variables_combined.txt
- Kimi independently validated Peter's variables (3pt variance, minutes load, lookahead)
- Peter's insight: these aren't gut — they're math/analytical pattern recognition
- NEXT: Build master consensus list, Peter reviews, lock Step 3

## KALSHI — GAME-LEVEL PREDICTION MARKET (Chinny find)
- **CFTC-regulated** — cleaner pricing than Polymarket
- **Game-level spread contracts:** KXNCAAMBSPREAD, KXNBA1HSPREAD, KXNFLSPREAD
- **Multiple lines per game** — Duke wins by 1.5, 4.5, 7.5, 10.5 = probability CURVE
- **Also has:** 1H spreads, 3Q winner, race to 10, 3-game ML combos
- **API:** Free to read (auth for trading). Endpoint: api.elections.kalshi.com/trade-api/v2/
- **USE CASE:** Step 7 in pipeline. Compare our model probability against Kalshi's market-implied probability. Divergence = edge signal.
- **Series tickers:** KXNCAAMBSPREAD, KXNBA1HSPREAD, KXNCAAMB1HSPREAD, KXNBA3QWINNER

## THE CREW (Edge Crew Inner Circle)
- **Chinny** — Thinks at the system level. Found Kalshi game-level markets. Give opportunities.
- **DJ** — Same mindset. Sees the architecture, not just the picks.
- Hard to find people who think like Peter. These two do.

## MISSING VARIABLE (Peter)
- **Momentum Decay** — Teams coming off emotional wins (rivalry, buzzer-beaters) underperform next game. Not in box scores. Behavioral economics.

## MAC MINI SPEC (TODO)
- Alyssa Spawn for personal research, music, deep dives
- Async work: Polymarket/Kalshi integration research, variable backtesting, model comparison
- Personal stuff stays there, work stays on ThinkStation
- Three machine architecture: ThinkStation (work) + Mac Mini (research) + Razer (support)

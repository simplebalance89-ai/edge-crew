# Kimi Profiler — S224 Build + Test Results

**Date:** March 23, 2026 | **Session:** S224 | **Status:** LIVE ON PROD

---

## What It Is

Kimi K2.5 runs as a **profiler** — separate from the main grading engine. It scores every game on 4 dimensions and attaches a scouting report to the card. It does NOT kill games, does NOT override grades. It's bonus context — a second lens.

---

## The 4 Dimensions (0-10 each)

| Dimension | What It Measures |
|-----------|-----------------|
| Tactical DNA | Style clash, pace mismatch, offensive/defensive identity |
| H2H Context | Historical patterns, tournament history, psychological edge |
| Structural Edge | Rest, travel, venue, momentum, seeding/ranking |
| Market Signal | Line movement, public vs sharp money, trap indicators |

**Profile Score** = average of 4 dimensions
**Profile Tag** = Kimi's one-line label (e.g., "SHARP CONTRARIAN", "MOMENTUM MISMATCH")

---

## Where It Runs in the Pipeline

```
Step 1: Data Grab (odds, injuries, form)
Step 2: Crowdsource (Llama 4, Grok 4 Fast, DeepSeek V3.2)
  ↓
Step 2.5: KIMI PROFILER ← HERE (scores ALL games, no kills)
  ↓
Step 3: Deep Thinker (DeepSeek-R1 on B+ games only — sees Kimi profile in prompt)
Step 4-6: Post-processing → Consensus → Peter Decides
```

Kimi profile feeds INTO the thinker prompt as "KIMI PROFILER SCOUTING REPORT" — bonus context, not override.

---

## API Configuration

| Setting | Value |
|---------|-------|
| Provider | Moonshot Direct (NOT Azure) |
| Endpoint | `https://api.moonshot.ai/v1` |
| Model | `kimi-k2.5` |
| Context | 262K tokens |
| Temperature | Fixed (cannot be modified) |
| Thinking | Disabled (profiler doesn't need deep reasoning) |
| Response Format | `json_object` |
| Max Completion Tokens | 8000 (prod) / 16000 (manual batch) |
| Timeout | 180s |

**Key gotchas discovered during build:**
- Endpoint is `.ai` NOT `.cn`
- Temperature cannot be set — omit it entirely
- Use `max_completion_tokens` NOT `max_tokens`
- Must pass `thinking` and `response_format` via `extra_body`
- Reasoning model — if thinking is enabled, burns tokens on internal reasoning and may return empty content

---

## How to Run Manually (Bypass Server)

For games not yet in the Odds API (e.g., Thursday/Friday NCAA slate), run Kimi directly:

```python
from openai import OpenAI
import json

client = OpenAI(
    base_url='https://api.moonshot.ai/v1',
    api_key='MOONSHOT_API_KEY',
    timeout=180,
)

games = """1. Illinois +3.5 @ Houston -3.5 | ML: ILL +140 / HOU -160 | O/U 140 | Public: 30% ILL / 70% HOU
2. St. Johns +6.5 @ Duke -6.5 | ML: SJU +234 / DUK -285 | O/U 142 | Public: 69% SJU / 31% DUK"""

prompt = f"""You are an elite NCAAB game profiler. Score each game on 4 dimensions (0-10).

DIMENSIONS:
1. Tactical DNA (style clash, pace, offensive/defensive identity)
2. H2H Context (history, tournament history, psychological edge)
3. Structural Edge (rest, travel, venue, momentum, seeding)
4. Market Signal (line movement, public vs sharp, trap indicators)

GAMES:
{games}

Return ONLY valid JSON:
{{"profiles": [{{"matchup": "AWAY @ HOME", "tactical_dna": {{"score": 7, "note": "..."}}, "h2h_context": {{"score": 5, "note": "..."}}, "structural_edge": {{"score": 8, "note": "..."}}, "market_signal": {{"score": 6, "note": "..."}}, "profile_score": 6.5, "profile_tag": "TAG", "profile_summary": "summary"}}]}}"""

resp = client.chat.completions.create(
    model='kimi-k2.5',
    messages=[{'role': 'user', 'content': prompt}],
    max_completion_tokens=8000,
    extra_body={
        'thinking': {'type': 'disabled'},
        'response_format': {'type': 'json_object'},
    },
)
data = json.loads(resp.choices[0].message.content)
for p in data['profiles']:
    print(f"{p['matchup']} — {p['profile_score']}/10 — {p['profile_tag']}")
```

**Rate limit note:** Kimi K2.5 on $5 Moonshot plan can get overloaded. Max ~5 games per batch. Wait 10-15s between batches.

---

## S224 Test Results

### NBA (March 24, 2026) — Via Prod Pipeline

| Game | Profile | Tag | Top Signal |
|------|---------|-----|------------|
| SAC @ CHA | 5.0/10 | DATA VOID + TRAP LINE | -17 spread screams trap |
| NOP @ NYK | 5.8/10 | ELITE CLASH + HEAVY SPREAD | Two surging teams, no structural edge |
| **ORL @ CLE** | **6.8/10** | **MOMENTUM MISMATCH + DEFENSIVE COLLAPSE** | Magic 0-5 freefall vs Cavs peak form |
| DEN @ PHX | 5.5/10 | RIVALRY GAME + ROAD VULNERABILITY | Nuggets road woes create Suns opportunity |

### NCAAB Sweet 16 / NIT — Via Manual Kimi Call

| Game | Profile | Tag | Top Signal |
|------|---------|-----|------------|
| **Tennessee @ Iowa State** | **8.0/10** | **SHARP CONTRARIAN** | Reverse line movement -5.5→-4, sharp TEN money |
| **Illinois @ Houston** | **7.0/10** | **SHARP CONTRARIAN** | 70% public HOU, sharps absorbing on ILL +140 |
| Alabama @ Michigan | 6.3/10 | CHAOS OVER | 174.5 total, pace-and-space fireworks |
| St. John's @ Duke | 6.25/10 | FADE PUBLIC DOG | 69% public SJU, sharps backing DUK cover |
| Michigan State @ Connecticut | 6.0/10 | COACHING LEGACY | Coin-flip, coaching adjustments decide it |
| Illinois State @ Dayton | 5.0/10 | HOME CHALK BORE | Dayton's NIT fortress, no sharp signal |

### Detailed Profiles

**Tennessee @ Iowa State (8.0/10 — HIGHEST ON SLATE)**
- Tactical DNA: 9/10 — Barnes' pack-line defense vs Otzelberger's scrambling Hilton havoc
- H2H Context: 7/10 — Tennessee's Sweet 16 heartbreak fuel vs ISU's home redemption arc
- Structural Edge: 8/10 — Hilton Coliseum 81% win rate but Tennessee's veteran core unfazed; Vols' extra day rest
- Market Signal: 8/10 — 69% public on ISU yet line dropped from -5.5 to -4, classic reverse line movement = sharp Tennessee money

**Illinois @ Houston (7.0/10)**
- Tactical DNA: 8/10 — Sampson's half-court defense vs Underwood's foul-drawing offense, pace tug-of-war
- H2H Context: 4/10 — No recent history, Houston carries Sweet 16 baggage
- Structural Edge: 7/10 — Houston's Dallas proximity = quasi-home court, Illinois on 48-hour turnaround from First Four
- Market Signal: 9/10 — Heavy 70% public on Houston, line stable at -3.5, reverse line movement toward Illinois ML

**ORL @ CLE (6.8/10)**
- Tactical DNA: 7.0/10 — Cavs' offensive explosion vs Magic's defensive identity in crisis
- H2H Context: 7.5/10 — Magic 0-5 freefall vs Cavs 3-game win streak
- Structural Edge: 7.0/10 — Psychological edge massive for Cleveland
- Market Signal: 5.5/10 — -10 line reflects ORL dysfunction, potential value trap if injuries hidden

---

## What Changed in S224

| Change | Detail |
|--------|--------|
| grok-3 → grok-4-fast-reasoning | All 4 locations (default, fallback, both pools). 1K→50K TPM |
| Dead models removed | Mistral (429), Cohere (unknown_model), Qwen (8K context) |
| Kimi Profiler added | Step 2.5 in pipeline, Moonshot direct API |
| Crowdsource pool cleaned | Light: Llama 4 + Grok 4 + DeepSeek V3.2. Legacy: + Kimi K2.5 |
| Azure upgraded | Peter moved from S0 free → pay-as-you-go |

---

*Built by Pietra — S224 | March 23, 2026*

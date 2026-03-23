"""
STAGE 3: TEAM DNA MATCH — Tactical compatibility analysis.

THE CHALK KILLER: AI sees ONLY tactical data. No odds, no names, no results.
This forces the model to evaluate style matchups objectively.

PASS: Score >5.0, clear style mismatch creates edge
KILL: No tactical angle, styles neutralize

AI Exposure (CONTROLLED):
  INCLUDE: formation, press intensity (PPDA), buildup quality, pace
  EXCLUDE: odds, recent results, league table, public sentiment, team names
"""

import json
import logging
import os
from typing import Dict, List, Optional

from engines.stage_models import HurdleGame, StageResult, Verdict

logger = logging.getLogger("edge-crew")

PASS_THRESHOLD = 5.0

# Sport-specific tactical factors that create edge
TACTICAL_EDGES = {
    "nba": {
        "pace_mismatch": "Fast pace team vs slow pace — tempo control edge",
        "three_pt_volume": "High 3PT volume vs weak perimeter D",
        "paint_dominance": "Paint scoring team vs undersized frontcourt",
        "transition": "Fast break team vs poor transition D",
        "defensive_scheme": "Switch-heavy D vs isolation scorer",
    },
    "nhl": {
        "forecheck_pressure": "Heavy forecheck vs weak breakout",
        "power_play_vs_pk": "Elite PP vs bottom PK",
        "shot_generation": "High xGF team vs leaky defense",
        "goalie_style": "Stand-up goalie vs high-danger team",
        "pace_control": "Possession team vs run-and-gun",
    },
    "soccer": {
        "press_vs_buildup": "High press team vs slow buildup — press trap",
        "wing_play_vs_narrow": "Wide attack vs narrow defense",
        "set_pieces": "Set piece team vs poor aerial defense",
        "counter_attack": "Counter team vs high defensive line",
        "possession_vs_direct": "Possession team vs direct play",
    },
    "ncaab": {
        "tempo_mismatch": "Fast tempo vs slow tempo — who controls pace",
        "three_pt_variance": "High 3PT team variance — boom or bust",
        "rebounding_edge": "Dominant boards vs weak rebounding",
        "turnover_forcing": "Press team vs turnover-prone",
        "free_throw_rate": "Foul-drawing team vs foul-prone D",
    },
}


async def run_stage3(
    games: List[HurdleGame],
    ai_caller=None,
) -> List[HurdleGame]:
    """
    Run Team DNA analysis on all games.

    Args:
        games: Survivors from Stage 2
        ai_caller: Optional async function(prompt) -> str for AI analysis.
                   If None, uses math-only fallback.

    Returns: List of games that pass (KILL verdicts are filtered out)
    """
    passed = []
    killed = 0

    for game in games:
        if ai_caller:
            result = await _evaluate_with_ai(game, ai_caller)
        else:
            result = _evaluate_math_only(game)

        game.stage_results.append(result)

        if result.verdict in (Verdict.PASS, Verdict.DEGRADE):
            passed.append(game)
        else:
            killed += 1
            logger.info(
                f"[STAGE 3] KILL {game.away_team} @ {game.home_team}: {result.notes}"
            )

    logger.info(f"[STAGE 3] {len(passed)} passed, {killed} killed")
    return passed


async def _evaluate_with_ai(game: HurdleGame, ai_caller) -> StageResult:
    """Use AI with controlled (blinded) prompt for tactical analysis."""
    prompt = _build_controlled_prompt(game)

    try:
        response = await ai_caller(prompt)
        score, notes = _parse_ai_response(response)
    except Exception as e:
        logger.warning(f"[STAGE 3] AI call failed for {game.game_id}: {e} — falling back to math")
        return _evaluate_math_only(game)

    verdict = Verdict.PASS if score >= PASS_THRESHOLD else Verdict.KILL

    return StageResult(
        stage=3,
        name="Team DNA Match",
        game_id=game.game_id,
        score=round(score, 1),
        threshold=PASS_THRESHOLD,
        verdict=verdict,
        confidence=0.65,
        next_stage=4 if verdict == Verdict.PASS else None,
        factors={"ai_score": score, "ai_notes": notes, "method": "ai_blinded"},
        notes=notes or f"Tactical score: {score:.1f}",
    )


def _evaluate_math_only(game: HurdleGame) -> StageResult:
    """
    Math-only fallback when AI is unavailable.
    Uses metadata fields if available, otherwise passes with neutral score.
    """
    score = 5.0  # Neutral — don't kill without information
    factors = {}
    notes_parts = []

    meta = game.metadata

    # Check for tactical data in metadata
    home_pace = meta.get("home_pace", 0)
    away_pace = meta.get("away_pace", 0)
    if home_pace and away_pace:
        pace_diff = abs(home_pace - away_pace)
        if pace_diff > 4:  # Significant pace mismatch
            score += 1.5
            factors["pace_mismatch"] = {"diff": pace_diff}
            notes_parts.append(f"Pace mismatch: {pace_diff:.1f}")

    # Offensive vs defensive ranking mismatch
    home_off = meta.get("home_off_rank", 0)
    away_def = meta.get("away_def_rank", 0)
    if home_off and away_def:
        if home_off <= 10 and away_def >= 20:  # Top 10 offense vs bottom 10 defense
            score += 2.0
            factors["off_vs_def"] = {"home_off_rank": home_off, "away_def_rank": away_def}
            notes_parts.append(f"Home top-{home_off} O vs away bottom-{30-away_def} D")

    # If we have no tactical data at all, pass with neutral score
    # (don't kill games just because we lack tactical data)
    if not notes_parts:
        notes_parts.append("No tactical data — passing with neutral score")
        factors["method"] = "no_data_passthrough"

    verdict = Verdict.PASS if score >= PASS_THRESHOLD else Verdict.KILL

    return StageResult(
        stage=3,
        name="Team DNA Match",
        game_id=game.game_id,
        score=round(score, 1),
        threshold=PASS_THRESHOLD,
        verdict=verdict,
        confidence=0.5 if not factors.get("method") == "no_data_passthrough" else 0.3,
        next_stage=4 if verdict == Verdict.PASS else None,
        factors=factors,
        notes="; ".join(notes_parts),
    )


def _build_controlled_prompt(game: HurdleGame) -> str:
    """
    Build a BLINDED prompt for tactical analysis.

    THE KEY INNOVATION: AI sees tactical data ONLY.
    Team names are replaced with "Team A" and "Team B".
    No odds, no standings, no recent results.
    """
    sport = game.sport
    edges = TACTICAL_EDGES.get(sport, TACTICAL_EDGES.get("nba", {}))
    meta = game.metadata

    return f"""STAGE 3: TACTICAL MATCHUP ANALYSIS — CONTROLLED CONTEXT

You are analyzing a {sport.upper()} game for tactical edge ONLY.

*** CRITICAL: You do NOT know and must NOT consider ***
- Team names (blinded as Team A vs Team B)
- Current odds or betting lines
- Recent results or win/loss record
- League standings or playoff position
- Public sentiment or media narratives

*** DATA PROVIDED ***

TEAM A (Home):
- Formation/System: {meta.get('home_formation', 'Standard')}
- Offensive Rating: {meta.get('home_off_rank', 'N/A')}
- Defensive Rating: {meta.get('home_def_rank', 'N/A')}
- Pace: {meta.get('home_pace', 'N/A')}
- Press Intensity (PPDA): {meta.get('home_ppda', 'N/A')}
- Buildup Style: {meta.get('home_buildup', 'N/A')}
- Three-Point Rate: {meta.get('home_3pt_rate', 'N/A')}

TEAM B (Away):
- Formation/System: {meta.get('away_formation', 'Standard')}
- Offensive Rating: {meta.get('away_off_rank', 'N/A')}
- Defensive Rating: {meta.get('away_def_rank', 'N/A')}
- Pace: {meta.get('away_pace', 'N/A')}
- Press Intensity (PPDA): {meta.get('away_ppda', 'N/A')}
- Buildup Style: {meta.get('away_buildup', 'N/A')}
- Three-Point Rate: {meta.get('away_3pt_rate', 'N/A')}

TACTICAL EDGES TO EVALUATE:
{chr(10).join(f'- {k}: {v}' for k, v in edges.items())}

*** YOUR TASK ***
1. Score this matchup 0-10 for tactical edge (does one side have a clear style advantage?)
2. Identify the specific tactical mismatch (if any)
3. State which side benefits (Team A or Team B)

Respond in JSON:
{{"score": 7.2, "edge": "Team A's high press exploits Team B's slow buildup", "favors": "A"}}
"""


def _parse_ai_response(response: str) -> tuple:
    """Parse AI response to extract score and notes."""
    # Try JSON parse first
    try:
        # Find JSON in response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            score = float(data.get("score", 5.0))
            edge = data.get("edge", "")
            favors = data.get("favors", "")
            notes = f"{edge} (favors {favors})" if edge else f"Score: {score}"
            return min(max(score, 0), 10), notes
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: look for a number
    import re
    numbers = re.findall(r'(\d+\.?\d*)\s*/\s*10', response)
    if numbers:
        score = float(numbers[0])
        return min(max(score, 0), 10), response[:100]

    # Default neutral
    return 5.0, "Could not parse AI response — neutral score"

"""
STAGE 8: FINAL GRADE & CROWDSOURCE TRIGGER

Synthesizes all stage scores into a final composite grade.
Determines bet type (ML vs Spread vs Total), sizing, and
whether to trigger crowdsource for borderline cases.

Auto-Crowdsource If:
  - Model disagreement >= 1 full grade (Stage 3 vs Stage 6)
  - EV borderline (2.0-3.0%)
  - Structural >7.0 but Market <4.0 (divergence)

Skip Crowdsource If:
  - Models agree within 0.3 STD
  - EV >4%, all flags green
  - Peter overrides with conviction

Output: Top 5 by (grade x EV x confidence), with bet type recommendation.
"""

import logging
from typing import Dict, List

from engines.stage_models import HurdleGame, StageResult, Verdict

logger = logging.getLogger("edge-crew")

# Stage weights for composite score
STAGE_WEIGHTS = {
    0: 0.05,   # Morning Filter (binary — already filtered)
    1: 0.00,   # Fixture (binary — already filtered)
    2: 0.15,   # Origin Check (injuries matter)
    3: 0.10,   # Team DNA (tactical)
    4: 0.05,   # H2H Context
    5: 0.25,   # Structural Edge (the core)
    6: 0.25,   # Market Validation (the money)
    7: 0.15,   # Lineup Lock (confirmation)
}

# Grade thresholds
GRADE_MAP = [
    (9.0, "A+"), (8.0, "A"), (7.0, "A-"),
    (6.5, "B+"), (6.0, "B"), (5.5, "B-"),
    (5.0, "C+"), (4.0, "C"), (0.0, "D"),
]

# Sizing based on grade
SIZING_MAP = {
    "A+": "MAX",     # 3 units
    "A": "LARGE",    # 2 units
    "A-": "LARGE",   # 2 units
    "B+": "MEDIUM",  # 1.5 units
    "B": "STANDARD", # 1 unit
    "B-": "SMALL",   # 0.5 units
    "C+": "SMALL",
    "C": "PASS",
    "D": "PASS",
}


def run_stage8(games: List[HurdleGame], dynamic_weights: Dict = None) -> List[Dict]:
    """
    Run final grading on all survivors.
    Returns sorted list of final picks (top 5).
    """
    results = []

    for game in games:
        result = _grade_game(game, dynamic_weights or {})
        game.stage_results.append(result)
        if result.verdict == Verdict.PASS:
            results.append(_build_final_output(game, result))

    # Sort by composite score descending, take top 5
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    top = results[:5]

    logger.info(
        f"[STAGE 8] {len(results)} graded, top {len(top)} selected"
    )
    return top


def _grade_game(game: HurdleGame, dynamic_weights: Dict) -> StageResult:
    """Calculate final composite grade for a game."""
    factors = {}

    # Calculate weighted composite from stage chain
    composite = 0.0
    stage_scores = {}

    for sr in game.stage_results:
        weight = STAGE_WEIGHTS.get(sr.stage, 0.0)
        weighted = sr.score * weight
        composite += weighted
        stage_scores[f"stage_{sr.stage}"] = {
            "score": sr.score,
            "weight": weight,
            "weighted": round(weighted, 2),
            "name": sr.name,
        }

    # Apply degrade factor
    if game.degraded:
        composite *= game.degrade_factor
        factors["degraded"] = True
        factors["degrade_factor"] = game.degrade_factor

    composite = round(min(10.0, max(0.0, composite)), 1)

    # Letter grade
    grade = "D"
    for threshold, g in GRADE_MAP:
        if composite >= threshold:
            grade = g
            break

    # Sizing
    sizing = SIZING_MAP.get(grade, "PASS")

    # Bet type recommendation (ML vs Spread)
    bet_type = _recommend_bet_type(game)

    # Crowdsource trigger check
    crowdsource, crowd_reason = _check_crowdsource_trigger(game, composite)

    factors["stage_scores"] = stage_scores
    factors["composite"] = composite
    factors["grade"] = grade
    factors["sizing"] = sizing
    factors["bet_type"] = bet_type
    factors["crowdsource"] = crowdsource
    factors["crowdsource_reason"] = crowd_reason

    # Final verdict — only PASS if grade is C+ or better
    if grade in ("D", "C"):
        verdict = Verdict.KILL
        notes = f"Grade {grade} ({composite}) — below threshold"
    else:
        verdict = Verdict.PASS
        notes = f"Grade {grade} ({composite}) | {sizing} | {bet_type['type']} {bet_type.get('line', '')}"

    return StageResult(
        stage=8,
        name="Final Grade",
        game_id=game.game_id,
        score=composite,
        threshold=5.0,
        verdict=verdict,
        confidence=0.8 if grade.startswith("A") else 0.6,
        next_stage=None,
        factors=factors,
        notes=notes,
    )


def _recommend_bet_type(game: HurdleGame) -> Dict:
    """
    Recommend ML vs Spread vs Total based on edge characteristics.

    Peter's insight: teams winning straight up but not covering spread.
    If edge is "this team wins" but spread is tight, recommend ML.
    """
    rec = {"type": "SPREAD", "line": game.home_spread, "reasoning": ""}

    spread = game.home_spread
    if spread is None:
        # No spread available — ML only
        return {"type": "ML", "line": game.home_ml, "reasoning": "No spread available"}

    abs_spread = abs(spread)

    # Check structural edge direction
    structural_score = 0
    market_score = 0
    for sr in game.stage_results:
        if sr.stage == 5:
            structural_score = sr.score
        if sr.stage == 6:
            market_score = sr.score

    # RULE: If spread is tight (1-4.5 pts) and structural edge is strong,
    # recommend ML instead — team wins but may not cover
    if 1.0 <= abs_spread <= 4.5 and structural_score >= 6.0:
        rec = {
            "type": "ML",
            "line": game.home_ml if spread < 0 else game.away_ml,
            "reasoning": f"Tight spread ({spread}) with strong structural edge — ML safer",
        }
    # RULE: Large spread (>10) — underdogs get free points, take spread
    elif abs_spread > 10:
        rec = {
            "type": "SPREAD",
            "line": spread,
            "reasoning": f"Large spread ({spread}) — book overreacting, spread has value",
        }
    # RULE: Medium spread with market confirmation — take spread
    elif 4.5 < abs_spread <= 10 and market_score >= 6.0:
        rec = {
            "type": "SPREAD",
            "line": spread,
            "reasoning": f"Medium spread ({spread}) with market confirmation — spread play",
        }
    # Default: spread
    else:
        rec["reasoning"] = "Standard spread play"

    return rec


def _check_crowdsource_trigger(
    game: HurdleGame, composite: float
) -> tuple:
    """Check if this game should trigger crowdsource consensus."""
    # Get stage scores
    tactical_score = 0
    market_score = 0
    structural_score = 0

    for sr in game.stage_results:
        if sr.stage == 3:
            tactical_score = sr.score
        elif sr.stage == 5:
            structural_score = sr.score
        elif sr.stage == 6:
            market_score = sr.score

    # Trigger 1: Tactical vs Market disagreement (>2 points)
    disagreement = abs(tactical_score - market_score)
    if disagreement > 2.0:
        return True, f"Tactical ({tactical_score}) vs Market ({market_score}) disagreement = {disagreement:.1f}"

    # Trigger 2: Structural strong but Market weak (or vice versa)
    if structural_score > 7.0 and market_score < 4.0:
        return True, f"Structural strong ({structural_score}) but market weak ({market_score})"
    if market_score > 7.0 and structural_score < 4.0:
        return True, f"Market strong ({market_score}) but structural weak ({structural_score})"

    # Trigger 3: Borderline composite (5.0-6.0)
    if 5.0 <= composite <= 6.0:
        return True, f"Borderline composite ({composite}) — need consensus"

    # No trigger
    return False, ""


def _build_final_output(game: HurdleGame, final_result: StageResult) -> Dict:
    """Build the final output dict for a graded game."""
    factors = final_result.factors
    bet = factors.get("bet_type", {})

    return {
        "game_id": game.game_id,
        "sport": game.sport,
        "matchup": f"{game.away_team} @ {game.home_team}",
        "home_team": game.home_team,
        "away_team": game.away_team,
        "commence_time": game.commence_time,

        # The pick
        "grade": factors.get("grade", "D"),
        "composite_score": factors.get("composite", 0),
        "sizing": factors.get("sizing", "PASS"),
        "bet_type": bet.get("type", "SPREAD"),
        "bet_line": bet.get("line"),
        "bet_reasoning": bet.get("reasoning", ""),

        # Edge summary
        "edge_side": _determine_edge_side(game),
        "degraded": game.degraded,
        "degrade_factor": game.degrade_factor,

        # Crowdsource
        "crowdsource_flag": factors.get("crowdsource", False),
        "crowdsource_reason": factors.get("crowdsource_reason", ""),

        # Stage chain trace
        "stage_chain": [
            {
                "stage": sr.stage,
                "name": sr.name,
                "score": sr.score,
                "verdict": sr.verdict.value,
            }
            for sr in game.stage_results
        ],

        # Stage breakdown
        "stage_scores": factors.get("stage_scores", {}),
    }


def _determine_edge_side(game: HurdleGame) -> str:
    """Determine which side the overall edge favors."""
    # Check Stage 2 (injury edge)
    for sr in game.stage_results:
        if sr.stage == 2 and "edge_favors" in sr.factors:
            edge = sr.factors["edge_favors"]
            if edge != "neutral":
                return edge

    # Check Stage 5 (structural edge)
    for sr in game.stage_results:
        if sr.stage == 5 and "alignment" in sr.factors:
            return sr.factors["alignment"].get("dominant_edge", "unclear")

    # Default: favor home team if they have better spread
    if game.home_spread is not None and game.home_spread < 0:
        return game.home_team
    return "unclear"

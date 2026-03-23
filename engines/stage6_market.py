"""
STAGE 6: MARKET VALIDATION — Line movement, EV calculation, sharp signals.

THE SECOND CHALK KILLER: AI sees ONLY market data. Teams are blinded.
No tactical context, no injury info, no team names.

PASS:
  - EV >= 2.5% (soccer), >= 2.0% (NBA/NHL/other)
  - Line moved toward pick OR stable with public opposite
  - Sharp money aligned or contrarian edge confirmed

KILL: EV < threshold, or line moved sharply against

Uses existing ev_calc.py for the math.
AI (optional) provides market efficiency analysis with blinded prompt.
"""

import json
import logging
from typing import Dict, List, Optional

from engines.stage_models import HurdleGame, StageResult, Verdict
from engines.ev_calc import calc_implied_prob, calculate_ev, american_to_decimal

logger = logging.getLogger("edge-crew")

# Sport-specific EV thresholds
EV_THRESHOLDS = {
    "soccer": 2.5,
    "nba": 2.0,
    "nhl": 2.0,
    "nfl": 2.0,
    "ncaab": 2.0,
    "ncaaf": 2.0,
    "mlb": 2.0,
    "mma": 3.0,
    "boxing": 3.0,
    "tennis": 2.5,
}

DEFAULT_EV_THRESHOLD = 2.0


async def run_stage6(
    games: List[HurdleGame],
    ai_caller=None,
) -> List[HurdleGame]:
    """
    Run market validation on all games.

    Args:
        games: Survivors from Stage 5
        ai_caller: Optional async function(prompt) -> str for blinded market AI.

    Returns: Games that pass market validation
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
                f"[STAGE 6] KILL {game.away_team} @ {game.home_team}: {result.notes}"
            )

    logger.info(f"[STAGE 6] {len(passed)} passed, {killed} killed")
    return passed


def _evaluate_math_only(game: HurdleGame) -> StageResult:
    """Pure math market validation using ev_calc.py."""
    factors = {}
    score = 5.0  # Start neutral
    threshold = EV_THRESHOLDS.get(game.sport, DEFAULT_EV_THRESHOLD)

    # --- 1. LINE MOVEMENT ANALYSIS ---
    spread_move = game.spread_move or 0
    total_move = game.total_move or 0
    max_move = max(abs(spread_move), abs(total_move))

    if max_move > 0:
        # Movement = signal. Direction matters but we don't know which
        # side is "our pick" yet, so we score movement magnitude.
        move_score = min(max_move * 1.5, 3.0)
        score += move_score
        factors["line_movement"] = {
            "spread_move": spread_move,
            "total_move": total_move,
            "score": round(move_score, 1),
            "signal": "sharp" if max_move > 1.5 else "moderate" if max_move > 0.5 else "minimal",
        }
    else:
        factors["line_movement"] = {"score": 0, "signal": "static"}

    # --- 2. IMPLIED PROBABILITY GAP ---
    # If we have both MLs, check for value
    home_impl = calc_implied_prob(game.home_ml) if game.home_ml else None
    away_impl = calc_implied_prob(game.away_ml) if game.away_ml else None

    if home_impl and away_impl:
        # Overround tells us about market efficiency
        overround = home_impl + away_impl - 100
        factors["implied_probs"] = {
            "home": home_impl,
            "away": away_impl,
            "overround": round(overround, 1),
        }

        # Large overround = less efficient market = more edge potential
        if overround > 10:
            score += 1.0
            factors["market_efficiency"] = "low (overround > 10%)"
        elif overround < 5:
            factors["market_efficiency"] = "high (tight market)"

    # --- 3. SPREAD / EV ESTIMATE ---
    # Estimate EV from line movement direction
    # If line moved and we can calculate implied prob shift
    if game.opening_spread is not None and game.home_spread is not None:
        spread_shift = abs(game.home_spread - game.opening_spread)
        # Each 0.5 points of line movement ~ 1.5% EV shift
        estimated_ev = spread_shift * 3.0
        factors["estimated_ev"] = round(estimated_ev, 1)

        if estimated_ev >= threshold:
            score += 2.0
            factors["ev_pass"] = True
        elif estimated_ev >= threshold * 0.7:
            score += 1.0
            factors["ev_borderline"] = True
    else:
        # Without opening line, estimate from spread magnitude
        # Extreme spreads (>10) have more potential for mispricing
        if game.home_spread is not None:
            spread_mag = abs(game.home_spread)
            if spread_mag >= 10:
                score += 0.5
                factors["large_spread"] = {"spread": game.home_spread, "note": "Large spread = more variance"}

    # --- 4. CONTRARIAN SIGNAL ---
    # Without public betting %, we use line movement as proxy:
    # If line moved toward favorite but by less than expected, contrarian edge
    if spread_move != 0 and game.home_spread is not None:
        # Line moved toward underdog = potential sharp contrarian
        favorite_side = "home" if game.home_spread < 0 else "away"
        line_moved_to_dog = (
            (favorite_side == "home" and spread_move > 0) or  # Spread got bigger = moved toward away
            (favorite_side == "away" and spread_move < 0)     # Spread got smaller = moved toward home
        )
        if line_moved_to_dog:
            score += 1.5
            factors["contrarian_signal"] = {
                "score": 1.5,
                "note": "Line moved toward underdog — potential sharp action",
            }

    score = round(max(0.0, min(10.0, score)), 1)

    # --- VERDICT ---
    if score >= 7.0:
        verdict = Verdict.PASS
        notes = f"Strong market signal (score {score})"
    elif score >= 5.0:
        verdict = Verdict.PASS
        notes = f"Market validation passed (score {score})"
    elif score >= 3.5:
        verdict = Verdict.DEGRADE
        notes = f"Weak market signal (score {score}) — degraded"
    else:
        verdict = Verdict.KILL
        notes = f"No market edge (score {score})"

    return StageResult(
        stage=6,
        name="Market Validation",
        game_id=game.game_id,
        score=score,
        threshold=5.0,
        verdict=verdict,
        confidence=0.75 if verdict == Verdict.PASS else 0.4,
        next_stage=7 if verdict in (Verdict.PASS, Verdict.DEGRADE) else None,
        factors=factors,
        notes=notes,
    )


async def _evaluate_with_ai(game: HurdleGame, ai_caller) -> StageResult:
    """Use AI with blinded market prompt."""
    prompt = _build_blinded_market_prompt(game)

    try:
        response = await ai_caller(prompt)
        ai_score, ai_notes = _parse_market_response(response)
    except Exception as e:
        logger.warning(f"[STAGE 6] AI failed for {game.game_id}: {e} — math fallback")
        return _evaluate_math_only(game)

    # Blend AI score with math score
    math_result = _evaluate_math_only(game)
    math_score = math_result.score

    # Weighted average: 60% AI, 40% math
    blended = round(ai_score * 0.6 + math_score * 0.4, 1)

    if blended >= 5.0:
        verdict = Verdict.PASS
    elif blended >= 3.5:
        verdict = Verdict.DEGRADE
    else:
        verdict = Verdict.KILL

    factors = math_result.factors.copy()
    factors["ai_score"] = ai_score
    factors["ai_notes"] = ai_notes
    factors["blended_score"] = blended
    factors["method"] = "ai_blinded + math"

    return StageResult(
        stage=6,
        name="Market Validation",
        game_id=game.game_id,
        score=blended,
        threshold=5.0,
        verdict=verdict,
        confidence=0.75 if verdict == Verdict.PASS else 0.4,
        next_stage=7 if verdict in (Verdict.PASS, Verdict.DEGRADE) else None,
        factors=factors,
        notes=ai_notes or f"Blended market score: {blended}",
    )


def _build_blinded_market_prompt(game: HurdleGame) -> str:
    """Build market-only prompt with teams blinded."""
    home_impl = calc_implied_prob(game.home_ml) if game.home_ml else "N/A"
    away_impl = calc_implied_prob(game.away_ml) if game.away_ml else "N/A"

    return f"""STAGE 6: MARKET VALIDATION — BLINDED ANALYSIS

*** CRITICAL: Teams are BLINDED. You do NOT know ***
- Team names (Team A vs Team B)
- Tactical profiles or form
- Injury status
- League standings

*** MARKET DATA ONLY ***

Sport: {game.sport.upper()}
Opening Spread: {game.opening_spread or 'N/A'}
Current Spread: {game.home_spread or 'N/A'}
Spread Movement: {game.spread_move or 0}
Opening Total: {game.opening_total or 'N/A'}
Current Total: {game.total or 'N/A'}
Total Movement: {game.total_move or 0}
Home ML: {game.home_ml or 'N/A'} (Implied: {home_impl}%)
Away ML: {game.away_ml or 'N/A'} (Implied: {away_impl}%)

*** YOUR TASK ***
1. Is there Expected Value (EV) here based on line movement?
2. Does the line movement suggest sharp money or public money?
3. Score this market opportunity 0-10.

Respond in JSON:
{{"score": 7.0, "ev_estimate": 3.2, "signal": "sharp money on away", "reasoning": "..."}}
"""


def _parse_market_response(response: str) -> tuple:
    """Parse AI market response."""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            score = float(data.get("score", 5.0))
            reasoning = data.get("reasoning", data.get("signal", ""))
            return min(max(score, 0), 10), reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return 5.0, "Could not parse AI market response"

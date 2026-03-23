"""
STAGE 0: MORNING FILTER — Kill 70% of slate before expensive AI calls.

Pass criteria (ANY of):
  - Line movement >0.5 points overnight
  - Rest differential >= 2 days
  - B2B vs fresh team
  - AND no star injury uncertainty

Kill: Static lines, no structural edge, star confirmed out
Soft fail: Star questionable → retry at T-60min

Also includes the Dynamic Weight Engine that adjusts downstream
stage weights based on slate conditions.
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple

from engines.stage_models import HurdleGame, StageResult, Verdict

logger = logging.getLogger("edge-crew")

# --- Thresholds ---
PASS_THRESHOLD = 2.0
LINE_MOVE_MIN = 0.5
REST_DIFF_MIN = 2
STAR_PENALTY = 2.0


async def run_stage0(games: List[HurdleGame]) -> Dict:
    """
    Run the morning filter on the full slate.
    Returns dict with passed/killed/soft_fail lists + dynamic weights + slate profile.
    """
    passed = []
    killed = []
    soft_fail = []

    for game in games:
        result = _evaluate_game(game)
        game.stage_results.append(result)

        if result.verdict == Verdict.PASS:
            passed.append(game)
        elif result.verdict == Verdict.SOFT_FAIL:
            soft_fail.append(game)
        else:
            killed.append(game)

    # Analyze slate conditions for dynamic weighting
    slate_profile = _classify_slate(games)
    dynamic_weights = _calculate_dynamic_weights(games, slate_profile)

    total = len(games)
    kill_rate = len(killed) / total * 100 if total else 0

    logger.info(
        f"[STAGE 0] {len(passed)} passed, {len(killed)} killed, "
        f"{len(soft_fail)} soft_fail | Kill rate: {kill_rate:.0f}% | "
        f"Profile: {slate_profile}"
    )

    return {
        "passed": passed,
        "killed": killed,
        "soft_fail": soft_fail,
        "total": total,
        "kill_rate": round(kill_rate, 1),
        "slate_profile": slate_profile,
        "dynamic_weights": dynamic_weights,
    }


def _evaluate_game(game: HurdleGame) -> StageResult:
    """Score a single game through the morning filter."""
    score = 0.0
    factors = {}

    # --- 1. Line Movement ---
    spread_move = abs(game.spread_move) if game.spread_move else 0
    total_move = abs(game.total_move) if game.total_move else 0
    line_move = max(spread_move, total_move)

    if line_move > LINE_MOVE_MIN:
        pts = min(line_move * 2, 4.0)  # Cap at 4
        score += pts
        factors["line_movement"] = {
            "score": round(pts, 1),
            "spread_move": game.spread_move,
            "total_move": game.total_move,
        }

    # --- 2. Rest Differential ---
    if game.home_rest_days >= 0 and game.away_rest_days >= 0:
        rest_diff = abs(game.home_rest_days - game.away_rest_days)
        if rest_diff >= REST_DIFF_MIN:
            pts = min(rest_diff * 1.5, 4.0)  # Cap at 4
            score += pts
            factors["rest_differential"] = {
                "score": round(pts, 1),
                "home_rest": game.home_rest_days,
                "away_rest": game.away_rest_days,
                "diff": rest_diff,
            }

    # --- 3. B2B vs Fresh ---
    home_b2b_fresh = game.home_b2b and not game.away_b2b
    away_b2b_fresh = game.away_b2b and not game.home_b2b
    if home_b2b_fresh or away_b2b_fresh:
        score += 2.0
        b2b_team = game.home_team if home_b2b_fresh else game.away_team
        fresh_team = game.away_team if home_b2b_fresh else game.home_team
        factors["b2b_mismatch"] = {
            "score": 2.0,
            "b2b_team": b2b_team,
            "fresh_team": fresh_team,
        }

    # --- 4. Spread Magnitude (large spreads = market inefficiency potential) ---
    if game.home_spread is not None:
        abs_spread = abs(game.home_spread)
        if abs_spread >= 10:
            pts = 1.5
            score += pts
            factors["large_spread"] = {
                "score": pts,
                "spread": game.home_spread,
                "note": "Large spread = more variance, potential value",
            }
        elif abs_spread <= 3.0:
            # Tight spread = close game = potential ML value
            pts = 1.0
            score += pts
            factors["tight_spread"] = {
                "score": pts,
                "spread": game.home_spread,
                "note": "Tight spread = close game, ML angle",
            }

    # --- 5. Hot/Cold Streaks (bonus, not primary) ---
    for side, streak, team in [
        ("home", game.home_streak, game.home_team),
        ("away", game.away_streak, game.away_team),
    ]:
        if abs(streak) >= 4:
            bonus = 1.0
            score += bonus
            factors[f"{side}_streak"] = {
                "score": bonus,
                "team": team,
                "streak": streak,
            }

    # --- 5. Star Injury Penalty ---
    star_out_total = game.home_star_out + game.away_star_out
    star_q_total = game.home_star_questionable + game.away_star_questionable

    if star_out_total > 0:
        penalty = STAR_PENALTY * star_out_total
        score -= penalty
        factors["star_injuries_out"] = {
            "score": -penalty,
            "count": star_out_total,
        }

    # --- Verdict ---
    # Stages 0-3 are SCORING stages — accumulate data, never kill.
    # First hurdle is at Stage 4 where cumulative score decides.
    # Stage 0 only soft-fails on questionable stars (need more info).
    if star_q_total > 0:
        verdict = Verdict.SOFT_FAIL
        notes = f"Score {score:.1f} — {star_q_total} star(s) questionable, retry T-60min"
    else:
        verdict = Verdict.PASS
        notes = f"Score {score:.1f} — accumulated (no kill at Stage 0)"

    return StageResult(
        stage=0,
        name="Morning Filter",
        game_id=game.game_id,
        score=round(score, 1),
        threshold=PASS_THRESHOLD,
        verdict=verdict,
        confidence=0.6 if verdict == Verdict.PASS else 0.3,
        next_stage=1 if verdict == Verdict.PASS else None,
        factors=factors,
        notes=notes,
    )


# ================================================================
# DYNAMIC WEIGHT ENGINE
# ================================================================

# Base weights for downstream stages (default conditions)
BASE_WEIGHTS = {
    "line_movement": 9,
    "star_player_status": 10,
    "rest_advantage": 9,
    "tactical_matchup": 7,
    "h2h_season": 5,
    "structural_edge": 8,
    "market_validation": 9,
    "lineup_lock": 8,
    "form_composite": 7,
    "sharp_vs_public": 7,
    "depth_injuries": 6,
}


def _classify_slate(games: List[HurdleGame]) -> str:
    """Classify the day's slate into a profile type."""
    if not games:
        return "EMPTY"

    n = len(games)

    # Metrics
    avg_line_move = sum(
        max(abs(g.spread_move or 0), abs(g.total_move or 0)) for g in games
    ) / n
    star_injury_count = sum(g.home_star_out + g.away_star_out for g in games)
    b2b_count = sum(1 for g in games if g.home_b2b or g.away_b2b)
    b2b_pct = b2b_count / n

    volatile = avg_line_move > 1.5
    injury_heavy = star_injury_count > 3
    congested = b2b_pct > 0.4

    if volatile and injury_heavy:
        return "CHAOS"
    if volatile:
        return "VOLATILE"
    if congested:
        return "CONGESTED"
    if injury_heavy:
        return "INJURY_HEAVY"
    if avg_line_move < 0.3:
        return "QUIET"
    return "CLEAN"


def _calculate_dynamic_weights(
    games: List[HurdleGame], slate_profile: str
) -> Dict[str, int]:
    """Adjust stage weights based on slate conditions."""
    weights = BASE_WEIGHTS.copy()

    if slate_profile == "VOLATILE":
        weights["line_movement"] += 2
        weights["sharp_vs_public"] += 1
        weights["form_composite"] -= 2

    elif slate_profile == "INJURY_HEAVY":
        weights["star_player_status"] += 2
        weights["depth_injuries"] += 2
        weights["tactical_matchup"] -= 1

    elif slate_profile == "CONGESTED":
        weights["rest_advantage"] += 2
        weights["structural_edge"] += 1
        weights["form_composite"] -= 2

    elif slate_profile == "CHAOS":
        weights["star_player_status"] += 2
        weights["depth_injuries"] += 2
        weights["line_movement"] += 1
        weights["form_composite"] -= 3
        weights["tactical_matchup"] -= 1

    elif slate_profile == "QUIET":
        weights["structural_edge"] += 1
        weights["line_movement"] -= 2

    # Clamp all weights to 1-12
    return {k: max(1, min(12, v)) for k, v in weights.items()}


def format_stage0_report(result: Dict) -> str:
    """Human-readable summary for logging/Discord."""
    lines = [
        f"=== STAGE 0: MORNING FILTER ===",
        f"Slate: {result['total']} games | Profile: {result['slate_profile']}",
        f"Kill rate: {result['kill_rate']}%",
        f"",
        f"PASSED ({len(result['passed'])}):",
    ]
    for g in result["passed"]:
        r = g.stage_results[0]
        lines.append(f"  {g.away_team} @ {g.home_team} — Score: {r.score} | {', '.join(r.factors.keys())}")

    lines.append(f"")
    lines.append(f"KILLED ({len(result['killed'])}):")
    for g in result["killed"]:
        r = g.stage_results[0]
        lines.append(f"  {g.away_team} @ {g.home_team} — Score: {r.score} | {r.notes}")

    if result["soft_fail"]:
        lines.append(f"")
        lines.append(f"SOFT FAIL ({len(result['soft_fail'])}) — retry T-60min:")
        for g in result["soft_fail"]:
            r = g.stage_results[0]
            lines.append(f"  {g.away_team} @ {g.home_team} — {r.notes}")

    lines.append(f"")
    lines.append(f"Dynamic weights: {result['dynamic_weights']}")
    return "\n".join(lines)

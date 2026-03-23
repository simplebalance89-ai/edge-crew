"""
STAGE 7: LINEUP LOCK — Final confirmation before grading.

Checks confirmed lineups and last-minute changes.
Recalculates edge viability post-lineup.

PASS: 90%+ confirmed, no last-minute breaks
DEGRADE: Late changes reduce edge 20-30%
KILL: Key late out destroys edge
INCOMPLETE: Lineups not yet confirmed (hold for later)

For games where lineups aren't confirmed yet, the system
marks them INCOMPLETE rather than killing — they can be
re-evaluated when lineups drop (usually 30-60min pre-game).
"""

import logging
from typing import Dict, List, Tuple

from engines.stage_models import HurdleGame, StageResult, Verdict

logger = logging.getLogger("edge-crew")


async def run_stage7(
    games: List[HurdleGame],
    lineup_data: Dict = None,
) -> Tuple[List[HurdleGame], List[HurdleGame]]:
    """
    Run lineup lock on all games.

    Args:
        games: Survivors from Stage 6
        lineup_data: Optional dict of confirmed lineup info per game_id.
                     Format: {game_id: {"confirmed": bool, "late_changes": [...], "pct_confirmed": float}}

    Returns: (passed_or_degraded, incomplete)
             Incomplete games should be re-evaluated when lineups confirm.
    """
    passed = []
    incomplete = []
    killed = 0

    for game in games:
        game_lineup = (lineup_data or {}).get(game.game_id, {})
        result = _evaluate_lineup(game, game_lineup)
        game.stage_results.append(result)

        if result.verdict == Verdict.PASS:
            passed.append(game)
        elif result.verdict == Verdict.DEGRADE:
            game.degraded = True
            game.degrade_factor *= 0.75  # Late changes = bigger confidence hit
            passed.append(game)
        elif result.verdict == Verdict.INCOMPLETE:
            incomplete.append(game)
        else:
            killed += 1
            logger.info(
                f"[STAGE 7] KILL {game.away_team} @ {game.home_team}: {result.notes}"
            )

    logger.info(
        f"[STAGE 7] {len(passed)} passed, {len(incomplete)} incomplete, {killed} killed"
    )
    return passed, incomplete


def _evaluate_lineup(game: HurdleGame, lineup_info: Dict) -> StageResult:
    """Evaluate a single game's lineup confirmation status."""
    factors = {}

    confirmed = lineup_info.get("confirmed", False)
    pct_confirmed = lineup_info.get("pct_confirmed", 0.0)
    late_changes = lineup_info.get("late_changes", [])

    factors["confirmed"] = confirmed
    factors["pct_confirmed"] = pct_confirmed
    factors["late_changes_count"] = len(late_changes)

    # If no lineup data at all, check if we're close enough to game time
    # For now, pass with a note — Phase 5 will add real-time lineup monitoring
    if not lineup_info:
        # No lineup data — pass with reduced confidence
        # The system should not kill games just because lineups aren't posted yet
        return StageResult(
            stage=7,
            name="Lineup Lock",
            game_id=game.game_id,
            score=6.0,
            threshold=6.0,
            verdict=Verdict.PASS,
            confidence=0.5,  # Lower confidence without confirmed lineups
            next_stage=8,
            factors={"confirmed": False, "method": "no_lineup_data_passthrough"},
            notes="No lineup data available — passing with reduced confidence",
        )

    # Calculate impact of late changes
    total_impact = 0
    change_details = []
    for change in late_changes:
        impact = change.get("impact", 0)
        total_impact += impact
        change_details.append({
            "player": change.get("player", "Unknown"),
            "status": change.get("status", "unknown"),
            "impact": impact,
        })

    factors["total_late_impact"] = total_impact
    factors["changes"] = change_details

    # Score
    if confirmed and pct_confirmed >= 0.9:
        if total_impact == 0:
            score = 9.0
        elif total_impact <= 5:
            score = 7.0
        elif total_impact <= 15:
            score = 5.0
        else:
            score = 3.0
    elif pct_confirmed >= 0.7:
        score = 5.0
    else:
        score = 3.0

    # Check if late changes affect our edge
    # Look at Stage 2 result to see which side the injury edge favored
    edge_side = None
    for sr in game.stage_results:
        if sr.stage == 2 and "edge_favors" in sr.factors:
            edge_side = sr.factors["edge_favors"]
            break

    # If late changes affect the team we were betting ON (not against), degrade more
    for change in late_changes:
        change_team = (change.get("team", "") or "").lower()
        if edge_side and edge_side.lower() in change_team:
            score -= 2.0
            factors["edge_team_affected"] = True

    score = max(0.0, min(10.0, score))

    # Verdict
    if not confirmed and pct_confirmed < 0.5:
        verdict = Verdict.INCOMPLETE
        notes = f"Lineups {pct_confirmed*100:.0f}% confirmed — hold for update"
    elif total_impact > 15:
        verdict = Verdict.KILL
        notes = f"Late changes impact {total_impact}% — edge destroyed"
    elif total_impact > 5 or score < 6.0:
        verdict = Verdict.DEGRADE
        notes = f"Late changes (impact {total_impact}%) — edge reduced"
    else:
        verdict = Verdict.PASS
        notes = f"Lineups confirmed ({pct_confirmed*100:.0f}%), impact {total_impact}%"

    return StageResult(
        stage=7,
        name="Lineup Lock",
        game_id=game.game_id,
        score=round(score, 1),
        threshold=6.0,
        verdict=verdict,
        confidence=0.85 if verdict == Verdict.PASS else 0.5,
        next_stage=8 if verdict in (Verdict.PASS, Verdict.DEGRADE) else None,
        factors=factors,
        notes=notes,
    )

"""
STAGE 1: FIXTURE VALIDATION — Confirm market exists, game is playable.

PASS: Odds available, not postponed, liquidity sufficient (has spread + total)
KILL: Postponed, no market, or only moneyline (no spread/total = thin market)

This is a fast binary gate — no scoring, just pass/kill.
"""

import logging
from typing import List

from engines.stage_models import HurdleGame, StageResult, Verdict

logger = logging.getLogger("edge-crew")


def run_stage1(games: List[HurdleGame]) -> List[HurdleGame]:
    """
    Run fixture validation on all games.
    Returns only games that pass. Killed games get a StageResult appended.
    """
    passed = []
    killed = 0

    for game in games:
        result = _validate_fixture(game)
        game.stage_results.append(result)

        if result.verdict == Verdict.PASS:
            passed.append(game)
        else:
            killed += 1
            logger.info(
                f"[STAGE 1] KILL {game.away_team} @ {game.home_team}: {result.notes}"
            )

    logger.info(
        f"[STAGE 1] {len(passed)} passed, {killed} killed"
    )
    return passed


def _validate_fixture(game: HurdleGame) -> StageResult:
    """Validate a single game's market readiness."""
    factors = {}
    issues = []

    # 1. Has odds at all?
    has_ml = game.home_ml is not None and game.away_ml is not None
    has_spread = game.home_spread is not None
    has_total = game.total is not None
    factors["has_moneyline"] = has_ml
    factors["has_spread"] = has_spread
    factors["has_total"] = has_total

    if not has_ml:
        issues.append("no moneyline")

    # 2. Is game postponed/cancelled?
    status = (game.metadata.get("status", "") or "").lower()
    if status in ("postponed", "cancelled", "suspended"):
        issues.append(f"game {status}")
        factors["game_status"] = status

    # 3. Lines completeness — need at least ML + one of spread/total
    # For MMA/Boxing, ML-only is fine
    ml_only_sports = {"mma", "boxing"}
    if game.sport not in ml_only_sports:
        if has_ml and not has_spread and not has_total:
            issues.append("thin market (ML only, no spread/total)")
            factors["thin_market"] = True

    # 4. Commence time sanity — game should be in the future
    # (Past games shouldn't be in slate, but safety check)
    if game.commence_time:
        factors["commence_time"] = game.commence_time

    # Verdict
    if issues:
        verdict = Verdict.KILL
        score = 0.0
        notes = "Fixture invalid: " + "; ".join(issues)
    else:
        verdict = Verdict.PASS
        score = 10.0
        lines_desc = []
        if has_spread:
            lines_desc.append(f"spread {game.home_spread}")
        if has_total:
            lines_desc.append(f"total {game.total}")
        if has_ml:
            lines_desc.append("ML")
        notes = f"Market confirmed: {', '.join(lines_desc)}"

    return StageResult(
        stage=1,
        name="Fixture Validation",
        game_id=game.game_id,
        score=score,
        threshold=5.0,
        verdict=verdict,
        confidence=0.95 if verdict == Verdict.PASS else 0.0,
        next_stage=2 if verdict == Verdict.PASS else None,
        factors=factors,
        notes=notes,
    )

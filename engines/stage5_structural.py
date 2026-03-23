"""
STAGE 5: STRUCTURAL EDGE — Rest, travel, congestion, weather composite.

Pure math. No AI. Multiple structural factors scored and combined.

PASS: Score >5.0 with >=2 structural factors aligned
DEGRADE: Score 3.0-5.0 (weak edge, reduce confidence)
KILL: Score <3.0 or structural disadvantage on all fronts

Factors:
  - Rest differential: +/-3 days = +/-3 points
  - B2B penalty: Big penalty for team on B2B vs rested opponent
  - Fixture congestion: Games per 7 days
  - Streak momentum: Hot/cold streaks as structural indicator
  - EV pre-check: If estimated EV <2.0%, flag for review
"""

import logging
from typing import Dict, List

from engines.stage_models import HurdleGame, StageResult, Verdict

logger = logging.getLogger("edge-crew")

PASS_THRESHOLD = 5.0
DEGRADE_THRESHOLD = 3.0

# Factor max contributions (caps prevent single factor from dominating)
MAX_REST_PTS = 4.0
MAX_B2B_PTS = 3.0
MAX_CONGESTION_PTS = 3.0
MAX_STREAK_PTS = 2.0


def run_stage5(games: List[HurdleGame]) -> List[HurdleGame]:
    """
    Run structural edge analysis on all games.
    Returns games that pass or are degraded.
    """
    passed = []
    killed = 0

    for game in games:
        result = _evaluate_structural(game)
        game.stage_results.append(result)

        if result.verdict == Verdict.PASS:
            passed.append(game)
        elif result.verdict == Verdict.DEGRADE:
            game.degraded = True
            game.degrade_factor *= 0.85
            passed.append(game)
        else:
            killed += 1
            logger.info(
                f"[STAGE 5] KILL {game.away_team} @ {game.home_team}: {result.notes}"
            )

    logger.info(f"[STAGE 5] {len(passed)} passed, {killed} killed")
    return passed


def _evaluate_structural(game: HurdleGame) -> StageResult:
    """Score structural factors for a single game."""
    factors = {}
    aligned_count = 0  # How many factors point in same direction
    score = 0.0

    # --- 1. REST DIFFERENTIAL ---
    if game.home_rest_days >= 0 and game.away_rest_days >= 0:
        rest_diff = game.home_rest_days - game.away_rest_days
        # Positive = home rested more, negative = away rested more
        rest_pts = min(abs(rest_diff) * 1.3, MAX_REST_PTS)
        score += rest_pts
        if rest_pts > 0:
            aligned_count += 1
        factors["rest_differential"] = {
            "score": round(rest_pts, 1),
            "home_rest": game.home_rest_days,
            "away_rest": game.away_rest_days,
            "diff": rest_diff,
            "edge": game.home_team if rest_diff > 0 else game.away_team if rest_diff < 0 else "even",
        }

    # --- 2. B2B PENALTY ---
    home_b2b = game.home_b2b
    away_b2b = game.away_b2b
    if home_b2b != away_b2b:  # One side on B2B, other isn't
        b2b_pts = MAX_B2B_PTS
        score += b2b_pts
        aligned_count += 1
        b2b_team = game.home_team if home_b2b else game.away_team
        fresh_team = game.away_team if home_b2b else game.home_team
        factors["b2b_penalty"] = {
            "score": b2b_pts,
            "b2b_team": b2b_team,
            "fresh_team": fresh_team,
        }
    elif home_b2b and away_b2b:
        # Both on B2B — neutralizes, slight penalty
        score -= 1.0
        factors["b2b_both"] = {"score": -1.0, "note": "Both teams on B2B — no edge"}

    # --- 3. FIXTURE CONGESTION ---
    home_g7 = game.home_games_7d
    away_g7 = game.away_games_7d
    if home_g7 > 0 or away_g7 > 0:
        congestion_diff = away_g7 - home_g7  # Positive = away more congested (good for home)
        if abs(congestion_diff) >= 1:
            congestion_pts = min(abs(congestion_diff) * 1.0, MAX_CONGESTION_PTS)
            score += congestion_pts
            aligned_count += 1
            factors["congestion"] = {
                "score": round(congestion_pts, 1),
                "home_games_7d": home_g7,
                "away_games_7d": away_g7,
                "diff": congestion_diff,
                "edge": game.home_team if congestion_diff > 0 else game.away_team,
            }

    # --- 4. STREAK MOMENTUM ---
    home_streak = game.home_streak
    away_streak = game.away_streak
    streak_diff = home_streak - away_streak
    if abs(streak_diff) >= 3:
        streak_pts = min(abs(streak_diff) * 0.5, MAX_STREAK_PTS)
        score += streak_pts
        aligned_count += 1
        factors["streak_momentum"] = {
            "score": round(streak_pts, 1),
            "home_streak": home_streak,
            "away_streak": away_streak,
            "edge": game.home_team if streak_diff > 0 else game.away_team,
        }

    # --- 5. EDGE DIRECTION ALIGNMENT ---
    # Check if multiple factors point in the same direction
    edge_directions = []
    for key, fdata in factors.items():
        if isinstance(fdata, dict) and "edge" in fdata and fdata.get("score", 0) > 0:
            edge_directions.append(fdata["edge"])

    if edge_directions:
        from collections import Counter
        direction_counts = Counter(edge_directions)
        dominant_dir, dominant_count = direction_counts.most_common(1)[0]
        alignment_bonus = 0.5 * (dominant_count - 1)  # Bonus for aligned factors
        score += alignment_bonus
        factors["alignment"] = {
            "dominant_edge": dominant_dir,
            "aligned_factors": dominant_count,
            "bonus": round(alignment_bonus, 1),
        }

    score = round(max(0.0, min(10.0, score)), 1)
    factors["aligned_factor_count"] = aligned_count

    # --- VERDICT ---
    # Stages 0-5: scoring only, no kills. First hurdle at Stage 6 (AI).
    if score >= PASS_THRESHOLD and aligned_count >= 2:
        verdict = Verdict.PASS
        notes = f"Structural score {score} with {aligned_count} aligned factors"
    elif score >= DEGRADE_THRESHOLD:
        verdict = Verdict.DEGRADE
        notes = f"Structural score {score} — weak structural edge"
    else:
        verdict = Verdict.PASS
        notes = f"Structural score {score} — no structural edge found, accumulated"

    return StageResult(
        stage=5,
        name="Structural Edge",
        game_id=game.game_id,
        score=score,
        threshold=PASS_THRESHOLD,
        verdict=verdict,
        confidence=0.7 if verdict == Verdict.PASS else 0.5,
        next_stage=6 if verdict in (Verdict.PASS, Verdict.DEGRADE) else None,
        factors=factors,
        notes=notes,
    )

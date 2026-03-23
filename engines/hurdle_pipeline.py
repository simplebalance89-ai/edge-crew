"""
8-STAGE HURDLE PIPELINE — Orchestrator

Chains games through Stage 0 → 1 → 2 → ... → 8.
Each stage returns PASS/KILL/SOFT_FAIL/DEGRADE.
Killed games drop. Survivors proceed. Returns top 5 by composite score.

Phase 1: Stage 0 (morning filter) is fully wired.
Phase 2: Stages 1-2 (fixture validation + origin check) wired.
Stages 3-8 are stubbed — they pass through to the existing analysis engine.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from engines.stage_models import HurdleGame, StageResult, Verdict
from engines.stage0_filter import run_stage0, format_stage0_report
from engines.stage1_fixture import run_stage1
from engines.stage2_origin import run_stage2
from engines.stage3_team_dna import run_stage3
from engines.stage4_h2h import run_stage4
from engines.stage5_structural import run_stage5
from engines.stage6_market import run_stage6

logger = logging.getLogger("edge-crew")

# PST timezone offset (UTC-8, or UTC-7 during DST)
try:
    from zoneinfo import ZoneInfo
    PST = ZoneInfo("America/Los_Angeles")
except ImportError:
    PST = timezone(timedelta(hours=-7))  # Approximate PDT


async def run_hurdle_pipeline(
    sport: str,
    fetch_odds_fn,
    build_team_profile_fn,
    fetch_injuries_fn,
    opening_lines: dict,
    scores_archive: dict = None,
    team_match_fn=None,
    ai_caller=None,
    force: bool = False,
) -> Dict:
    """
    Main pipeline entry point.

    Args:
        sport: Sport key (nba, nhl, soccer, etc.)
        fetch_odds_fn: Async function that returns odds dict with "games" list.
                       This is the internal _fetch_odds or equivalent from server.py.
        build_team_profile_fn: Function(team, sport) -> dict with last_5, streak, etc.
        fetch_injuries_fn: Async function(sport) -> injury text or structured data.
        opening_lines: Reference to server.py's _opening_lines dict for line movement.
        scores_archive: Dict of completed games for H2H lookups (Stage 4).
        team_match_fn: Function(team, game, sport) -> 'home'|'away'|'' for fuzzy matching.
        ai_caller: Optional async function(prompt) -> str for AI stages (3, 6).
        force: If True, skip cache and re-run.

    Returns:
        Dict with stage0 results, survivors, timing, and (future) final picks.
    """
    start = time.time()
    sport_lower = sport.lower()

    # ================================================================
    # STEP 1: Fetch today's games from odds
    # ================================================================
    odds_data = await fetch_odds_fn(sport_lower)
    raw_games = []
    if isinstance(odds_data, dict):
        raw_games = odds_data.get("games", [])
    elif isinstance(odds_data, list):
        raw_games = odds_data

    if not raw_games:
        return {
            "sport": sport_lower,
            "error": "No games found for today",
            "stage0": None,
            "survivors": [],
            "timing_ms": int((time.time() - start) * 1000),
        }

    # ================================================================
    # STEP 2: Hydrate HurdleGames with all available data
    # ================================================================
    hurdle_games = []
    for raw in raw_games:
        game = HurdleGame.from_odds_game(raw, sport_lower)

        # --- Line movement from opening_lines ---
        _enrich_line_movement(game, raw, opening_lines)

        # --- Rest / B2B / Streaks from team profiles ---
        _enrich_rest_and_streaks(game, build_team_profile_fn, sport_lower)

        # --- Star injury counts ---
        _enrich_star_injuries(game, raw)

        hurdle_games.append(game)

    # ================================================================
    # STEP 3: Run Stage 0 — Morning Filter
    # ================================================================
    stage0_result = await run_stage0(hurdle_games)

    survivors = stage0_result["passed"]
    soft_fails = stage0_result["soft_fail"]

    # ================================================================
    # STAGE 1: Fixture Validation
    # ================================================================
    stage1_survivors = run_stage1(survivors)
    stage1_killed = len(survivors) - len(stage1_survivors)

    # ================================================================
    # STAGE 2: Origin Check (Injuries + Star Impact)
    # ================================================================
    stage2_passed, stage2_soft_fail, stage2_killed = await run_stage2(
        stage1_survivors,
        fetch_injuries_fn=fetch_injuries_fn,
        sport=sport_lower,
    )

    # Merge soft fails from Stage 2 into the overall soft_fail list
    all_soft_fails = soft_fails + stage2_soft_fail

    # Final survivors after Stages 0-2
    final_survivors = stage2_passed

    # ================================================================
    # STAGE 3: Team DNA (Tactical Matchup — AI Blinded)
    # ================================================================
    stage3_in = len(final_survivors)
    final_survivors = await run_stage3(final_survivors, ai_caller=ai_caller)
    stage3_killed = stage3_in - len(final_survivors)

    # ================================================================
    # STAGE 4: H2H Context (Decay-Weighted History)
    # ================================================================
    stage4_in = len(final_survivors)
    final_survivors = run_stage4(
        final_survivors,
        scores_archive=scores_archive or {},
        team_match_fn=team_match_fn,
    )
    stage4_killed = stage4_in - len(final_survivors)

    # ================================================================
    # STAGE 5: Structural Edge (Rest, Travel, Congestion)
    # ================================================================
    stage5_in = len(final_survivors)
    final_survivors = run_stage5(final_survivors)
    stage5_killed = stage5_in - len(final_survivors)

    # ================================================================
    # STAGE 6: Market Validation (Blinded AI + EV Math)
    # ================================================================
    stage6_in = len(final_survivors)
    final_survivors = await run_stage6(final_survivors, ai_caller=ai_caller)
    stage6_killed = stage6_in - len(final_survivors)

    # ================================================================
    # STAGES 7-8: STUB — Phase 4-5 will add lineup lock + final grade.
    # ================================================================

    # Build response
    elapsed = int((time.time() - start) * 1000)

    return {
        "sport": sport_lower,
        "timestamp": datetime.now(PST).isoformat(),
        "total_games": len(hurdle_games),
        "stage0": {
            "passed": len(survivors),
            "killed": len(stage0_result["killed"]),
            "soft_fail": len(soft_fails),
            "kill_rate": stage0_result["kill_rate"],
            "slate_profile": stage0_result["slate_profile"],
            "dynamic_weights": stage0_result["dynamic_weights"],
        },
        "stage1": {
            "passed": len(stage1_survivors),
            "killed": stage1_killed,
        },
        "stage2": {
            "passed": len(stage2_passed),
            "soft_fail": len(stage2_soft_fail),
            "killed": len(stage2_killed),
        },
        "stage3": {"killed": stage3_killed},
        "stage4": {"killed": stage4_killed},
        "stage5": {"killed": stage5_killed},
        "stage6": {"killed": stage6_killed},
        "survivors": [g.to_dict() for g in final_survivors],
        "killed": [
            {
                "matchup": f"{g.away_team} @ {g.home_team}",
                "game_id": g.game_id,
                "killed_at_stage": g.stage_results[-1].stage if g.stage_results else 0,
                "score": g.stage_results[-1].score if g.stage_results else 0,
                "reason": g.stage_results[-1].notes if g.stage_results else "unknown",
            }
            for g in (stage0_result["killed"] + stage2_killed)
        ],
        "soft_fail": [
            {
                "matchup": f"{g.away_team} @ {g.home_team}",
                "game_id": g.game_id,
                "stage": g.stage_results[-1].stage if g.stage_results else 0,
                "reason": g.stage_results[-1].notes if g.stage_results else "unknown",
            }
            for g in all_soft_fails
        ],
        "survivor_matchups": [
            f"{g.away_team} @ {g.home_team}" for g in final_survivors
        ],
        "report": format_stage0_report(stage0_result),
        "timing_ms": elapsed,
    }


# ================================================================
# DATA ENRICHMENT FUNCTIONS
# ================================================================

def _enrich_line_movement(
    game: HurdleGame, raw: Dict, opening_lines: dict
) -> None:
    """Calculate line movement from opening_lines dict."""
    today = datetime.now(PST).strftime("%Y-%m-%d")
    game_id = raw.get("id", raw.get("game_id", ""))
    key = f"{today}:{game_id}"

    opening = opening_lines.get(key)
    if not opening:
        return

    current_spread = game.home_spread
    current_total = game.total
    open_spread = opening.get("spread")
    open_total = opening.get("total")

    if current_spread is not None and open_spread is not None:
        game.spread_move = round(current_spread - open_spread, 1)
        game.opening_spread = open_spread

    if current_total is not None and open_total is not None:
        game.total_move = round(current_total - open_total, 1)
        game.opening_total = open_total


def _enrich_rest_and_streaks(
    game: HurdleGame, build_profile_fn, sport: str
) -> None:
    """Get rest days, B2B, and streaks from team profiles (scores archive)."""
    now = datetime.now(PST)
    today_str = now.strftime("%Y-%m-%d")

    for side, team in [("home", game.home_team), ("away", game.away_team)]:
        if not team:
            continue
        try:
            profile = build_profile_fn(team, sport)
            if not profile:
                continue

            # Streak
            streak = profile.get("streak", 0)
            if side == "home":
                game.home_streak = streak
            else:
                game.away_streak = streak

            # L5 record
            l5_record = profile.get("l5_record", "")
            if side == "home":
                game.home_l5_record = l5_record
            else:
                game.away_l5_record = l5_record

            # Rest days — calculate from last game date
            last_5 = profile.get("last_5", [])
            if last_5:
                last_game = last_5[0]
                last_date_str = last_game.get("date", "")
                if last_date_str:
                    try:
                        last_date = datetime.strptime(last_date_str[:10], "%Y-%m-%d")
                        rest = (now.replace(tzinfo=None) - last_date).days - 1
                        rest = max(0, rest)
                        if side == "home":
                            game.home_rest_days = rest
                            game.home_b2b = rest == 0
                        else:
                            game.away_rest_days = rest
                            game.away_b2b = rest == 0
                    except (ValueError, TypeError):
                        pass

                # Games in last 7 days
                week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d")
                games_7d = sum(
                    1 for g in last_5
                    if g.get("date", "") >= week_ago
                )
                if side == "home":
                    game.home_games_7d = games_7d
                else:
                    game.away_games_7d = games_7d

        except Exception as e:
            logger.debug(f"Profile enrichment failed for {team}: {e}")


def _enrich_star_injuries(game: HurdleGame, raw: Dict) -> None:
    """
    Count star injuries from game data.

    For Phase 1, we use a simple heuristic: any injury with "OUT" status
    on a key player counts as star_out. "GTD"/"Questionable" = star_questionable.

    Phase 2 will replace this with the full ESPN injury parser + star tier lookup.
    """
    injuries = raw.get("injuries", [])
    if not injuries:
        return

    for inj in injuries:
        status = (inj.get("status", "") or "").upper()
        team = (inj.get("team", "") or "").lower()
        is_star = inj.get("is_star", False) or inj.get("impact", 0) > 0.7

        if not is_star:
            continue

        home_norm = game.home_team.lower()
        away_norm = game.away_team.lower()

        is_home = team in home_norm or home_norm in team
        is_away = team in away_norm or away_norm in team

        if status in ("OUT", "O"):
            if is_home:
                game.home_star_out += 1
            elif is_away:
                game.away_star_out += 1
        elif status in ("GTD", "QUESTIONABLE", "DOUBTFUL", "DAY-TO-DAY"):
            if is_home:
                game.home_star_questionable += 1
            elif is_away:
                game.away_star_questionable += 1

    game.injuries = injuries

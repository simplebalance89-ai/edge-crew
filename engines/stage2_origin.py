"""
STAGE 2: ORIGIN CHECK — Player fitness and availability gate.

Uses sport-specific star impact tables to quantify how much each injury
affects win probability. Wraps existing ESPN + RotoWire injury data.

PASS: Key players >80% fit, no critical outs (score >= 4.0)
SOFT_FAIL: Star questionable → retry at T-60min
DEGRADE: Some depth concerns but not disqualifying
KILL: Multiple key outs destroy structural edge (star_impact > 20%)
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from engines.stage_models import HurdleGame, StageResult, Verdict

logger = logging.getLogger("edge-crew")

# ================================================================
# SPORT-SPECIFIC STAR IMPACT TABLES
# Impact = estimated win probability swing when player is OUT
# ================================================================

STAR_IMPACT = {
    "nba": {
        "mvp": 22,          # Top 5 player (Jokic, Shai, Luka, etc.)
        "all_star": 12,     # All-Star caliber
        "starter": 6,       # Quality starter
        "role": 3,          # Rotation player
        "deep_bench": 1,    # Minimal impact
    },
    "nhl": {
        "elite_goalie": 20,     # Franchise goalie (Shesterkin, Hellebuyck)
        "starting_goalie": 14,  # Starting goalie
        "top_line_forward": 8,  # First-line scorer
        "top_pair_d": 7,        # Top-pair defenseman
        "second_line": 4,       # Depth forward
        "depth": 2,
    },
    "soccer": {
        "elite_striker": 15,    # Haaland, Mbappe tier
        "creative_mid": 12,     # Playmaker (De Bruyne, Odegaard)
        "defensive_anchor": 10, # CB anchor or holding mid
        "starting_keeper": 12,  # First-choice GK
        "fullback": 5,          # Starting fullback
        "rotation": 3,
        "depth": 1,
    },
    "ncaab": {
        "all_american": 12,     # All-American / conference POY
        "starter": 6,           # Key starter
        "role": 3,
        "bench": 1,
    },
    "nfl": {
        "franchise_qb": 25,     # Elite QB
        "starting_qb": 18,      # Starting QB
        "all_pro": 10,          # All-Pro skill player / edge rusher
        "starter": 5,
        "depth": 2,
    },
    "mlb": {
        "ace_pitcher": 15,      # Starting pitcher (today)
        "star_hitter": 8,       # MVP-caliber bat
        "closer": 6,            # Elite closer
        "starter": 3,
        "depth": 1,
    },
}

# Injury status severity ordering
STATUS_SEVERITY = {
    "out": 4,
    "o": 4,
    "doubtful": 3,
    "questionable": 2,
    "gtd": 2,
    "day-to-day": 2,
    "probable": 1,
    "p": 1,
    "active": 0,
}

# Threshold: total star impact % above which we KILL
KILL_THRESHOLD = 20  # 20% win prob impact = too much uncertainty
DEGRADE_THRESHOLD = 10
PASS_SCORE_THRESHOLD = 4.0


async def run_stage2(
    games: List[HurdleGame],
    fetch_injuries_fn=None,
    sport: str = "",
) -> Tuple[List[HurdleGame], List[HurdleGame], List[HurdleGame]]:
    """
    Run origin check on all games.

    If fetch_injuries_fn is provided, will call it to get fresh injury data.
    Otherwise relies on injury data already on the HurdleGame objects.

    Returns: (passed, soft_fail, killed)
    """
    # Optionally fetch fresh injuries (Phase 2+ — for now, use what's on the game)
    if fetch_injuries_fn and sport:
        try:
            injury_text = await fetch_injuries_fn(sport)
            if injury_text:
                _parse_injury_text_onto_games(games, injury_text, sport)
        except Exception as e:
            logger.warning(f"[STAGE 2] Injury fetch failed: {e} — using existing data")

    passed = []
    soft_fail = []
    killed = []

    for game in games:
        result = _evaluate_origin(game)
        game.stage_results.append(result)

        if result.verdict == Verdict.PASS:
            passed.append(game)
        elif result.verdict == Verdict.SOFT_FAIL:
            soft_fail.append(game)
        elif result.verdict == Verdict.DEGRADE:
            game.degraded = True
            game.degrade_factor *= 0.8  # Reduce downstream confidence by 20%
            passed.append(game)  # Degraded games still proceed
        else:
            killed.append(game)

    logger.info(
        f"[STAGE 2] {len(passed)} passed (inc. degraded), "
        f"{len(soft_fail)} soft_fail, {len(killed)} killed"
    )
    return passed, soft_fail, killed


def _evaluate_origin(game: HurdleGame) -> StageResult:
    """Score a single game's injury situation."""
    factors = {}
    sport = game.sport

    # Calculate impact for each side
    home_impact = _calculate_side_impact(
        game.home_star_out, game.home_star_questionable, sport, "home"
    )
    away_impact = _calculate_side_impact(
        game.away_star_out, game.away_star_questionable, sport, "away"
    )

    factors["home_impact"] = home_impact
    factors["away_impact"] = away_impact

    # The EDGE from injuries: if one side is hurt more, the other side benefits
    # We care about differential — a game where BOTH sides are hurt is less edgy
    home_pct = home_impact["total_impact_pct"]
    away_pct = away_impact["total_impact_pct"]
    impact_diff = abs(home_pct - away_pct)
    max_impact = max(home_pct, away_pct)

    factors["impact_differential"] = round(impact_diff, 1)
    factors["max_side_impact"] = round(max_impact, 1)

    # Determine which side the edge favors
    if home_pct > away_pct:
        factors["edge_favors"] = game.away_team
        factors["hurt_side"] = game.home_team
    elif away_pct > home_pct:
        factors["edge_favors"] = game.home_team
        factors["hurt_side"] = game.away_team
    else:
        factors["edge_favors"] = "neutral"

    # Score: start at 5 (neutral), adjust based on differential
    # High differential = good (clear edge), both sides hurt = bad (chaos)
    score = 5.0

    # Bonus for injury differential (creates edge)
    if impact_diff >= 15:
        score += 3.0
    elif impact_diff >= 10:
        score += 2.0
    elif impact_diff >= 5:
        score += 1.0

    # Penalty if both sides are heavily impacted (unpredictable)
    if home_pct > 10 and away_pct > 10:
        score -= 2.0
        factors["both_sides_hurt"] = True

    # Penalty for extreme single-side impact (too much uncertainty)
    if max_impact > KILL_THRESHOLD:
        score -= 3.0

    score = max(0.0, min(10.0, score))

    # --- Verdict ---
    has_questionable = (
        game.home_star_questionable > 0 or game.away_star_questionable > 0
    )

    if max_impact > KILL_THRESHOLD and impact_diff < 5:
        # Both sides devastated or one side so hurt it's unpredictable
        verdict = Verdict.KILL
        notes = (
            f"Excessive injury impact ({max_impact:.0f}%) with low differential "
            f"({impact_diff:.0f}%) — too uncertain"
        )
    elif has_questionable and score >= PASS_SCORE_THRESHOLD:
        verdict = Verdict.SOFT_FAIL
        notes = (
            f"Score {score:.1f} but star(s) questionable — retry when status confirmed"
        )
    elif max_impact > DEGRADE_THRESHOLD and score >= PASS_SCORE_THRESHOLD:
        verdict = Verdict.DEGRADE
        notes = (
            f"Score {score:.1f}, impact {max_impact:.0f}% — edge exists but "
            f"degraded confidence"
        )
    elif score >= PASS_SCORE_THRESHOLD:
        verdict = Verdict.PASS
        notes = f"Score {score:.1f} — injury edge clear"
    else:
        verdict = Verdict.KILL
        notes = f"Score {score:.1f} < {PASS_SCORE_THRESHOLD} — no injury edge"

    return StageResult(
        stage=2,
        name="Origin Check",
        game_id=game.game_id,
        score=round(score, 1),
        threshold=PASS_SCORE_THRESHOLD,
        verdict=verdict,
        confidence=0.7 if verdict == Verdict.PASS else 0.4,
        next_stage=3 if verdict in (Verdict.PASS, Verdict.DEGRADE) else None,
        factors=factors,
        notes=notes,
    )


def _calculate_side_impact(
    stars_out: int, stars_questionable: int, sport: str, side: str
) -> Dict:
    """
    Calculate total win probability impact for one side's injuries.

    For Phase 1, we use star counts with average tier impact.
    Phase 2+ will have actual player names → tier lookups.
    """
    impact_table = STAR_IMPACT.get(sport, STAR_IMPACT.get("nba", {}))

    # Use mid-tier default per star count
    # (Phase 2 will have real player → tier mapping)
    avg_star_impact = impact_table.get("all_star", impact_table.get("starter", 8))
    avg_role_impact = impact_table.get("role", 3)

    out_impact = stars_out * avg_star_impact
    q_impact = stars_questionable * (avg_star_impact * 0.4)  # 40% weight for questionable

    total = min(out_impact + q_impact, 40)  # Cap at 40%

    return {
        "stars_out": stars_out,
        "stars_questionable": stars_questionable,
        "out_impact_pct": round(out_impact, 1),
        "questionable_impact_pct": round(q_impact, 1),
        "total_impact_pct": round(total, 1),
        "side": side,
    }


def _parse_injury_text_onto_games(
    games: List[HurdleGame], injury_text: str, sport: str
) -> None:
    """
    Parse injury text from ESPN/RotoWire and enrich HurdleGame objects.

    This is a best-effort parser for the text format our injury fetchers return.
    Phase 2 will have structured injury data instead of text parsing.
    """
    if not injury_text:
        return

    # Simple heuristic: look for OUT/GTD/Questionable patterns near team names
    lines = injury_text.split("\n")
    current_team = ""

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Detect team headers (usually bold or all caps)
        # Pattern: "TEAM NAME" or "**Team Name**"
        upper_line = line_stripped.upper()

        # Check if this line names a team that's in our games
        for game in games:
            for team_attr in [game.home_team, game.away_team]:
                if not team_attr:
                    continue
                team_norm = team_attr.lower()
                if team_norm in line_stripped.lower():
                    current_team = team_attr
                    break

        # Look for injury status keywords
        if current_team:
            is_star_indicator = any(
                kw in line_stripped.lower()
                for kw in ["star", "mvp", "all-star", "all star", "key", "starter"]
            )

            if "OUT" in upper_line or "OUT -" in upper_line:
                _increment_injury(games, current_team, "out", is_star_indicator)
            elif any(s in upper_line for s in ["GTD", "QUESTIONABLE", "DOUBTFUL"]):
                _increment_injury(games, current_team, "questionable", is_star_indicator)


def _increment_injury(
    games: List[HurdleGame], team: str, status: str, is_star: bool
) -> None:
    """Increment injury counters on matching HurdleGame."""
    if not is_star:
        return  # Only track star injuries for Stage 2

    team_lower = team.lower()
    for game in games:
        if team_lower in game.home_team.lower() or game.home_team.lower() in team_lower:
            if status == "out":
                game.home_star_out += 1
            else:
                game.home_star_questionable += 1
            return
        if team_lower in game.away_team.lower() or game.away_team.lower() in team_lower:
            if status == "out":
                game.away_star_out += 1
            else:
                game.away_star_questionable += 1
            return

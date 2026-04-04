"""
STAGE 2: PLAYER DATA — Injuries, star availability, usage, lineup status.

Displays and aggregates all player-level data for the pipeline.
Uses sport-specific star impact tables to quantify how much each injury
affects win probability. Wraps existing ESPN + RotoWire injury data.

SCORING STAGES (0-5): accumulate data, NO KILLS.
S2 never kills — only PASS, SOFT_FAIL, or DEGRADE.

Key methodology:
- Fresh scratches (0-3 days) = market hasn't adjusted = edge opportunity
- Stale absences (30+ days, team winning without them) = already priced in
- Injury differential = clear edge for the healthier side
- Both sides heavily hurt = chaos, reduce confidence
"""

import logging
import re
from datetime import datetime, timedelta
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
    "game-time decision": 2,
    "day-to-day": 2,
    "probable": 1,
    "p": 1,
    "active": 0,
    "available": 0,
}

# Thresholds
DEGRADE_THRESHOLD = 10
PASS_SCORE_THRESHOLD = 4.0

# Freshness boundaries (days)
FRESH_INJURY_DAYS = 3       # 0-3 days = fresh scratch, market hasn't adjusted
STALE_INJURY_DAYS = 30      # 30+ days = priced in if team is winning without them


async def run_stage2(
    games: List[HurdleGame],
    soft_fail_games: Optional[List[HurdleGame]] = None,
    fetch_injuries_fn=None,
) -> Tuple[List[HurdleGame], List[HurdleGame], List[HurdleGame]]:
    """
    Run player data stage on all games.

    Evaluates injuries, star availability, usage patterns, and lineup status.
    Never kills — only PASS, SOFT_FAIL, or DEGRADE.

    Args:
        games: Active games to evaluate.
        soft_fail_games: Previously soft-failed games to re-evaluate.
        fetch_injuries_fn: Optional async callable(sport) -> injury text/data.

    Returns: (passed, soft_fail, killed) — killed is always empty.
    """
    all_games = list(games)
    if soft_fail_games:
        all_games.extend(soft_fail_games)

    # Optionally fetch fresh injuries
    if fetch_injuries_fn:
        # Group games by sport for efficient fetching
        sports_seen = set()
        for g in all_games:
            if g.sport and g.sport not in sports_seen:
                sports_seen.add(g.sport)
                try:
                    injury_data = await fetch_injuries_fn(g.sport)
                    if injury_data:
                        sport_games = [gm for gm in all_games if gm.sport == g.sport]
                        _parse_injury_text_onto_games(sport_games, injury_data, g.sport)
                except Exception as e:
                    logger.warning(f"[STAGE 2] Injury fetch failed for {g.sport}: {e}")

    passed = []
    soft_fail = []
    killed = []  # Always empty — S2 never kills

    for game in all_games:
        result = _evaluate_player_data(game)
        game.stage_results.append(result)

        if result.verdict == Verdict.PASS:
            passed.append(game)
        elif result.verdict == Verdict.SOFT_FAIL:
            soft_fail.append(game)
        elif result.verdict == Verdict.DEGRADE:
            game.degraded = True
            game.degrade_factor *= 0.8  # Reduce downstream confidence by 20%
            passed.append(game)  # Degraded games still proceed

    logger.info(
        f"[STAGE 2] Player Data: {len(passed)} passed (inc. degraded), "
        f"{len(soft_fail)} soft_fail, 0 killed"
    )
    return passed, soft_fail, killed


def _evaluate_player_data(game: HurdleGame) -> StageResult:
    """Score a single game's full player picture."""
    factors: Dict = {}
    sport = game.sport

    # ------------------------------------------------------------------
    # 1. STAR AVAILABILITY — calculate impact for each side
    # ------------------------------------------------------------------
    home_impact = _calculate_side_impact(
        game.home_star_out, game.home_star_questionable, sport, "home"
    )
    away_impact = _calculate_side_impact(
        game.away_star_out, game.away_star_questionable, sport, "away"
    )

    factors["home_impact"] = home_impact
    factors["away_impact"] = away_impact

    home_pct = home_impact["total_impact_pct"]
    away_pct = away_impact["total_impact_pct"]
    impact_diff = abs(home_pct - away_pct)
    max_impact = max(home_pct, away_pct)

    factors["impact_differential"] = round(impact_diff, 1)
    factors["max_side_impact"] = round(max_impact, 1)

    # Which side has the edge from injuries
    if home_pct > away_pct:
        factors["edge_favors"] = game.away_team
        factors["hurt_side"] = game.home_team
    elif away_pct > home_pct:
        factors["edge_favors"] = game.home_team
        factors["hurt_side"] = game.away_team
    else:
        factors["edge_favors"] = "neutral"

    # ------------------------------------------------------------------
    # 2. INJURY FRESHNESS — fresh scratches vs stale absences
    # ------------------------------------------------------------------
    freshness = _assess_injury_freshness(game)
    factors["injury_freshness"] = freshness

    # ------------------------------------------------------------------
    # 3. INJURY DETAIL BREAKDOWN (per-player if available)
    # ------------------------------------------------------------------
    if game.injuries:
        home_injuries = []
        away_injuries = []
        for inj in game.injuries:
            entry = {
                "player": inj.get("player", "Unknown"),
                "status": inj.get("status", "unknown"),
                "tier": inj.get("tier", "unknown"),
                "days_out": inj.get("days_out", -1),
                "reason": inj.get("reason", ""),
            }
            side = inj.get("side", "")
            if side == "home":
                home_injuries.append(entry)
            elif side == "away":
                away_injuries.append(entry)
        if home_injuries:
            factors["home_injuries_detail"] = home_injuries
        if away_injuries:
            factors["away_injuries_detail"] = away_injuries

    # ------------------------------------------------------------------
    # 4. USAGE / LOAD MANAGEMENT SIGNALS
    # ------------------------------------------------------------------
    usage_flags = _check_usage_signals(game)
    if usage_flags:
        factors["usage_signals"] = usage_flags

    # ------------------------------------------------------------------
    # 5. LINEUP CONFIRMATION STATUS
    # ------------------------------------------------------------------
    lineup_info = _check_lineup_status(game)
    factors["lineup_status"] = lineup_info

    # ------------------------------------------------------------------
    # 6. SCORING — start at 5.0 (neutral)
    # ------------------------------------------------------------------
    score = 5.0

    # Bonus for injury differential (creates edge)
    if impact_diff >= 15:
        score += 3.0
    elif impact_diff >= 10:
        score += 2.0
    elif impact_diff >= 5:
        score += 1.0

    # Penalty if both sides heavily impacted (unpredictable chaos)
    if home_pct > 10 and away_pct > 10:
        score -= 2.0
        factors["both_sides_hurt"] = True

    # Penalty for extreme single-side impact (too much uncertainty)
    if max_impact > 20:
        score -= 3.0

    # Fresh injury bonus: market hasn't fully adjusted
    if freshness.get("has_fresh_scratch"):
        score += 1.0
        factors["fresh_scratch_bonus"] = True

    # Stale injury discount: already priced in
    if freshness.get("has_stale_absence"):
        score -= 1.0
        factors["stale_absence_discount"] = True

    score = max(0.0, min(10.0, score))

    # ------------------------------------------------------------------
    # 7. VERDICT — never KILL, only PASS / SOFT_FAIL / DEGRADE
    # ------------------------------------------------------------------
    has_questionable = (
        game.home_star_questionable > 0 or game.away_star_questionable > 0
    )

    if has_questionable and score >= PASS_SCORE_THRESHOLD:
        verdict = Verdict.SOFT_FAIL
        notes = (
            f"Score {score:.1f} — star(s) questionable, retry when status confirmed"
        )
    elif max_impact > DEGRADE_THRESHOLD:
        verdict = Verdict.DEGRADE
        notes = (
            f"Score {score:.1f}, impact {max_impact:.0f}% — degraded confidence"
        )
    else:
        verdict = Verdict.PASS
        notes = f"Score {score:.1f} — player data accumulated"

    # Add summary to notes
    summary_parts = []
    if game.home_star_out or game.away_star_out:
        summary_parts.append(
            f"OUT: {game.home_team}={game.home_star_out}, "
            f"{game.away_team}={game.away_star_out}"
        )
    if game.home_star_questionable or game.away_star_questionable:
        summary_parts.append(
            f"GTD: {game.home_team}={game.home_star_questionable}, "
            f"{game.away_team}={game.away_star_questionable}"
        )
    if freshness.get("has_fresh_scratch"):
        summary_parts.append("FRESH scratch detected")
    if freshness.get("has_stale_absence"):
        summary_parts.append("STALE absence (priced in)")
    if summary_parts:
        notes += " | " + "; ".join(summary_parts)

    return StageResult(
        stage=2,
        name="Player Data",
        game_id=game.game_id,
        score=round(score, 1),
        threshold=PASS_SCORE_THRESHOLD,
        verdict=verdict,
        confidence=_calc_confidence(verdict, lineup_info),
        next_stage=3 if verdict in (Verdict.PASS, Verdict.DEGRADE) else None,
        factors=factors,
        notes=notes,
    )


def _calculate_side_impact(
    stars_out: int, stars_questionable: int, sport: str, side: str
) -> Dict:
    """
    Calculate total win probability impact for one side's injuries.

    Uses mid-tier defaults per star count.
    Phase 2+ will have actual player names -> tier lookups.
    """
    impact_table = STAR_IMPACT.get(sport, STAR_IMPACT.get("nba", {}))

    # Use mid-tier default per star count
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


def _assess_injury_freshness(game: HurdleGame) -> Dict:
    """
    Evaluate whether injuries are fresh scratches or stale absences.

    Fresh (0-3 days): Market hasn't fully adjusted -> edge opportunity.
    Stale (30+ days, team winning): Already priced in -> discount.
    """
    result = {
        "has_fresh_scratch": False,
        "has_stale_absence": False,
        "fresh_players": [],
        "stale_players": [],
    }

    if not game.injuries:
        return result

    for inj in game.injuries:
        days_out = inj.get("days_out", -1)
        player = inj.get("player", "Unknown")
        status = inj.get("status", "").lower()

        # Only care about players actually out or doubtful
        if status not in ("out", "o", "doubtful"):
            continue

        if 0 <= days_out <= FRESH_INJURY_DAYS:
            result["has_fresh_scratch"] = True
            result["fresh_players"].append({
                "player": player,
                "days_out": days_out,
                "note": "Fresh scratch — market may not have adjusted",
            })
        elif days_out >= STALE_INJURY_DAYS:
            # Check if team is winning without them (use streak as proxy)
            side = inj.get("side", "")
            team_winning = False
            if side == "home" and game.home_streak > 0:
                team_winning = True
            elif side == "away" and game.away_streak > 0:
                team_winning = True

            if team_winning:
                result["has_stale_absence"] = True
                result["stale_players"].append({
                    "player": player,
                    "days_out": days_out,
                    "note": "Stale absence — team winning without them, priced in",
                })

    return result


def _check_usage_signals(game: HurdleGame) -> List[Dict]:
    """
    Check for usage / load management signals from metadata.

    Looks for load management flags, minutes restrictions, usage rate data
    that may have been injected by upstream data fetchers.
    """
    signals = []
    meta = game.metadata or {}

    # Load management flags (set by NBA usage engine or upstream fetcher)
    for key in ("load_management", "usage_signals", "minutes_restrictions"):
        if key in meta:
            data = meta[key]
            if isinstance(data, list):
                for item in data:
                    signals.append({
                        "type": key,
                        "player": item.get("player", "Unknown"),
                        "detail": item.get("detail", str(item)),
                    })
            elif isinstance(data, dict):
                signals.append({
                    "type": key,
                    "detail": data,
                })

    # Check for B2B load management (back-to-back = likely rest stars)
    if game.home_b2b:
        signals.append({
            "type": "b2b_rest_risk",
            "team": game.home_team,
            "detail": f"{game.home_team} on back-to-back — star rest risk elevated",
        })
    if game.away_b2b:
        signals.append({
            "type": "b2b_rest_risk",
            "team": game.away_team,
            "detail": f"{game.away_team} on back-to-back — star rest risk elevated",
        })

    return signals


def _check_lineup_status(game: HurdleGame) -> Dict:
    """
    Check lineup confirmation status from metadata.

    Returns confidence level based on whether lineups are confirmed,
    projected, or unknown.
    """
    meta = game.metadata or {}
    lineup_data = meta.get("lineup_status", {})

    if lineup_data:
        return {
            "confirmed": lineup_data.get("confirmed", False),
            "source": lineup_data.get("source", "unknown"),
            "confidence": "high" if lineup_data.get("confirmed") else "projected",
            "home_lineup": lineup_data.get("home_lineup", []),
            "away_lineup": lineup_data.get("away_lineup", []),
        }

    # Default: no lineup data available
    return {
        "confirmed": False,
        "source": "none",
        "confidence": "unknown",
        "note": "Lineup data not yet available",
    }


def _calc_confidence(verdict: Verdict, lineup_info: Dict) -> float:
    """Calculate confidence based on verdict and data quality."""
    base = 0.7 if verdict == Verdict.PASS else 0.5 if verdict == Verdict.DEGRADE else 0.4

    # Boost confidence if lineups are confirmed
    if lineup_info.get("confirmed"):
        base = min(base + 0.1, 0.9)

    # Reduce confidence if no lineup data at all
    if lineup_info.get("source") == "none":
        base = max(base - 0.05, 0.2)

    return round(base, 2)


# ==================================================================
# INJURY TEXT PARSING (ESPN / RotoWire raw text -> game enrichment)
# ==================================================================

def _parse_injury_text_onto_games(
    games: List[HurdleGame], injury_text: str, sport: str
) -> None:
    """
    Parse injury text from ESPN/RotoWire and enrich HurdleGame objects.

    Best-effort parser for the text format our injury fetchers return.
    Phase 2 will have structured injury data instead of text parsing.
    """
    if not injury_text:
        return

    lines = injury_text.split("\n")
    current_team = ""

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

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
            elif "PROBABLE" in upper_line:
                # Track probable but don't count as questionable
                pass


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

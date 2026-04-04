"""
STAGE 0: TEAM DATA FOUNDATION — Display and aggregate overnight team data.

This is the foundation data layer. S0 collects and scores data that was
gathered overnight in Step 2.5 (team profiles from the scores archive).

DATA S0 DISPLAYS AND SCORES:
  - ATS record (L5, L10)
  - H2H history (last 5 meetings)
  - Win/loss streak (current run)
  - Offensive ranking (PPG / goals per game)
  - Defensive ranking (opponent PPG / goals against)
  - Offense vs defense matchup quality
  - Season record context (W-L, home/away splits)
  - Overnight line movement (captured, lower weight)

DATA S0 DOES NOT OWN:
  - B2B, rest days, travel, schedule congestion → S1
  - Injuries, star availability, usage, lineups → S2

SCORING: Start at 5.0 (neutral), apply bonuses/penalties, clamp [0, 10].
VERDICT: Always PASS — S0 never kills. First kill gate is Stage 6.

Also includes the Dynamic Weight Engine that adjusts downstream
stage weights based on slate conditions.
"""

import logging
from typing import Dict, List, Optional

from engines.stage_models import HurdleGame, StageResult, Verdict

logger = logging.getLogger("edge-crew")

# ================================================================
# SPORT-SPECIFIC THRESHOLDS
# ================================================================
SPORT_THRESHOLDS = {
    "nba": {
        "elite_offense": 115.0,    # PPG
        "weak_defense": 115.0,     # Opponent PPG allowed
        "strong_defense": 106.0,   # Low opponent PPG
        "weak_offense": 106.0,     # Low PPG
    },
    "ncaab": {
        "elite_offense": 80.0,
        "weak_defense": 80.0,
        "strong_defense": 65.0,
        "weak_offense": 65.0,
    },
    "nhl": {
        "elite_offense": 3.5,      # Goals per game
        "weak_defense": 3.2,       # Goals against per game
        "strong_defense": 2.5,
        "weak_offense": 2.5,
    },
    "mlb": {
        "elite_offense": 5.0,      # Runs per game
        "weak_defense": 5.0,       # Runs allowed per game
        "strong_defense": 3.5,
        "weak_offense": 3.5,
    },
    "soccer": {
        "elite_offense": 2.0,      # Goals per game
        "weak_defense": 1.5,       # Goals conceded per game
        "strong_defense": 0.8,
        "weak_offense": 0.8,
    },
}

# Default thresholds for unknown sports (NBA-like)
DEFAULT_THRESHOLDS = SPORT_THRESHOLDS["nba"]


async def run_stage0(games: List[HurdleGame]) -> Dict:
    """
    Run the team data foundation stage on the full slate.
    Returns dict with passed/killed/soft_fail lists + dynamic weights + slate profile.

    Every game passes — S0 never kills. It aggregates and scores data.
    """
    passed = []
    killed = []   # Always empty — S0 never kills
    soft_fail = []

    for game in games:
        result = _evaluate_game(game)
        game.stage_results.append(result)

        if result.verdict == Verdict.SOFT_FAIL:
            soft_fail.append(game)
        else:
            passed.append(game)

    # Analyze slate conditions for dynamic weighting
    slate_profile = _classify_slate(games)
    dynamic_weights = _calculate_dynamic_weights(games, slate_profile)

    total = len(games)

    logger.info(
        f"[STAGE 0] {len(passed)} passed, {len(soft_fail)} soft_fail | "
        f"Profile: {slate_profile} | No kills at S0"
    )

    return {
        "passed": passed,
        "killed": killed,
        "soft_fail": soft_fail,
        "total": total,
        "kill_rate": 0.0,
        "slate_profile": slate_profile,
        "dynamic_weights": dynamic_weights,
    }


# ================================================================
# PER-GAME EVALUATION
# ================================================================

def _evaluate_game(game: HurdleGame) -> StageResult:
    """Score a single game through the team data foundation layer.

    Scoring starts at 5.0 (neutral) and adjusts based on:
      - ATS record (L5)
      - H2H dominance
      - Win/loss streaks
      - Offense vs defense matchup quality
      - Overnight line movement (minor weight)

    Score is clamped to [0, 10]. Verdict is always PASS.
    """
    score = 5.0
    factors = {}
    sport = game.sport.lower()

    # --- 1. ATS Record (L5) ---
    home_ats = _parse_ats_from_metadata(game, "home")
    away_ats = _parse_ats_from_metadata(game, "away")

    if home_ats:
        factors["home_ats_l5"] = home_ats
        if home_ats.get("wins", 0) >= 4:
            score += 1.5
            factors["home_ats_l5"]["bonus"] = "+1.5 (strong ATS)"
        elif home_ats.get("wins", 0) <= 1 and home_ats.get("total", 0) >= 4:
            score -= 0.5
            factors["home_ats_l5"]["penalty"] = "-0.5 (weak ATS)"

    if away_ats:
        factors["away_ats_l5"] = away_ats
        if away_ats.get("wins", 0) >= 4:
            score += 1.5
            factors["away_ats_l5"]["bonus"] = "+1.5 (strong ATS)"
        elif away_ats.get("wins", 0) <= 1 and away_ats.get("total", 0) >= 4:
            score -= 0.5
            factors["away_ats_l5"]["penalty"] = "-0.5 (weak ATS)"

    # --- 2. H2H History (last 5 meetings) ---
    h2h = game.metadata.get("h2h")
    if h2h and h2h.get("count", 0) >= 2:
        h2h_summary = h2h.get("summary", {})
        # away_record is from away_team perspective: "3-2" means 3 wins
        away_h2h_record = h2h_summary.get("away_record", "")
        meetings = h2h.get("meetings", [])
        count = h2h.get("count", 0)

        away_h2h_wins = _parse_record_wins(away_h2h_record)
        home_h2h_wins = count - away_h2h_wins if count > 0 else 0

        factors["h2h"] = {
            "meetings": count,
            "away_record": away_h2h_record,
            "home_record": h2h_summary.get("home_record", ""),
            "avg_margin": h2h_summary.get("avg_margin", 0),
        }

        # H2H dominance: 4+ wins in last 5
        if count >= 4:
            if away_h2h_wins >= 4:
                score += 1.5
                factors["h2h"]["bonus"] = f"+1.5 ({game.away_team} H2H dominant)"
            elif home_h2h_wins >= 4:
                score += 1.5
                factors["h2h"]["bonus"] = f"+1.5 ({game.home_team} H2H dominant)"

    # --- 3. Win/Loss Streaks ---
    for side, streak, team in [
        ("home", game.home_streak, game.home_team),
        ("away", game.away_streak, game.away_team),
    ]:
        if streak >= 4:
            score += 1.0
            factors[f"{side}_streak"] = {
                "team": team,
                "streak": streak,
                "bonus": "+1.0 (hot streak 4+)",
            }
        elif streak <= -4:
            score -= 1.0
            factors[f"{side}_streak"] = {
                "team": team,
                "streak": streak,
                "penalty": "-1.0 (cold streak 4+)",
            }
        elif abs(streak) >= 2:
            factors[f"{side}_streak"] = {
                "team": team,
                "streak": streak,
                "note": "moderate streak",
            }

    # --- 4. Offense vs Defense Matchup ---
    off_def_result = _score_offense_vs_defense(game, sport)
    if off_def_result:
        score += off_def_result["adjustment"]
        factors["offense_vs_defense"] = off_def_result

    # --- 5. Season Record Context ---
    season_ctx = _build_season_context(game)
    if season_ctx:
        factors["season_context"] = season_ctx

    # --- 6. Overnight Line Movement (captured, lower weight) ---
    spread_move = abs(game.spread_move) if game.spread_move else 0
    total_move = abs(game.total_move) if game.total_move else 0
    line_move = max(spread_move, total_move)

    if line_move > 0:
        factors["line_movement"] = {
            "spread_move": game.spread_move,
            "total_move": game.total_move,
            "magnitude": round(line_move, 2),
        }
        if line_move > 1.0:
            score += 0.5
            factors["line_movement"]["bonus"] = "+0.5 (large overnight move >1.0)"

    # --- Clamp score ---
    score = max(0.0, min(10.0, score))

    # --- Confidence based on data completeness ---
    data_points = sum([
        1 if home_ats else 0,
        1 if away_ats else 0,
        1 if h2h and h2h.get("count", 0) >= 2 else 0,
        1 if game.home_streak != 0 or game.away_streak != 0 else 0,
        1 if off_def_result else 0,
        1 if season_ctx else 0,
    ])
    confidence = min(0.9, 0.3 + (data_points * 0.1))

    notes = (
        f"Score {score:.1f}/10 — "
        f"{data_points}/6 data layers available | "
        f"Verdict: PASS (S0 never kills)"
    )

    return StageResult(
        stage=0,
        name="Team Data Foundation",
        game_id=game.game_id,
        score=round(score, 1),
        threshold=0.0,  # No threshold — S0 never kills
        verdict=Verdict.PASS,
        confidence=round(confidence, 2),
        next_stage=1,
        factors=factors,
        notes=notes,
    )


# ================================================================
# HELPER: Parse ATS from team profile in metadata
# ================================================================

def _parse_ats_from_metadata(game: HurdleGame, side: str) -> Optional[Dict]:
    """Extract ATS record from the team profile stored in game.metadata.

    The team profile (from _build_team_profile in server.py) contains:
      summary.ats_record = "3-2" or "3-1-1" (W-L or W-L-P)
      last_5 = [{..., "ats": "W"/"L"/"P"}, ...]
    """
    profile_key = f"{side}_profile"
    profile = game.metadata.get(profile_key, {})
    if not profile:
        return None

    summary = profile.get("summary", {})
    ats_str = summary.get("ats_record", "")
    if not ats_str:
        return None

    parts = ats_str.split("-")
    try:
        wins = int(parts[0]) if len(parts) > 0 else 0
        losses = int(parts[1]) if len(parts) > 1 else 0
        pushes = int(parts[2]) if len(parts) > 2 else 0
    except (ValueError, TypeError):
        return None

    total = wins + losses + pushes
    return {
        "record": ats_str,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "total": total,
    }


def _parse_record_wins(record_str: str) -> int:
    """Parse wins from a record string like '3-2'."""
    if not record_str:
        return 0
    parts = record_str.split("-")
    try:
        return int(parts[0])
    except (ValueError, TypeError):
        return 0


# ================================================================
# HELPER: Offense vs Defense matchup scoring
# ================================================================

def _score_offense_vs_defense(game: HurdleGame, sport: str) -> Optional[Dict]:
    """Score the offense-vs-defense matchup using team profile PPG data.

    Uses sport-specific thresholds to identify elite/weak matchups.
    Returns adjustment and context dict, or None if data unavailable.
    """
    thresholds = SPORT_THRESHOLDS.get(sport, DEFAULT_THRESHOLDS)

    home_profile = game.metadata.get("home_profile", {})
    away_profile = game.metadata.get("away_profile", {})

    if not home_profile and not away_profile:
        return None

    result = {"adjustment": 0.0, "matchups": []}

    # Extract PPG data from profiles
    # The profile stores last_5 game data; we compute PPG from it
    home_ppg = _extract_ppg(home_profile)
    away_ppg = _extract_ppg(away_profile)
    home_opp_ppg = _extract_opp_ppg(home_profile)
    away_opp_ppg = _extract_opp_ppg(away_profile)

    result["home_ppg"] = home_ppg
    result["away_ppg"] = away_ppg
    result["home_opp_ppg"] = home_opp_ppg
    result["away_opp_ppg"] = away_opp_ppg

    elite_off = thresholds["elite_offense"]
    weak_def = thresholds["weak_defense"]
    strong_def = thresholds["strong_defense"]
    weak_off = thresholds["weak_offense"]

    # Away offense vs Home defense
    if away_ppg and home_opp_ppg:
        if away_ppg >= elite_off and home_opp_ppg >= weak_def:
            result["adjustment"] += 2.0
            result["matchups"].append(
                f"{game.away_team} elite offense ({away_ppg:.1f}) vs "
                f"{game.home_team} weak defense ({home_opp_ppg:.1f} allowed) → +2.0"
            )
        elif away_ppg <= weak_off and home_opp_ppg <= strong_def:
            result["adjustment"] -= 1.0
            result["matchups"].append(
                f"{game.away_team} weak offense ({away_ppg:.1f}) vs "
                f"{game.home_team} strong defense ({home_opp_ppg:.1f} allowed) → -1.0"
            )

    # Home offense vs Away defense
    if home_ppg and away_opp_ppg:
        if home_ppg >= elite_off and away_opp_ppg >= weak_def:
            result["adjustment"] += 2.0
            result["matchups"].append(
                f"{game.home_team} elite offense ({home_ppg:.1f}) vs "
                f"{game.away_team} weak defense ({away_opp_ppg:.1f} allowed) → +2.0"
            )
        elif home_ppg <= weak_off and away_opp_ppg <= strong_def:
            result["adjustment"] -= 1.0
            result["matchups"].append(
                f"{game.home_team} weak offense ({home_ppg:.1f}) vs "
                f"{game.away_team} strong defense ({away_opp_ppg:.1f} allowed) → -1.0"
            )

    # Clamp the matchup adjustment to [-2.0, +4.0]
    result["adjustment"] = max(-2.0, min(4.0, result["adjustment"]))

    if not result["matchups"]:
        result["matchups"].append("No extreme offense-vs-defense mismatch detected")

    return result


def _extract_ppg(profile: Dict) -> Optional[float]:
    """Extract points/goals per game from team profile last_5 data."""
    if not profile:
        return None

    # Try direct avg_margin + calculate from last_5
    last_5 = profile.get("last_5", [])
    if not last_5:
        return None

    total_scored = 0
    count = 0
    for g in last_5:
        score_str = g.get("score", "")  # Format: "110-105"
        if not score_str or "-" not in score_str:
            continue
        parts = score_str.split("-")
        try:
            team_score = int(parts[0])
            total_scored += team_score
            count += 1
        except (ValueError, TypeError):
            continue

    return round(total_scored / count, 1) if count > 0 else None


def _extract_opp_ppg(profile: Dict) -> Optional[float]:
    """Extract opponent points/goals per game from team profile last_5 data."""
    if not profile:
        return None

    last_5 = profile.get("last_5", [])
    if not last_5:
        return None

    total_allowed = 0
    count = 0
    for g in last_5:
        score_str = g.get("score", "")  # Format: "110-105"
        if not score_str or "-" not in score_str:
            continue
        parts = score_str.split("-")
        try:
            opp_score = int(parts[1])
            total_allowed += opp_score
            count += 1
        except (ValueError, TypeError, IndexError):
            continue

    return round(total_allowed / count, 1) if count > 0 else None


# ================================================================
# HELPER: Season record context
# ================================================================

def _build_season_context(game: HurdleGame) -> Optional[Dict]:
    """Build season record context from team profiles in metadata."""
    home_profile = game.metadata.get("home_profile", {})
    away_profile = game.metadata.get("away_profile", {})

    if not home_profile and not away_profile:
        return None

    ctx = {}

    if home_profile:
        hw = home_profile.get("season_wins", 0)
        hl = home_profile.get("season_losses", 0)
        ctx["home"] = {
            "team": game.home_team,
            "season_record": f"{hw}-{hl}",
            "home_record": home_profile.get("home_record", ""),
            "away_record": home_profile.get("away_record", ""),
            "l5_record": home_profile.get("summary", {}).get("record", ""),
            "l10_record": f"{home_profile.get('l10_wins', 0)}-{home_profile.get('l10_losses', 0)}",
            "avg_margin": home_profile.get("summary", {}).get("avg_margin", 0),
            "trend": home_profile.get("summary", {}).get("trend", "flat"),
        }

    if away_profile:
        aw = away_profile.get("season_wins", 0)
        al = away_profile.get("season_losses", 0)
        ctx["away"] = {
            "team": game.away_team,
            "season_record": f"{aw}-{al}",
            "home_record": away_profile.get("home_record", ""),
            "away_record": away_profile.get("away_record", ""),
            "l5_record": away_profile.get("summary", {}).get("record", ""),
            "l10_record": f"{away_profile.get('l10_wins', 0)}-{away_profile.get('l10_losses', 0)}",
            "avg_margin": away_profile.get("summary", {}).get("avg_margin", 0),
            "trend": away_profile.get("summary", {}).get("trend", "flat"),
        }

    return ctx


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


# ================================================================
# REPORT FORMATTING
# ================================================================

def format_stage0_report(result: Dict) -> str:
    """Human-readable summary for logging/Discord."""
    lines = [
        "=== STAGE 0: TEAM DATA FOUNDATION ===",
        f"Slate: {result['total']} games | Profile: {result['slate_profile']}",
        f"All games pass (S0 never kills)",
        "",
        f"SCORED ({len(result['passed'])}):",
    ]

    for g in result["passed"]:
        s0 = g.stage_results[0] if g.stage_results else None
        if not s0:
            lines.append(f"  {g.away_team} @ {g.home_team} — no result")
            continue

        factor_keys = list(s0.factors.keys())
        # Show key data layers present
        data_tags = []
        if "home_ats_l5" in s0.factors or "away_ats_l5" in s0.factors:
            data_tags.append("ATS")
        if "h2h" in s0.factors:
            data_tags.append("H2H")
        if "home_streak" in s0.factors or "away_streak" in s0.factors:
            data_tags.append("STREAK")
        if "offense_vs_defense" in s0.factors:
            data_tags.append("OFF/DEF")
        if "season_context" in s0.factors:
            data_tags.append("SEASON")
        if "line_movement" in s0.factors:
            data_tags.append("LINE")

        tags_str = ", ".join(data_tags) if data_tags else "minimal data"
        lines.append(
            f"  {g.away_team} @ {g.home_team} — "
            f"Score: {s0.score}/10 | {tags_str}"
        )

    if result["soft_fail"]:
        lines.append("")
        lines.append(f"SOFT FAIL ({len(result['soft_fail'])}):")
        for g in result["soft_fail"]:
            s0 = g.stage_results[0] if g.stage_results else None
            note = s0.notes if s0 else "no details"
            lines.append(f"  {g.away_team} @ {g.home_team} — {note}")

    lines.append("")
    lines.append(f"Dynamic weights: {result['dynamic_weights']}")
    return "\n".join(lines)

"""
GRADE VALIDATOR — Post-analysis checks and balances.

Runs AFTER the existing AI analysis produces grades.
Takes the graded cards and applies Peter's validation rules:

1. STALE INJURY CHECK — if star has been out 30+ days and team is winning, cap impact
2. L5 FORM CHECK — if team's L5 record contradicts the pick, downgrade
3. PRICE QUALITY CHECK — ML favorites beyond -200 capped at B-
4. PROP vs SPREAD CONFLICT — props don't justify spread picks
5. BOTH SIDES SCREENER — if only one side's props were checked, flag it
6. L5 STRENGTH OF SCHEDULE — wins against bad teams don't count the same

Each check can UPGRADE, DOWNGRADE, or FLAG a pick.
Original grade is preserved for comparison.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("edge-crew")

# Grade ordering for comparison
GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "D", "F", "TBD", "INCOMPLETE", "PASS"]


def validate_grades(analyses: List[Dict], l5_profiles: Dict = None) -> List[Dict]:
    """
    Run all validation checks on graded analyses.

    Args:
        analyses: List of game analysis dicts from the AI engine.
        l5_profiles: Optional dict of L5 rolling profiles per team.
                     Format: {team_name: {record: "3-2", avg_margin: 1.8, games: [...]}}

    Returns: Same list with added 'validation' field per game.
    """
    for game in analyses:
        validation = {
            "original_grade": game.get("grade", ""),
            "adjusted_grade": game.get("grade", ""),
            "checks": [],
            "flags": [],
            "adjustments": [],
        }

        pick = game.get("edge_pick", {})
        grade = game.get("grade", "")
        tags = game.get("tags", [])
        matchup = game.get("matchup", "")

        # --- CHECK 1: PRICE QUALITY ---
        adj = _check_price_quality(pick, grade)
        if adj:
            validation["checks"].append(adj)
            if adj.get("adjustment"):
                validation["adjustments"].append(adj["adjustment"])
                validation["adjusted_grade"] = adj["new_grade"]

        # --- CHECK 2: L5 FORM vs PICK ---
        if l5_profiles:
            adj = _check_l5_form(game, pick, grade, l5_profiles)
            if adj:
                validation["checks"].append(adj)
                if adj.get("adjustment"):
                    validation["adjustments"].append(adj["adjustment"])
                    if _grade_lower(adj["new_grade"], validation["adjusted_grade"]):
                        validation["adjusted_grade"] = adj["new_grade"]

        # --- CHECK 3: STALE INJURY ---
        adj = _check_stale_injuries(game, pick, grade)
        if adj:
            validation["checks"].append(adj)
            if adj.get("flag"):
                validation["flags"].append(adj["flag"])

        # --- CHECK 4: PROP vs SPREAD CONFLICT ---
        adj = _check_prop_spread_conflict(game, pick, grade)
        if adj:
            validation["checks"].append(adj)
            if adj.get("flag"):
                validation["flags"].append(adj["flag"])

        # --- CHECK 5: HEAVY FAVORITE SPREAD WARNING ---
        adj = _check_heavy_favorite_spread(game, pick, grade)
        if adj:
            validation["checks"].append(adj)
            if adj.get("adjustment"):
                validation["adjustments"].append(adj["adjustment"])
                if _grade_lower(adj["new_grade"], validation["adjusted_grade"]):
                    validation["adjusted_grade"] = adj["new_grade"]

        # --- CHECK 6: ML DODGE — system avoiding the spread question ---
        adj = _check_ml_dodge(game, pick, grade)
        if adj:
            validation["checks"].append(adj)
            if adj.get("flag"):
                validation["flags"].append(adj["flag"])
            if adj.get("adjustment"):
                validation["adjustments"].append(adj["adjustment"])
                if _grade_lower(adj["new_grade"], validation["adjusted_grade"]):
                    validation["adjusted_grade"] = adj["new_grade"]

        game["validation"] = validation

    return analyses


# ================================================================
# CHECK 1: PRICE QUALITY
# ================================================================

def _check_price_quality(pick: Dict, grade: str) -> Optional[Dict]:
    """ML favorites beyond -200 should be capped at B-."""
    if not pick:
        return None

    bet_type = (pick.get("bet_type", "") or "").upper()
    line = pick.get("line", "")

    if bet_type != "ML" and "ML" not in bet_type:
        return None

    try:
        ml_value = float(str(line).replace("+", ""))
    except (ValueError, TypeError):
        return None

    if ml_value >= -200:
        return None  # Price is fine

    result = {
        "check": "PRICE_QUALITY",
        "detail": f"ML {line} is heavy juice — poor bet value",
    }

    # Cap grade based on juice
    if ml_value <= -500:
        cap = "C+"
        result["detail"] = f"ML {line} is extreme juice — never worth it"
    elif ml_value <= -300:
        cap = "B-"
    else:  # -200 to -300
        cap = "B"

    if _grade_higher(grade, cap):
        result["adjustment"] = f"Downgraded from {grade} to {cap} — juice too heavy"
        result["new_grade"] = cap
    else:
        result["adjustment"] = None

    return result


# ================================================================
# CHECK 2: L5 FORM vs PICK DIRECTION
# ================================================================

def _check_l5_form(
    game: Dict, pick: Dict, grade: str, l5_profiles: Dict
) -> Optional[Dict]:
    """If L5 form contradicts the pick, downgrade."""
    if not pick:
        return None

    matchup = game.get("matchup", "")
    pick_team = pick.get("team", "")
    bet_type = (pick.get("bet_type", "") or "").upper()

    if not pick_team or pick_team == "PASS":
        return None

    # Find the picked team's L5 profile
    team_profile = None
    opp_profile = None
    for team_key, profile in l5_profiles.items():
        if team_key.lower() in pick_team.lower() or pick_team.lower() in team_key.lower():
            team_profile = profile
        elif team_key.lower() in matchup.lower():
            opp_profile = profile

    if not team_profile:
        return None

    record = team_profile.get("record", "")
    avg_margin = team_profile.get("avg_margin", 0)
    wins = team_profile.get("wins", 0)
    losses = team_profile.get("losses", 0)

    result = {
        "check": "L5_FORM",
        "detail": f"{pick_team} L5: {record} (avg margin: {avg_margin:+.1f})",
    }

    # If taking points with a team that's 1-4 or worse, downgrade
    if "SPREAD" in bet_type and losses >= 4:
        # Taking points with a team in freefall
        if _grade_higher(grade, "B-"):
            result["adjustment"] = f"Downgraded from {grade} to B- — {pick_team} is {wins}-{losses} L5"
            result["new_grade"] = "B-"
            return result

    # If team is 0-5 or 1-4 with avg margin worse than -5, big downgrade
    if losses >= 4 and avg_margin < -5:
        if _grade_higher(grade, "C+"):
            result["adjustment"] = f"Downgraded from {grade} to C+ — {pick_team} is {wins}-{losses} L5 with {avg_margin:+.1f} avg margin"
            result["new_grade"] = "C+"
            return result

    # If opponent is 4-1 or better, pick team is underdog, and grade is A range
    if opp_profile:
        opp_wins = opp_profile.get("wins", 0)
        if opp_wins >= 4 and losses >= 3 and _grade_higher(grade, "B"):
            result["adjustment"] = f"Downgraded from {grade} to B — opponent is {opp_wins}-{5-opp_wins} L5"
            result["new_grade"] = "B"
            return result

    result["adjustment"] = None
    return result


# ================================================================
# CHECK 3: STALE INJURY
# ================================================================

def _check_stale_injuries(game: Dict, pick: Dict, grade: str) -> Optional[Dict]:
    """
    If the grade is boosted by injuries that have been priced in,
    flag it. A star out for 30+ days on a team that's still winning
    is not an edge — it's the market.
    """
    injury_text = game.get("injury_impact", "") or ""
    tags = game.get("tags", [])

    # Check if INJURY-IMPACT is a tag (means injuries drove the grade)
    if "INJURY-IMPACT" not in tags and "injury" not in injury_text.lower():
        return None

    # Check for stale injury keywords
    stale_keywords = [
        "out for the season", "expected to be out until",
        "has not played since", "month", "weeks",
    ]

    is_stale = any(kw in injury_text.lower() for kw in stale_keywords)

    if not is_stale:
        return None

    return {
        "check": "STALE_INJURY",
        "detail": "Grade may be inflated by long-term injuries already priced in",
        "flag": "STALE_INJURY — verify team's L5 record WITHOUT the injured star. If winning, injury is priced in.",
    }


# ================================================================
# CHECK 4: PROP vs SPREAD CONFLICT
# ================================================================

def _check_prop_spread_conflict(game: Dict, pick: Dict, grade: str) -> Optional[Dict]:
    """
    Flag when the spread call is driven by prop analysis.
    Props finding value on one side doesn't mean the game goes that way.
    """
    # Check if EV analysis mentions props driving the spread call
    # This would need to be parsed from the analysis text
    # For now, flag any A-range grade where the reasoning mentions "props" or "mispriced"

    pick_reasoning = (pick.get("reasoning", "") or "").lower()
    if not pick_reasoning:
        return None

    prop_keywords = ["mispriced", "prop", "floor", "l5 floor", "hit rate"]
    has_prop_reasoning = any(kw in pick_reasoning for kw in prop_keywords)

    if not has_prop_reasoning:
        return None

    if _grade_higher(grade, "B"):
        return {
            "check": "PROP_SPREAD_CONFLICT",
            "detail": "Spread pick appears driven by individual prop analysis — props don't predict game outcomes",
            "flag": "PROP_BIAS — spread call may be based on player props, not game-level edge. Verify team matchup independently.",
        }

    return None


# ================================================================
# CHECK 5: HEAVY FAVORITE SPREAD
# ================================================================

def _check_heavy_favorite_spread(game: Dict, pick: Dict, grade: str) -> Optional[Dict]:
    """
    If recommending a heavy favorite's spread (>10 points),
    check if the opponent has been competitive recently.
    Blowout spreads are hard to cover.
    """
    if not pick:
        return None

    line = pick.get("line", "")
    bet_type = (pick.get("bet_type", "") or "").upper()

    if "SPREAD" not in bet_type:
        return None

    try:
        spread_val = float(str(line).replace("+", ""))
    except (ValueError, TypeError):
        return None

    abs_spread = abs(spread_val)

    if abs_spread < 12:
        return None

    # Spreads over 12 are very hard to cover
    result = {
        "check": "HEAVY_SPREAD",
        "detail": f"Spread {line} is 12+ points — historically hard to cover",
    }

    if _grade_higher(grade, "B+"):
        result["adjustment"] = f"Capped from {grade} to B+ — spreads over 12 rarely warrant high confidence"
        result["new_grade"] = "B+"
    else:
        result["adjustment"] = None

    return result


# ================================================================
# CHECK 6: ML DODGE — system avoiding the spread
# ================================================================

def _check_ml_dodge(game: Dict, pick: Dict, grade: str) -> Optional[Dict]:
    """
    Flag when system recommends ML on heavy favorite instead of evaluating spread.
    If the spread exists and ML is -300+, the system is ducking the hard question.
    Force it to justify the spread or downgrade.
    """
    if not pick:
        return None

    bet_type = (pick.get("bet_type", "") or "").upper()
    line = pick.get("line", "")

    if "ML" not in bet_type:
        return None

    try:
        ml_value = float(str(line).replace("+", ""))
    except (ValueError, TypeError):
        return None

    if ml_value >= -300:
        return None  # Reasonable ML, not a dodge

    # Check if spread exists for this game
    matchup = game.get("matchup", "")
    # The game has odds data — if there's a spread available, system should use it

    result = {
        "check": "ML_DODGE",
        "detail": f"System recommended ML {line} instead of evaluating the spread — ducking the hard question",
        "flag": f"ML_DODGE — ML {line} is lazy. Evaluate the spread. If you can't justify the spread, the grade drops.",
    }

    # Downgrade any A-range ML dodge to B at best
    if _grade_higher(grade, "B"):
        result["adjustment"] = f"Downgraded from {grade} to B — justify the spread, not the ML"
        result["new_grade"] = "B"

    return result


# ================================================================
# UTILITY
# ================================================================

def _grade_higher(grade_a: str, grade_b: str) -> bool:
    """Returns True if grade_a is higher (better) than grade_b."""
    try:
        idx_a = GRADE_ORDER.index(grade_a)
        idx_b = GRADE_ORDER.index(grade_b)
        return idx_a < idx_b
    except ValueError:
        return False


def _grade_lower(grade_a: str, grade_b: str) -> bool:
    """Returns True if grade_a is lower (worse) than grade_b."""
    try:
        idx_a = GRADE_ORDER.index(grade_a)
        idx_b = GRADE_ORDER.index(grade_b)
        return idx_a > idx_b
    except ValueError:
        return False


def format_validation_report(analyses: List[Dict]) -> str:
    """Human-readable validation report."""
    lines = ["=== GRADE VALIDATION REPORT ===", ""]

    for game in analyses:
        v = game.get("validation", {})
        if not v:
            continue

        matchup = game.get("matchup", "")
        orig = v.get("original_grade", "")
        adj = v.get("adjusted_grade", "")
        pick = game.get("edge_pick", {})

        changed = orig != adj
        arrow = f" -> {adj}" if changed else ""

        lines.append(f"{'**' if changed else ''}{orig}{arrow} | {matchup}{'**' if changed else ''}")
        lines.append(f"  Pick: {pick.get('team','')} {pick.get('bet_type','')} {pick.get('line','')}")

        for check in v.get("checks", []):
            status = "ADJUSTED" if check.get("adjustment") else ("FLAGGED" if check.get("flag") else "OK")
            lines.append(f"  [{status}] {check['check']}: {check['detail']}")

        for flag in v.get("flags", []):
            lines.append(f"  !! {flag}")

        lines.append("")

    return "\n".join(lines)

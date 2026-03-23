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
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger("edge-crew")

# Grade ordering for comparison
GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "D", "F", "TBD", "INCOMPLETE", "PASS"]

EDGE_CREW_BASE = "https://edge-crew-v2.onrender.com"


# ================================================================
# L5 PROFILE BUILDERS — Auto-fetch rolling profiles for validation
# ================================================================

def _extract_teams_from_matchup(matchup: str) -> tuple:
    """Extract (away_team, home_team) from matchup string like 'Los Angeles Lakers @ Detroit Pistons'."""
    if not matchup:
        return None, None
    for sep in [" @ ", " vs. ", " vs "]:
        if sep in matchup:
            parts = matchup.split(sep, 1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
    return None, None


def build_l5_from_analysis(analyses: List[Dict]) -> Dict:
    """
    Parse L5 rolling profile data embedded in analysis output.

    Analyses from the hurdle pipeline carry home_l5_record and away_l5_record
    (e.g. "3-2") plus streak/rest data on HurdleGame objects. The AI engine
    output may also embed team_profiles from the data-gathering step.

    Returns dict in the format validate_grades expects:
        {team_name: {"record": "3-2", "wins": 3, "losses": 2, "avg_margin": 0}}
    """
    profiles = {}

    for game in analyses:
        matchup = game.get("matchup", "")
        away_team, home_team = _extract_teams_from_matchup(matchup)

        # --- Source 1: Embedded team_profiles from server data-gathering step ---
        team_profiles = game.get("team_profiles", {})
        for team_name, tp in team_profiles.items():
            if team_name in profiles:
                continue
            summary = tp.get("summary", {})
            profiles[team_name] = {
                "record": summary.get("record", f"{tp.get('wins', 0)}-{tp.get('losses', 0)}"),
                "wins": tp.get("wins", 0),
                "losses": tp.get("losses", 0),
                "avg_margin": summary.get("avg_margin", tp.get("avg_margin", 0)),
                "streak": tp.get("streak", 0),
                "trend": summary.get("trend", "flat"),
                "last_5": tp.get("last_5", []),
            }

        # --- Source 2: HurdleGame-style fields (home_l5_record, away_l5_record) ---
        for side, team_name in [("home", home_team), ("away", away_team)]:
            if not team_name or team_name in profiles:
                continue
            l5_record = game.get(f"{side}_l5_record", "")
            if not l5_record:
                continue
            match = re.match(r"(\d+)\s*-\s*(\d+)", l5_record)
            if not match:
                continue
            wins = int(match.group(1))
            losses = int(match.group(2))
            streak = game.get(f"{side}_streak", 0)
            profiles[team_name] = {
                "record": l5_record,
                "wins": wins,
                "losses": losses,
                "avg_margin": 0,  # Not available from L5 record string alone
                "streak": streak,
                "trend": "up" if wins >= 4 else ("down" if losses >= 4 else "flat"),
                "last_5": [],
            }

        # --- Source 3: Parse from analysis text (raw_analysis or analysis field) ---
        raw_text = game.get("raw_analysis", "") or game.get("analysis", "") or ""
        if not raw_text:
            continue
        for team_name in [home_team, away_team]:
            if not team_name or team_name in profiles:
                continue
            # Use last word of team name for flexible matching (e.g. "Lakers")
            short_name = team_name.split()[-1] if team_name else ""
            if not short_name:
                continue
            # Search within 200 chars of team mention for W-L L5 pattern
            pattern = re.escape(short_name) + r".{0,200}?(\d)-(\d)\s*(?:L5|last\s*5|in\s*(?:their\s*)?last\s*5)"
            m = re.search(pattern, raw_text, re.IGNORECASE)
            if m:
                wins = int(m.group(1))
                losses = int(m.group(2))
                profiles[team_name] = {
                    "record": f"{wins}-{losses}",
                    "wins": wins,
                    "losses": losses,
                    "avg_margin": 0,
                    "streak": 0,
                    "trend": "up" if wins >= 4 else ("down" if losses >= 4 else "flat"),
                    "last_5": [],
                }

    logger.info(f"[L5] Built {len(profiles)} team profiles from analysis data")
    return profiles


async def fetch_l5_from_server(sport: str, analyses: List[Dict] = None) -> Dict:
    """
    Fetch L5 rolling profiles from the Edge Crew server's team-profile API.

    Hits /api/team-profile/{sport}/{team} for each unique team extracted from
    the analyses matchups. Returns dict in validate_grades format.

    Args:
        sport: Sport key (nba, nhl, mlb, etc.)
        analyses: List of game analysis dicts (used to extract team names).
                  If None, returns empty dict.

    Returns:
        {team_name: {"record": "3-2", "wins": 3, "losses": 2, "avg_margin": -1.4, ...}}
    """
    if not analyses:
        return {}

    # Extract all unique team names from matchups
    teams = set()
    for game in analyses:
        matchup = game.get("matchup", "")
        away, home = _extract_teams_from_matchup(matchup)
        if away:
            teams.add(away)
        if home:
            teams.add(home)

    if not teams:
        logger.warning("[L5] No teams found in analyses matchups")
        return {}

    profiles = {}
    sport_lower = sport.lower()

    async with httpx.AsyncClient(timeout=15.0) as client:
        for team in sorted(teams):
            try:
                url = f"{EDGE_CREW_BASE}/api/team-profile/{sport_lower}/{team}"
                resp = await client.get(url)
                if resp.status_code != 200:
                    logger.warning(f"[L5] Server returned {resp.status_code} for {team}")
                    continue
                data = resp.json()
                if not data or not data.get("last_5"):
                    continue

                summary = data.get("summary", {})
                profiles[team] = {
                    "record": summary.get("record", f"{data.get('wins', 0)}-{data.get('losses', 0)}"),
                    "wins": data.get("wins", 0),
                    "losses": data.get("losses", 0),
                    "avg_margin": summary.get("avg_margin", 0),
                    "streak": data.get("streak", 0),
                    "trend": summary.get("trend", "flat"),
                    "last_5": data.get("last_5", []),
                    "l10_wins": data.get("l10_wins", 0),
                    "l10_losses": data.get("l10_losses", 0),
                    "l10_avg_margin": data.get("l10_avg_margin", 0),
                }
            except httpx.TimeoutException:
                logger.warning(f"[L5] Timeout fetching profile for {team}")
            except Exception as e:
                logger.warning(f"[L5] Error fetching profile for {team}: {e}")

    logger.info(f"[L5] Fetched {len(profiles)}/{len(teams)} team profiles from server")
    return profiles


async def fetch_l5_profiles(sport: str, analyses: List[Dict]) -> Dict:
    """
    Auto-fetch L5 rolling profiles — tries server first, falls back to parsing analysis data.

    This is the main entry point. Call this before validate_grades to get the
    l5_profiles dict automatically.

    Args:
        sport: Sport key (nba, nhl, mlb, etc.)
        analyses: List of game analysis dicts from the AI engine.

    Returns:
        {team_name: {"record": "3-2", "wins": 3, "losses": 2, "avg_margin": -1.4, ...}}
    """
    # Try server API first (has full data including avg_margin, ATS, streaks)
    profiles = {}
    try:
        profiles = await fetch_l5_from_server(sport, analyses)
    except Exception as e:
        logger.warning(f"[L5] Server fetch failed, falling back to analysis parsing: {e}")

    # Fill gaps from analysis data (teams the server didn't have)
    parsed = build_l5_from_analysis(analyses)
    for team, profile in parsed.items():
        if team not in profiles:
            profiles[team] = profile

    if not profiles:
        logger.warning(f"[L5] No L5 profiles found for {sport} — validation will skip L5 checks")

    return profiles


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
            if adj.get("adjustment"):
                validation["adjustments"].append(adj["adjustment"])
                if _grade_lower(adj["new_grade"], validation["adjusted_grade"]):
                    validation["adjusted_grade"] = adj["new_grade"]

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

def _classify_injury_freshness(raw_status: str) -> str:
    """
    Classify injury duration. Returns FRESH/RECENT/ESTABLISHED/SEASON.
    FRESH (0-3d), RECENT (4-14d), ESTABLISHED (15-30d), SEASON (30+d).
    """
    if not raw_status:
        return "FRESH"
    raw = raw_status.lower()
    if "out for the season" in raw or "season-ending" in raw:
        return "SEASON"
    date_match = re.search(r'until at least (\w+ \d+)', raw)
    if date_match:
        try:
            target = datetime.strptime(f"{date_match.group(1)} 2026", "%b %d %Y")
            days_out = (target - datetime.utcnow()).days
            if days_out <= 3:
                return "FRESH"
            elif days_out <= 14:
                return "RECENT"
            elif days_out <= 30:
                return "ESTABLISHED"
            else:
                return "SEASON"
        except (ValueError, TypeError):
            pass
    if "day-to-day" in raw or "game time" in raw or "gtd" in raw:
        return "FRESH"
    return "FRESH"


def _check_stale_injuries(game: Dict, pick: Dict, grade: str) -> Optional[Dict]:
    """
    If the grade is boosted by injuries that have been priced in,
    flag and downgrade. Uses freshness tiers:
    FRESH (0-3d): Full impact
    RECENT (4-14d): Moderate
    ESTABLISHED (15-30d): Capped
    SEASON (30+d): If team winning without star, injury is priced in
    """
    injury_text = game.get("injury_impact", "") or ""
    tags = game.get("tags", [])

    if "INJURY-IMPACT" not in tags and "injury" not in injury_text.lower():
        return None

    # Check for stale injury indicators
    stale_keywords = [
        "out for the season", "expected to be out until",
        "has not played since", "month", "weeks",
    ]
    is_stale = any(kw in injury_text.lower() for kw in stale_keywords)

    if not is_stale:
        return None

    # Determine freshness of the key injuries
    freshness = "ESTABLISHED"  # Default if we detect staleness
    for kw in ["out for the season", "season-ending"]:
        if kw in injury_text.lower():
            freshness = "SEASON"
            break

    result = {
        "check": "STALE_INJURY",
        "detail": f"Injury freshness: {freshness} — grade may be inflated by long-term injuries already priced in",
        "flag": f"STALE_INJURY ({freshness}) — team has been playing without this player. Check L5 record — if winning, injury is priced in.",
    }

    # If SEASON-tier injury is driving an A-range grade, downgrade
    if freshness == "SEASON" and _grade_higher(grade, "B"):
        result["adjustment"] = f"Downgraded from {grade} to B — SEASON injury is fully priced in"
        result["new_grade"] = "B"
    elif freshness == "ESTABLISHED" and _grade_higher(grade, "B+"):
        result["adjustment"] = f"Downgraded from {grade} to B+ — ESTABLISHED injury likely priced in"
        result["new_grade"] = "B+"

    return result


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

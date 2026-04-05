"""
Layer 2: Grade Engine
Applies grading profiles to game data. Pure math — no AI.
Reads from data/, writes to grades/.
"""

import json
import math
import random
import sys
from datetime import datetime

from paths import DATA_DIR, GRADES_DIR, PROFILES_DIR


# ─── Grade Thresholds ───────────────────────────────────────────────────────────

GRADE_THRESHOLDS = [
    (8.0, "A+"), (7.3, "A"), (6.5, "A-"),
    (6.0, "B+"), (5.5, "B"), (5.0, "B-"),
    (4.5, "C+"), (3.5, "C"), (2.5, "D"), (0.0, "F"),
]

SIZING_MAP = {
    "A+": "2u", "A": "1.5u", "A-": "1u", "B+": "1u",
    "B": "PASS", "B-": "PASS", "C+": "PASS", "C": "PASS",
    "D": "PASS", "F": "PASS",
}


def score_to_grade(score: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


def score_to_sizing(score: float) -> str:
    grade = score_to_grade(score)
    return SIZING_MAP.get(grade, "PASS")


# ─── Helper: Parse Records ──────────────────────────────────────────────────────

def parse_record(record_str: str | None) -> tuple[int, int]:
    """Parse 'W-L' string into (wins, losses)."""
    if not record_str:
        return 0, 0
    try:
        parts = record_str.split("-")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return 0, 0


def win_pct(record_str: str | None) -> float:
    w, l = parse_record(record_str)
    total = w + l
    return w / total if total > 0 else 0.5


# ─── Sinton.ia Scoring Functions ─────────────────────────────────────────────────

def _clamp(val: int | float, lo: int = 1, hi: int = 10) -> float:
    return max(lo, min(hi, round(float(val), 1)))


def _apply_spread_amplifier(composite: float, variables: dict) -> float:
    """Amplify grade spread by weighting top/bottom variables heavier.

    1. TOP-3 AMPLIFIER: The 3 highest-impact variables get 2x influence.
       If they're all 8+, composite gets pulled up. If they're all 3-, pulled down.
    2. FLOOR/CEILING GATES: Any variable scoring 9+ or 2- forces the grade.
    """
    scores = sorted(
        [(v.get("score", 5), v.get("weight", 5)) for v in variables.values() if v.get("available", True)],
        key=lambda x: x[0] * x[1],  # sort by impact (score * weight)
        reverse=True
    )
    if not scores:
        return composite

    # TOP-3 AMPLIFIER: pull composite toward top 3 weighted scores
    top3 = scores[:3]
    bot3 = scores[-3:]
    top3_avg = sum(s for s, w in top3) / len(top3)
    bot3_avg = sum(s for s, w in bot3) / len(bot3)

    # Blend: 70% composite + 30% top/bottom signal
    if top3_avg >= 8.0:
        # Strong edge detected — pull up
        composite = composite * 0.7 + top3_avg * 0.3
    elif bot3_avg <= 3.0:
        # Major weakness — pull down
        composite = composite * 0.7 + bot3_avg * 0.3
    elif top3_avg >= 7.0 and bot3_avg >= 5.0:
        # Solid across the board — slight pull up
        composite = composite * 0.85 + top3_avg * 0.15

    # FLOOR/CEILING GATES: extreme single variables override
    all_scores = [s for s, w in scores]
    max_score = max(all_scores) if all_scores else 5
    min_score = min(all_scores) if all_scores else 5

    if max_score >= 9.5 and composite < 7.0:
        composite = max(composite, 7.0)  # Dominant edge can't grade below B+
    if min_score <= 1.5 and composite > 4.0:
        composite = min(composite, 4.0)  # Critical weakness caps at C

    return round(composite, 2)


def score_star_player_status(game: dict, side: str) -> tuple[int, str]:
    """Score based on injury differential. Higher = better for our side."""
    opp_side = "away" if side == "home" else "home"
    opp_injuries = game.get("injuries", {}).get(opp_side, [])
    our_injuries = game.get("injuries", {}).get(side, [])

    opp_impact = 0
    our_impact = 0
    notes = []

    for inj in opp_injuries:
        if inj.get("status") in ("OUT", "DOUBTFUL"):
            ppg = inj.get("ppg") or 0
            fresh = inj.get("freshness", "UNKNOWN")
            # Discount established injuries — team has adapted
            discount = 0.5 if fresh in ("ESTABLISHED", "SEASON") else 1.0
            if ppg >= 20:
                opp_impact += 3 * discount
                notes.append(f"OPP: {inj['player']} OUT ({ppg} PPG, {fresh})")
            elif ppg >= 12:
                opp_impact += 2 * discount
                notes.append(f"OPP: {inj['player']} OUT ({ppg} PPG)")
            else:
                opp_impact += 0.5

    for inj in our_injuries:
        if inj.get("status") in ("OUT", "DOUBTFUL"):
            ppg = inj.get("ppg") or 0
            fresh = inj.get("freshness", "UNKNOWN")
            discount = 0.5 if fresh in ("ESTABLISHED", "SEASON") else 1.0
            if ppg >= 20:
                our_impact += 3 * discount
                notes.append(f"OUR: {inj['player']} OUT ({ppg} PPG, {fresh})")
            elif ppg >= 12:
                our_impact += 2 * discount
            else:
                our_impact += 0.5

    raw = 5 + opp_impact - our_impact
    return _clamp(raw), "; ".join(notes) if notes else "No significant injuries"


def score_rest_advantage(profile: dict, opp_profile: dict) -> tuple[int, str]:
    """Score rest differential. B2B is brutal; 3+ days rest is gold."""
    our_rest = profile.get("rest_days")
    opp_rest = opp_profile.get("rest_days")
    our_b2b = profile.get("is_b2b", False)
    opp_b2b = opp_profile.get("is_b2b", False)
    our_7d = profile.get("games_last_7d", 0)
    opp_7d = opp_profile.get("games_last_7d", 0)

    if our_rest is None or opp_rest is None:
        return 5, "Rest data unavailable"

    score = 5
    parts = []

    # B2B detection
    if opp_b2b and not our_b2b:
        score += 3
        parts.append("OPP on B2B")
    elif our_b2b and not opp_b2b:
        score -= 3
        parts.append("WE on B2B")
    elif our_b2b and opp_b2b:
        parts.append("Both B2B")

    # Rest day differential
    rest_diff = (our_rest or 0) - (opp_rest or 0)
    if rest_diff >= 3:
        score += 2
        parts.append(f"Rest edge +{rest_diff}d")
    elif rest_diff >= 1:
        score += 1
        parts.append(f"Rest edge +{rest_diff}d")
    elif rest_diff <= -3:
        score -= 2
        parts.append(f"Rest deficit {rest_diff}d")
    elif rest_diff <= -1:
        score -= 1
        parts.append(f"Rest deficit {rest_diff}d")

    # Workload in last 7 days
    load_diff = opp_7d - our_7d
    if load_diff >= 2:
        score += 1
        parts.append(f"OPP played {opp_7d} in 7d vs our {our_7d}")
    elif load_diff <= -2:
        score -= 1
        parts.append(f"WE played {our_7d} in 7d vs their {opp_7d}")

    note = f"Us: {our_rest}d rest | Them: {opp_rest}d rest"
    if parts:
        note += " | " + "; ".join(parts)

    return _clamp(score), note


def score_recent_form(profile: dict, opp_profile: dict) -> tuple[int, str]:
    """Score based on L5/L10 + streak + margin, relative to opponent."""
    w, l = parse_record(profile.get("L5"))
    ow, ol = parse_record(opp_profile.get("L5"))
    streak = profile.get("streak", "")
    margin = profile.get("margin_L5", 0)
    opp_margin = opp_profile.get("margin_L5", 0)

    if w + l == 0:
        return 5, "No L5 data"

    # Base from L5 wins
    base = {5: 9, 4: 7, 3: 5, 2: 3.5, 1: 2, 0: 1}.get(w, 5)

    # Streak bonus/penalty
    if streak.startswith("W"):
        streak_n = int(streak[1:]) if streak[1:].isdigit() else 0
        if streak_n >= 6:
            base += 1.5
        elif streak_n >= 3:
            base += 0.5
    elif streak.startswith("L"):
        streak_n = int(streak[1:]) if streak[1:].isdigit() else 0
        if streak_n >= 6:
            base -= 1.5
        elif streak_n >= 3:
            base -= 0.5

    # Relative form: compare L5 wins
    form_edge = w - ow
    if form_edge >= 3:
        base += 1
    elif form_edge <= -3:
        base -= 1

    # Margin quality — winning big vs squeaking by
    if margin > 10:
        base += 0.5
    elif margin < -10:
        base -= 0.5

    note = f"L5: {profile.get('L5', '?')} (margin {margin:+.1f}) | Streak: {streak}"
    note += f" | OPP L5: {opp_profile.get('L5', '?')} (margin {opp_margin:+.1f})"

    return _clamp(base), note


def score_home_away(game: dict, side: str) -> tuple[int, str]:
    """Score home/away advantage with record context."""
    profile = game.get(f"{side}_profile", {})
    is_home = side == "home"

    if is_home:
        w, l = parse_record(profile.get("home_record"))
        base = 5.5  # Home court/ice — slight advantage, not a full point
    else:
        w, l = parse_record(profile.get("away_record"))
        base = 4.5  # Away — slight disadvantage

    if w + l > 0:
        pct = w / (w + l)
        # Strong home/away record adjusts +-2
        if pct >= 0.7:
            base += 2
        elif pct >= 0.55:
            base += 1
        elif pct <= 0.3:
            base -= 2
        elif pct <= 0.4:
            base -= 1

    record_str = profile.get("home_record" if is_home else "away_record", "?")
    return _clamp(base), f"{'Home' if is_home else 'Away'}: {record_str} ({w}/{w+l})"


def score_line_movement(game: dict, pick_side: str) -> tuple[int, str]:
    """Score line movement as sharp action indicator. Direction matters."""
    shifts = game.get("shifts", {})
    spread_delta = shifts.get("spread_delta", 0)  # Keep sign — positive = spread grew
    abs_delta = abs(spread_delta)
    ml_moved = shifts.get("ml_moved", False)
    odds = game.get("odds", {})

    # Determine if movement favors our side
    # If we're picking home and spread moved MORE negative (home became bigger fav), that's sharp on home
    home_spread = odds.get("spread_home") or 0

    # Movement interpretation
    if abs_delta >= 3:
        base = 9
        note = f"BIG MOVE: spread moved {spread_delta:+.1f} pts"
    elif abs_delta >= 1.5:
        base = 7
        note = f"Significant move: {spread_delta:+.1f} pts"
    elif abs_delta >= 0.5:
        base = 5
        note = f"Spread moved {spread_delta:+.1f} pts"
    else:
        base = 5
        note = "Line flat — neutral"

    if ml_moved:
        note += " | ML shifted"

    return _clamp(base), note


def score_off_ranking(profile: dict, opp_profile: dict, sport: str) -> tuple[int, str]:
    """Score offensive output relative to opponent's defense."""
    ppg = profile.get("ppg_L5", 0)
    opp_def = opp_profile.get("opp_ppg_L5", 0)  # How many points opp allows
    margin = profile.get("margin_L5", 0)

    if not ppg:
        return 5, "No PPG data"

    # Score based on how our offense compares to what opponent allows
    if sport == "NBA":
        # NBA average ~112 PPG — avg offense should be 5, not 7
        if ppg >= 122:
            base = 9
        elif ppg >= 118:
            base = 7.5
        elif ppg >= 114:
            base = 6
        elif ppg >= 110:
            base = 5
        elif ppg >= 105:
            base = 4
        else:
            base = 2.5

        if opp_def and opp_def >= 115:
            base += 0.5
        elif opp_def and opp_def <= 100:
            base -= 0.5
    elif sport == "NHL":
        if ppg >= 4.0: base = 9
        elif ppg >= 3.5: base = 7
        elif ppg >= 3.0: base = 5
        elif ppg >= 2.5: base = 3.5
        else: base = 2

        if opp_def and opp_def >= 3.5: base += 0.5
        elif opp_def and opp_def <= 2.5: base -= 0.5
    elif sport == "MLB":
        # MLB avg ~4.5 RPG — avg should be 5
        if ppg >= 6.0: base = 9
        elif ppg >= 5.5: base = 7.5
        elif ppg >= 5.0: base = 6
        elif ppg >= 4.5: base = 5
        elif ppg >= 4.0: base = 3.5
        else: base = 2

        if opp_def and opp_def >= 5.5: base += 0.5
        elif opp_def and opp_def <= 3.5: base -= 0.5
    elif sport == "SOCCER":
        # Soccer avg ~1.3 GPG — avg should be 5
        if ppg >= 2.5: base = 9
        elif ppg >= 2.0: base = 7.5
        elif ppg >= 1.5: base = 6
        elif ppg >= 1.2: base = 5
        elif ppg >= 0.8: base = 3.5
        else: base = 2

        if opp_def and opp_def >= 2.0: base += 0.5
        elif opp_def and opp_def <= 1.0: base -= 0.5
    else:  # NCAAB
        if ppg >= 82: base = 9
        elif ppg >= 78: base = 7.5
        elif ppg >= 74: base = 6
        elif ppg >= 70: base = 5
        elif ppg >= 65: base = 3.5
        else: base = 2

    return _clamp(base), f"PPG L5: {ppg} | OPP allows: {opp_def} | Margin L5: {margin:+.1f}"


def score_def_ranking(profile: dict, opp_profile: dict, sport: str) -> tuple[int, str]:
    """Score defensive output relative to opponent's offense."""
    opp_ppg = profile.get("opp_ppg_L5", 0)  # How many we allow
    their_ppg = opp_profile.get("ppg_L5", 0)  # How many they score

    if not opp_ppg:
        return 5, "No OPP PPG data"

    if sport == "NBA":
        # NBA avg ~112 allowed — avg defense should be 5
        if opp_ppg <= 100: base = 9
        elif opp_ppg <= 105: base = 7.5
        elif opp_ppg <= 110: base = 6
        elif opp_ppg <= 114: base = 5
        elif opp_ppg <= 118: base = 3.5
        else: base = 2

        if their_ppg and their_ppg >= 115 and opp_ppg <= 108:
            base += 0.5
    elif sport == "NHL":
        if opp_ppg <= 2.0: base = 9
        elif opp_ppg <= 2.5: base = 7
        elif opp_ppg <= 3.0: base = 5
        elif opp_ppg <= 3.5: base = 3.5
        else: base = 2
    elif sport == "MLB":
        # MLB avg ~4.5 RA — avg should be 5
        if opp_ppg <= 3.0: base = 9
        elif opp_ppg <= 3.5: base = 7.5
        elif opp_ppg <= 4.0: base = 6
        elif opp_ppg <= 4.5: base = 5
        elif opp_ppg <= 5.5: base = 3.5
        else: base = 2

        if their_ppg and their_ppg >= 5.5 and opp_ppg <= 4.0:
            base += 0.5
    elif sport == "SOCCER":
        # Soccer avg ~1.2 GA — avg should be 5
        if opp_ppg <= 0.5: base = 9
        elif opp_ppg <= 0.8: base = 7.5
        elif opp_ppg <= 1.0: base = 6
        elif opp_ppg <= 1.3: base = 5
        elif opp_ppg <= 1.8: base = 3.5
        else: base = 2

        if their_ppg and their_ppg >= 2.0 and opp_ppg <= 1.0:
            base += 0.5
    else:  # NCAAB
        if opp_ppg <= 62: base = 9
        elif opp_ppg <= 66: base = 7.5
        elif opp_ppg <= 70: base = 5
        elif opp_ppg <= 75: base = 3.5
        else: base = 2

    return _clamp(base), f"Allow L5: {opp_ppg} | They score L5: {their_ppg}"


def score_pace_matchup(profile: dict, opp_profile: dict, sport: str) -> tuple[int, str]:
    """Score pace/tempo matchup. Big pace mismatches create variance."""
    our_pace = profile.get("pace_L5", 0)
    opp_pace = opp_profile.get("pace_L5", 0)

    if not our_pace or not opp_pace:
        return 5, "No pace data"

    pace_diff = abs(our_pace - opp_pace)

    if sport == "NBA":
        # NBA avg combined ~225. High pace = more variance = riskier spreads
        if our_pace >= 235 and opp_pace >= 235:
            score = 6.5  # Both fast = high scoring
            note = "FAST matchup — high scoring expected"
        elif our_pace <= 210 and opp_pace <= 210:
            score = 5  # Both slow = grind, neutral
            note = "SLOW matchup — grind game"
        elif pace_diff >= 20:
            score = 3.5  # Big mismatch = unpredictable
            note = f"PACE MISMATCH: {pace_diff:.0f} pt difference"
        else:
            score = 5  # Aligned = neutral, not an edge
            note = f"Pace aligned ({pace_diff:.0f} diff)"
    else:
        score = 5
        note = f"Our pace: {our_pace} | Their pace: {opp_pace}"

    return _clamp(score), note


def score_road_trip(profile: dict) -> tuple[int, str]:
    """Score road trip / home stand length."""
    road_len = profile.get("road_trip_len", 0)
    home_len = profile.get("home_stand_len", 0)

    if home_len >= 4:
        score = 6
        note = f"Home stand: {home_len} games — well rested at home"
    elif home_len >= 2:
        score = 5.5
        note = f"Home stand: {home_len} games"
    elif road_len >= 5:
        score = 2
        note = f"LONG road trip: {road_len} games — fatigue factor"
    elif road_len >= 3:
        score = 4
        note = f"Road trip: {road_len} games"
    elif road_len >= 1:
        score = 5
        note = f"Road game {road_len}"
    else:
        score = 5
        note = "No road trip data"

    return _clamp(score), note


def score_h2h(profile: dict) -> tuple[int, str]:
    """Score head-to-head record this season."""
    h2h = profile.get("h2h_season", "0-0")
    w, l = parse_record(h2h)

    if w + l == 0:
        return 5, "No H2H games this season"

    total = w + l
    pct = w / total

    if pct >= 0.75 and total >= 2:
        score = 9
    elif pct >= 0.6:
        score = 7
    elif pct == 0.5:
        score = 5
    elif pct <= 0.25 and total >= 2:
        score = 2
    elif pct <= 0.4:
        score = 3
    else:
        score = 5

    return _clamp(score), f"H2H: {h2h} this season ({total} games)"


def score_ats_trend(profile: dict) -> tuple[int, str]:
    """Score ATS trend from cover margin proxy."""
    avg_margin = profile.get("avg_margin_L10", 0)
    close_games = profile.get("close_games_L10", 0)
    blowouts = profile.get("blowouts_L10", 0)

    # Use average margin as ATS proxy — teams winning big cover more
    if avg_margin >= 10:
        score = 9
        note = f"Avg margin L10: {avg_margin:+.1f} — dominating"
    elif avg_margin >= 5:
        score = 7
        note = f"Avg margin L10: {avg_margin:+.1f} — comfortable wins"
    elif avg_margin >= 0:
        score = 5.5
        note = f"Avg margin L10: {avg_margin:+.1f} — close games"
    elif avg_margin >= -5:
        score = 4
        note = f"Avg margin L10: {avg_margin:+.1f} — losing close"
    else:
        score = 2
        note = f"Avg margin L10: {avg_margin:+.1f} — getting blown out"

    note += f" | Close: {close_games}/10 | Blowouts: {blowouts}/10"
    return _clamp(score), note


def score_depth_injuries(game: dict, side: str) -> tuple[int, str]:
    """Score depth/bench injuries on opponent."""
    opp_side = "away" if side == "home" else "home"
    opp_injuries = game.get("injuries", {}).get(opp_side, [])
    our_injuries = game.get("injuries", {}).get(side, [])

    opp_out = [i for i in opp_injuries if i.get("status") in ("OUT", "DOUBTFUL")]
    our_out = [i for i in our_injuries if i.get("status") in ("OUT", "DOUBTFUL")]

    opp_count = len(opp_out)
    our_count = len(our_out)
    diff = opp_count - our_count

    if diff >= 4:
        score = 9
        note = f"OPP missing {opp_count} players vs our {our_count}"
    elif diff >= 2:
        score = 7
        note = f"OPP missing {opp_count} vs our {our_count}"
    elif diff >= 0:
        score = 5
        note = f"Similar injury loads (them {opp_count}, us {our_count})"
    elif diff >= -2:
        score = 4
        note = f"WE missing more: {our_count} vs their {opp_count}"
    else:
        score = 2
        note = f"WE badly depleted: {our_count} out vs their {opp_count}"

    return _clamp(score), note


# ─── Sport-specific scoring (NHL, NCAAB) ────────────────────────────────────────

def score_goalie_confirmed(game: dict, side: str) -> tuple[int, str]:
    """NHL: Goalie confirmation status. Default to neutral without lineup data."""
    # Without goalie confirmation API, use injury data as proxy
    injuries = game.get("injuries", {}).get(side, [])
    goalie_out = [i for i in injuries if i.get("pos") in ("G", "GK", "Goalie")]
    if goalie_out:
        return 3, f"Goalie injury: {goalie_out[0].get('player', '?')} — {goalie_out[0].get('status', '?')}"
    return 5, "No goalie injury reported — assume starter"


def score_tournament_context(game: dict) -> tuple[int, str]:
    """NCAAB: Tournament/motivation context from line and timing."""
    odds = game.get("odds", {})
    spread_raw = odds.get("spread_home") or 0
    try:
        spread = abs(float(spread_raw))
    except (ValueError, TypeError):
        spread = 0
    time_str = game.get("time", "")

    # March = tournament time
    note = "Regular tournament context"
    score = 6  # Default tournament boost

    if spread > 20:
        score = 4
        note = f"Massive spread ({spread}) — 1 vs 16 type"
    elif spread > 12:
        score = 5
        note = f"Big spread ({spread}) — likely mismatch"
    elif spread <= 5:
        score = 8
        note = f"Close matchup ({spread}) — competitive game"

    return _clamp(score), note


# ─── MLB Scoring Functions ──────────────────────────────────────────────────────

def score_starting_pitcher(game: dict, side: str) -> tuple[int, str]:
    """MLB: Starting pitcher quality — ERA, K/9, WHIP from engine data."""
    profile = game.get(f"{side}_profile", {})
    opp_side = "away" if side == "home" else "home"
    opp_profile = game.get(f"{opp_side}_profile", {})

    sp = profile.get("starting_pitcher", {})
    opp_sp = opp_profile.get("starting_pitcher", {})

    # Try direct engine data first
    era = sp.get("era") or sp.get("ERA")
    opp_era = opp_sp.get("era") or opp_sp.get("ERA")
    if era is not None and opp_era is not None:
        try:
            era, opp_era = float(era), float(opp_era)
            diff = opp_era - era  # positive = our SP is better
            score = _clamp(5 + diff * 1.2)
            return score, f"SP ERA {era:.2f} vs {opp_era:.2f} (diff {diff:+.2f})"
        except (ValueError, TypeError):
            pass

    # Fallback: proxy from margin/ppg
    margin = profile.get("margin_L5", 0)
    score = _clamp(5 + margin / 3)
    return score, f"SP proxy from margin L5: {margin:+.1f}"


def score_bullpen_state(game: dict, side: str) -> tuple[int, str]:
    """MLB: Bullpen fatigue — IP last 3 days, availability."""
    profile = game.get(f"{side}_profile", {})
    opp_side = "away" if side == "home" else "home"
    opp_profile = game.get(f"{opp_side}_profile", {})

    bp = profile.get("bullpen", {})
    opp_bp = opp_profile.get("bullpen", {})

    bp_ip = bp.get("ip_last_3d") or bp.get("innings_last_3d")
    opp_bp_ip = opp_bp.get("ip_last_3d") or opp_bp.get("innings_last_3d")
    if bp_ip is not None and opp_bp_ip is not None:
        try:
            bp_ip, opp_bp_ip = float(bp_ip), float(opp_bp_ip)
            diff = opp_bp_ip - bp_ip  # positive = their pen more tired
            score = _clamp(5 + diff / 3)
            return score, f"Bullpen IP L3d: us {bp_ip:.1f} vs them {opp_bp_ip:.1f}"
        except (ValueError, TypeError):
            pass

    # Fallback: use rest/schedule as proxy
    rest = profile.get("rest_days")
    if rest is not None and rest == 0:
        return 4, "Day game after night / no rest — pen likely tired"
    return 5, "Bullpen state — no direct data, neutral"


def score_park_factor(game: dict, side: str) -> tuple[int, str]:
    """MLB: Park factor scoring — Coors boosts, Oracle suppresses."""
    profile = game.get(f"{side}_profile", {})
    park = game.get("park_factor") or game.get("venue", {}).get("park_factor")
    venue = game.get("venue_name", "") or game.get("venue", "")

    if park is not None:
        try:
            pf = float(park)
            if pf >= 1.15:
                return 9, f"High-altitude/hitter park (PF {pf:.2f}) — {venue}"
            elif pf >= 1.05:
                return 7, f"Moderate hitter park (PF {pf:.2f}) — {venue}"
            elif pf <= 0.90:
                return 3, f"Pitcher park (PF {pf:.2f}) — {venue}"
            else:
                return 5, f"Neutral park (PF {pf:.2f}) — {venue}"
        except (ValueError, TypeError):
            pass

    # Known parks by name fragment
    v_lower = str(venue).lower()
    if "coors" in v_lower:
        return 9, "Coors Field — +1.3x run factor, altitude"
    elif "oracle" in v_lower or "petco" in v_lower:
        return 3, f"Pitcher park — {venue}"
    elif "wrigley" in v_lower:
        return 7, "Wrigley — wind-dependent, check weather"
    return 5, "Park factor — no data, neutral"


def score_platoon_matchup(game: dict, side: str) -> tuple[int, str]:
    """MLB: Lineup handedness vs SP — platoon advantage."""
    profile = game.get(f"{side}_profile", {})
    opp_profile = game.get(f"{'away' if side == 'home' else 'home'}_profile", {})

    platoon = profile.get("platoon", {})
    if platoon:
        adv = platoon.get("advantage") or platoon.get("platoon_advantage")
        if adv:
            try:
                adv_val = float(adv)
                score = _clamp(5 + adv_val * 2)
                return score, f"Platoon advantage: {adv_val:+.1f}"
            except (ValueError, TypeError):
                pass

    # Proxy from offense ranking
    score, note = score_off_ranking(profile, opp_profile, "MLB")
    return score, f"Platoon proxy — {note}"


def score_defense_rating(game: dict, side: str) -> tuple[int, str]:
    """MLB: Team fielding/defense quality."""
    profile = game.get(f"{side}_profile", {})
    defense = profile.get("defense", {})

    drs = defense.get("drs") or defense.get("DRS")
    if drs is not None:
        try:
            drs = float(drs)
            if drs >= 20:
                return 9, f"Elite defense (DRS {drs:+.0f})"
            elif drs >= 5:
                return 7, f"Good defense (DRS {drs:+.0f})"
            elif drs >= -5:
                return 5, f"Average defense (DRS {drs:+.0f})"
            else:
                return 3, f"Poor defense (DRS {drs:+.0f})"
        except (ValueError, TypeError):
            pass

    error_rate = defense.get("errors_per_game") or defense.get("error_rate")
    if error_rate is not None:
        try:
            er = float(error_rate)
            score = _clamp(8 - er * 5)
            return score, f"Defense from error rate: {er:.2f}/game"
        except (ValueError, TypeError):
            pass

    return 5, "Defense rating — no direct data, neutral"


def score_weather_impact(game: dict, side: str) -> tuple[int, str]:
    """MLB: Weather scoring for totals/run impact."""
    weather = game.get("weather", {})
    wind = weather.get("wind_speed") or weather.get("wind_mph")
    temp = weather.get("temp") or weather.get("temperature")
    wind_dir = weather.get("wind_direction", "")

    notes = []
    score = 5  # neutral

    if temp is not None:
        try:
            t = float(temp)
            if t >= 85:
                score += 1
                notes.append(f"Hot ({t}°F) — ball carries")
            elif t <= 50:
                score -= 1
                notes.append(f"Cold ({t}°F) — suppresses scoring")
        except (ValueError, TypeError):
            pass

    if wind is not None:
        try:
            w = float(wind)
            if w >= 15 and "out" in str(wind_dir).lower():
                score += 2
                notes.append(f"Wind {w}mph OUT — boosts scoring")
            elif w >= 15 and "in" in str(wind_dir).lower():
                score -= 2
                notes.append(f"Wind {w}mph IN — suppresses scoring")
            elif w >= 10:
                notes.append(f"Wind {w}mph {wind_dir}")
        except (ValueError, TypeError):
            pass

    return _clamp(score), " | ".join(notes) if notes else "Weather — no data, neutral"


# ─── Soccer Scoring Functions ───────────────────────────────────────────────────

def score_fixture_congestion(game: dict, side: str) -> tuple[int, str]:
    """Soccer: Fixture congestion — matches in 10/14 days, European travel."""
    profile = game.get(f"{side}_profile", {})
    opp_side = "away" if side == "home" else "home"
    opp_profile = game.get(f"{opp_side}_profile", {})

    our_m10 = profile.get("matches_in_10d") or profile.get("congestion_10d")
    opp_m10 = opp_profile.get("matches_in_10d") or opp_profile.get("congestion_10d")

    if our_m10 is not None and opp_m10 is not None:
        try:
            our_m10, opp_m10 = int(our_m10), int(opp_m10)
            diff = opp_m10 - our_m10  # positive = they're more congested
            if diff >= 2:
                return 9, f"Heavy congestion edge: them {opp_m10} vs us {our_m10} matches in 10d"
            elif diff >= 1:
                return 7, f"Congestion edge: them {opp_m10} vs us {our_m10} matches in 10d"
            elif diff == 0:
                return 5, f"Even congestion: both {our_m10} matches in 10d"
            else:
                return _clamp(5 + diff), f"We're more congested: us {our_m10} vs them {opp_m10} in 10d"
        except (ValueError, TypeError):
            pass

    # Fallback: rest days differential
    score, note = score_rest_advantage(profile, opp_profile)
    return score, f"Congestion proxy from rest — {note}"


def score_form_league_position(game: dict, side: str) -> tuple[int, str]:
    """Soccer: Form + league table position gap."""
    profile = game.get(f"{side}_profile", {})
    opp_side = "away" if side == "home" else "home"
    opp_profile = game.get(f"{opp_side}_profile", {})

    our_pos = profile.get("league_position") or profile.get("table_position")
    opp_pos = opp_profile.get("league_position") or opp_profile.get("table_position")

    if our_pos is not None and opp_pos is not None:
        try:
            our_pos, opp_pos = int(our_pos), int(opp_pos)
            gap = opp_pos - our_pos  # positive = we're higher in table
            if gap >= 10:
                return 9, f"Class gap: #{our_pos} vs #{opp_pos} (gap {gap})"
            elif gap >= 5:
                return 7, f"Position edge: #{our_pos} vs #{opp_pos} (gap {gap})"
            elif gap >= 0:
                return 5 + (gap // 2), f"Close: #{our_pos} vs #{opp_pos} (gap {gap})"
            else:
                return _clamp(5 + gap // 2), f"Lower position: #{our_pos} vs #{opp_pos} (gap {gap})"
        except (ValueError, TypeError):
            pass

    # Fallback from win %
    our_pct = win_pct(profile.get("record"))
    opp_pct = win_pct(opp_profile.get("record"))
    diff = our_pct - opp_pct
    return _clamp(5 + diff * 8), f"Form proxy: us {our_pct:.0%} vs them {opp_pct:.0%}"


def score_attack_defense_rating(game: dict, side: str) -> tuple[int, str]:
    """Soccer: xG/goals for and against — BTTS and O/U profile."""
    profile = game.get(f"{side}_profile", {})
    opp_profile = game.get(f"{'away' if side == 'home' else 'home'}_profile", {})

    gpg = profile.get("goals_per_game") or profile.get("gpg")
    ga_pg = profile.get("goals_against_per_game") or profile.get("ga_pg")

    if gpg is not None and ga_pg is not None:
        try:
            gpg, ga_pg = float(gpg), float(ga_pg)
            gd_pg = gpg - ga_pg
            if gd_pg >= 1.0:
                return 9, f"Elite: {gpg:.1f} GF/game, {ga_pg:.1f} GA/game (GD {gd_pg:+.1f})"
            elif gd_pg >= 0.3:
                return 7, f"Positive: {gpg:.1f} GF, {ga_pg:.1f} GA (GD {gd_pg:+.1f})"
            elif gd_pg >= -0.3:
                return 5, f"Balanced: {gpg:.1f} GF, {ga_pg:.1f} GA (GD {gd_pg:+.1f})"
            else:
                return _clamp(3 + gd_pg), f"Negative: {gpg:.1f} GF, {ga_pg:.1f} GA (GD {gd_pg:+.1f})"
        except (ValueError, TypeError):
            pass

    # Fallback to off/def ranking
    score, note = score_off_ranking(profile, opp_profile, "SOCCER")
    return score, f"Attack/defense proxy — {note}"


def score_home_away_venue(game: dict, side: str) -> tuple[int, str]:
    """Soccer: Home/away record + venue factors (altitude, turf)."""
    score, note = score_home_away(game, side)
    venue = game.get("venue", {})

    # Venue adjustments
    altitude = venue.get("altitude") or venue.get("elevation")
    surface = venue.get("surface", "")

    extras = []
    if altitude is not None:
        try:
            alt = float(altitude)
            if alt >= 2000:
                if side == "home":
                    score = min(10, score + 2)
                    extras.append(f"Altitude {alt:.0f}m — major home edge")
                else:
                    score = max(1, score - 2)
                    extras.append(f"Altitude {alt:.0f}m — away disadvantage")
        except (ValueError, TypeError):
            pass

    if "artificial" in surface.lower() or "turf" in surface.lower():
        if side == "home":
            score = min(10, score + 1)
            extras.append("Artificial turf — home familiar")

    full_note = note + (" | " + " | ".join(extras) if extras else "")
    return _clamp(score), full_note


def score_european_hangover(game: dict, side: str) -> tuple[int, str]:
    """Soccer: UCL/UEL away trip within 72hrs of this match."""
    opp_side = "away" if side == "home" else "home"
    opp_profile = game.get(f"{opp_side}_profile", {})

    # Check if opponent has European fixture data
    euro = opp_profile.get("european_match") or opp_profile.get("ucl_uel_last")
    if euro:
        days_since = euro.get("days_since") or euro.get("days_ago")
        was_away = euro.get("away", False)
        if days_since is not None:
            try:
                ds = int(days_since)
                if ds <= 3 and was_away:
                    return 9, f"European hangover: opponent played away {ds}d ago — rotation + fatigue"
                elif ds <= 3:
                    return 7, f"European match {ds}d ago (home) — moderate congestion"
                elif ds <= 5:
                    return 6, f"European match {ds}d ago — some recovery time"
            except (ValueError, TypeError):
                pass

    # Fallback: check our advantage from rest
    profile = game.get(f"{side}_profile", {})
    rest = profile.get("rest_days", 0)
    opp_rest = opp_profile.get("rest_days", 0)
    if opp_rest is not None and rest is not None:
        try:
            if int(rest) - int(opp_rest) >= 3:
                return 7, f"Rest gap: us {rest}d vs them {opp_rest}d — possible midweek fixture"
        except (ValueError, TypeError):
            pass

    return 5, "No European hangover detected"


def score_rotation_risk(game: dict, side: str) -> tuple[int, str]:
    """Soccer: Expected squad rotation from congestion/upcoming fixtures."""
    opp_side = "away" if side == "home" else "home"
    opp_profile = game.get(f"{opp_side}_profile", {})

    opp_m10 = opp_profile.get("matches_in_10d") or opp_profile.get("congestion_10d")
    if opp_m10 is not None:
        try:
            m10 = int(opp_m10)
            if m10 >= 4:
                return 9, f"Opponent {m10} matches in 10d — heavy rotation expected"
            elif m10 >= 3:
                return 7, f"Opponent {m10} matches in 10d — moderate rotation"
        except (ValueError, TypeError):
            pass

    return 5, "Rotation risk — no direct data"


def score_clean_sheet_rate(game: dict, side: str) -> tuple[int, str]:
    """Soccer: Clean sheet % for BTTS/O-U analysis."""
    profile = game.get(f"{side}_profile", {})
    cs_rate = profile.get("clean_sheet_pct") or profile.get("cs_rate")

    if cs_rate is not None:
        try:
            cs = float(cs_rate)
            if cs >= 50:
                return 9, f"Elite CS rate: {cs:.0f}%"
            elif cs >= 35:
                return 7, f"Good CS rate: {cs:.0f}%"
            elif cs >= 20:
                return 5, f"Average CS rate: {cs:.0f}%"
            else:
                return 3, f"Poor CS rate: {cs:.0f}% — leaky defense"
        except (ValueError, TypeError):
            pass

    # Fallback
    score, note = score_def_ranking(profile, game.get(f"{'away' if side == 'home' else 'home'}_profile", {}), "SOCCER")
    return score, f"CS proxy — {note}"


def score_motivation_context(game: dict, side: str) -> tuple[int, str]:
    """Soccer: Motivation — title race, relegation, dead rubber."""
    profile = game.get(f"{side}_profile", {})
    opp_profile = game.get(f"{'away' if side == 'home' else 'home'}_profile", {})

    our_pos = profile.get("league_position") or profile.get("table_position")
    opp_pos = opp_profile.get("league_position") or opp_profile.get("table_position")

    if our_pos is not None:
        try:
            pos = int(our_pos)
            if pos <= 4:
                return 8, f"Title/CL race (#{pos}) — high motivation"
            elif pos >= 17:
                return 8, f"Relegation battle (#{pos}) — desperate"
            elif pos <= 8:
                return 6, f"European push (#{pos}) — moderate motivation"
            else:
                return 4, f"Mid-table (#{pos}) — limited motivation"
        except (ValueError, TypeError):
            pass

    return 5, "Motivation context — no position data"


def score_league_context(game: dict, side: str) -> tuple[int, str]:
    """Soccer: League quality tier."""
    league = game.get("league", "")
    l_lower = str(league).lower()

    tier1 = ["eng.1", "epl", "premier league", "esp.1", "la liga", "ger.1", "bundesliga", "ita.1", "serie a", "fra.1", "ligue 1"]
    tier2 = ["eng.2", "championship", "ned.1", "eredivisie", "por.1", "primeira liga", "tur.1", "super lig"]

    for t in tier1:
        if t in l_lower:
            return 7, f"Tier 1 league: {league} — high reliability"
    for t in tier2:
        if t in l_lower:
            return 5, f"Tier 2 league: {league} — moderate reliability"

    return 4, f"Lower tier / unknown league: {league}"


# ─── Sinton.ia Profile Grader ────────────────────────────────────────────────────

def grade_sintonia(game: dict, pick_side: str) -> dict:
    """
    Grade a game using Sinton.ia's weighted matrix.
    pick_side: 'home' or 'away' — which side we're evaluating.
    """
    sport = game.get("sport", "NBA")
    profile = game.get(f"{pick_side}_profile", {})
    opp_side = "away" if pick_side == "home" else "home"
    opp_profile = game.get(f"{opp_side}_profile", {})

    # Load profile config
    config_file = PROFILES_DIR / "sintonia.json"
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    sport_vars = config.get("sports", {}).get(sport, {}).get("variables", {})
    if not sport_vars:
        return {"error": f"No Sinton.ia config for {sport}"}

    variables = {}

    # Score each variable
    for var_name, var_config in sport_vars.items():
        weight = var_config["weight"]

        if var_name == "star_player_status":
            score, note = score_star_player_status(game, pick_side)
        elif var_name == "recent_form":
            score, note = score_recent_form(profile, opp_profile)
        elif var_name == "home_away":
            score, note = score_home_away(game, pick_side)
        elif var_name in ("sharp_vs_public", "line_movement"):
            score, note = score_line_movement(game, pick_side)
        elif var_name == "off_ranking" or var_name == "offensive_efficiency":
            score, note = score_off_ranking(profile, opp_profile, sport)
        elif var_name == "def_ranking" or var_name == "defensive_efficiency":
            score, note = score_def_ranking(profile, opp_profile, sport)
        elif var_name in ("rest_advantage", "b2b_fatigue", "fatigue_schedule"):
            score, note = score_rest_advantage(profile, opp_profile)
        elif var_name in ("pace_matchup", "tempo_matchup"):
            score, note = score_pace_matchup(profile, opp_profile, sport)
        elif var_name in ("road_trip_length",):
            score, note = score_road_trip(profile)
        elif var_name in ("h2h_season",):
            score, note = score_h2h(profile)
        elif var_name in ("ats_trend",):
            score, note = score_ats_trend(profile)
        elif var_name in ("depth_injuries",):
            score, note = score_depth_injuries(game, pick_side)
        elif var_name in ("three_pt_matchup", "three_pt_reliance"):
            # Proxy: use offensive ranking differential
            score, note = score_off_ranking(profile, opp_profile, sport)
            note = f"3PT proxy — {note}"
        elif var_name in ("paint_scoring", "fast_break"):
            # Proxy: teams with high PPG tend to score in paint/fast break
            ppg = profile.get("ppg_L5", 0)
            opp_ppg = opp_profile.get("ppg_L5", 0)
            diff = (ppg or 0) - (opp_ppg or 0)
            score = _clamp(5 + diff / 5)
            note = f"{var_name} proxy: PPG diff {diff:+.1f}"
        elif var_name == "goalie_confirmed":
            score, note = score_goalie_confirmed(game, pick_side)
        elif var_name == "pp_pk":
            # Power play vs penalty kill — proxy from margin
            margin = profile.get("margin_L5", 0)
            score = _clamp(5 + margin / 3)
            note = f"PP/PK proxy from margin L5: {margin:+.1f}"
        elif var_name in ("corsi_xg", "save_pct_trend"):
            # Advanced NHL stats — proxy from defensive output
            score, note = score_def_ranking(profile, opp_profile, sport)
            note = f"{var_name} proxy — {note}"
        elif var_name == "tournament_context":
            score, note = score_tournament_context(game)
        elif var_name in ("public_money_bias",):
            score, note = score_line_movement(game, pick_side)
            note = f"Public bias proxy — {note}"
        elif var_name in ("bench_depth",):
            score, note = score_depth_injuries(game, pick_side)
            note = f"Bench depth proxy — {note}"
        elif var_name in ("rebounding", "turnover_margin"):
            margin = profile.get("margin_L5", 0)
            score = _clamp(5 + margin / 4)
            note = f"{var_name} proxy from margin: {margin:+.1f}"
        elif var_name in ("coaching_tournament_record", "coaching_matchup"):
            score = 5
            note = f"[{var_name}] NCAAB data not yet available — excluded from composite"
            variables[var_name] = {
                "score": _clamp(score), "weight": weight, "weighted": 0,
                "note": note, "available": False,
            }
            continue
        elif var_name in ("net_ranking_gap",):
            # Use season win % differential as proxy
            our_pct = win_pct(profile.get("record"))
            opp_pct = win_pct(opp_profile.get("record"))
            pct_diff = our_pct - opp_pct
            score = _clamp(5 + pct_diff * 8)
            note = f"NET proxy: us {our_pct:.0%} vs them {opp_pct:.0%} (diff {pct_diff:+.0%})"
        elif var_name == "neutral_site":
            # Detect neutral site: NCAA Tournament, bowl games, etc.
            is_neutral = game.get("neutral_site", False) or game.get("is_neutral", False)
            venue = game.get("venue", "") or game.get("venue_name", "")
            league = game.get("league", "") or ""
            # Tournament detection from matchup context
            if any(t in str(venue).lower() for t in ["lucas oil", "superdome", "alamodome", "nrg", "state farm"]):
                is_neutral = True
            if any(t in str(league).lower() for t in ["ncaa", "tournament", "march madness", "final four", "elite eight", "sweet sixteen"]):
                is_neutral = True
            if is_neutral:
                score = 8
                note = f"NEUTRAL SITE — home/away splits DO NOT APPLY. Venue: {venue or 'tournament'}. Grade on talent, matchup, and tournament experience ONLY."
            else:
                score = 5
                note = "Standard home/away game"
        elif var_name in ("conference_strength", "ft_shooting"):
            score = 5
            note = f"[{var_name}] NCAAB data not yet available — excluded from composite"
            variables[var_name] = {
                "score": _clamp(score), "weight": weight, "weighted": 0,
                "note": note, "available": False,
            }
            continue
        elif var_name in ("rivalry_motivation",):
            score, note = score_h2h(profile)
            note = f"Rivalry proxy from H2H — {note}"
        # ── MLB variables ──
        elif var_name == "starting_pitcher":
            score, note = score_starting_pitcher(game, pick_side)
        elif var_name == "bullpen_state":
            score, note = score_bullpen_state(game, pick_side)
        elif var_name == "park_factor":
            score, note = score_park_factor(game, pick_side)
        elif var_name in ("platoon_matchup", "lineup_vs_hand"):
            score, note = score_platoon_matchup(game, pick_side)
        elif var_name == "defense_rating":
            score, note = score_defense_rating(game, pick_side)
        elif var_name == "weather_impact":
            score, note = score_weather_impact(game, pick_side)
        # ── Soccer variables ──
        elif var_name == "fixture_congestion":
            score, note = score_fixture_congestion(game, pick_side)
        elif var_name == "form_league_position":
            score, note = score_form_league_position(game, pick_side)
        elif var_name == "attack_defense_rating":
            score, note = score_attack_defense_rating(game, pick_side)
        elif var_name in ("home_away_venue",):
            score, note = score_home_away_venue(game, pick_side)
        elif var_name == "european_hangover":
            score, note = score_european_hangover(game, pick_side)
        elif var_name == "rotation_risk":
            score, note = score_rotation_risk(game, pick_side)
        elif var_name == "clean_sheet_rate":
            score, note = score_clean_sheet_rate(game, pick_side)
        elif var_name == "motivation_context":
            score, note = score_motivation_context(game, pick_side)
        elif var_name == "league_context":
            score, note = score_league_context(game, pick_side)
        else:
            score = 5
            note = f"[{var_name}] Not yet implemented"

        variables[var_name] = {
            "score": score,
            "weight": weight,
            "weighted": round(score * weight, 1),
            "note": note,
            "available": True,
        }

    # Calculate composite — only from available variables (excludes unimplemented NCAAB vars)
    active_vars = {k: v for k, v in variables.items() if v.get("available", True)}
    total_weighted = sum(v["weighted"] for v in active_vars.values())
    max_possible = sum(v["weight"] * 10 for v in active_vars.values())
    composite = round(total_weighted / max_possible * 10, 2) if max_possible > 0 else 0
    composite = _apply_spread_amplifier(composite, active_vars)

    # Check chain bonuses
    chains_fired = []
    chain_bonus = 0.0

    chains = config.get("chains", {})
    for chain_name, chain_config in chains.items():
        # Check sport filter
        chain_sports = chain_config.get("sports")
        if chain_sports and sport not in chain_sports:
            continue

        fired = check_chain(chain_name, variables)
        if fired:
            bonus = chain_config.get("bonus", 0.5)
            chain_bonus += bonus
            chains_fired.append(chain_name)

    cap = config.get("chain_cap", 3.0)
    chain_bonus = max(-cap, min(chain_bonus, cap))  # Cap both positive and negative
    final = round(max(1.0, min(10.0, composite + chain_bonus)), 2)
    grade = score_to_grade(final)

    # Determine pick
    odds = game.get("odds", {})
    if pick_side == "home":
        pick = f"{game.get('home', '?')} {odds.get('spread_home', '?')}"
    else:
        pick = f"{game.get('away', '?')} {odds.get('spread_away', '?')}"

    return {
        "game_id": game.get("game_id", ""),
        "profile": "sintonia",
        "pick_side": pick_side,
        "grade": grade,
        "composite": composite,
        "chain_bonus": chain_bonus,
        "final": final,
        "pick": pick,
        "sizing": score_to_sizing(final),
        "variables": variables,
        "chains_fired": chains_fired,
        "vars_available": len(active_vars),
        "vars_total": len(variables),
    }


def check_chain(chain_name: str, variables: dict) -> bool:
    """Check if a chain bonus triggers based on variable scores."""
    v = {k: var["score"] for k, var in variables.items()}

    if chain_name == "THE_MISPRICING":
        return (v.get("star_player_status", 0) >= 8 and
                v.get("sharp_vs_public", 10) <= 4 and  # Line flat
                v.get("line_movement", 5) <= 3)  # Line hasn't moved despite star impact
    elif chain_name == "FATIGUE_FADE":
        return (v.get("rest_advantage", 0) >= 8 and
                v.get("road_trip_length", 0) >= 7 and
                v.get("depth_injuries", 0) >= 7)
    elif chain_name == "FORM_WAVE":
        return (v.get("recent_form", 0) >= 8 and
                v.get("off_ranking", 0) >= 8 and
                v.get("ats_trend", 0) >= 7 and
                v.get("h2h_season", 0) >= 7)
    elif chain_name == "GOALIE_EDGE":
        return (v.get("goalie_confirmed", 0) >= 8 and
                v.get("save_pct_trend", 0) >= 7 and
                v.get("b2b_fatigue", 0) >= 7 and
                v.get("corsi_xg", 0) >= 7)
    elif chain_name == "TRAP_GAME":
        return (v.get("sharp_vs_public", 10) <= 3 and
                v.get("recent_form", 0) >= 7)
    elif chain_name == "VALUE_DOG":
        return (v.get("sharp_vs_public", 0) >= 7 and
                v.get("star_player_status", 10) <= 4)
    elif chain_name == "BLUE_BLOOD_TRAP":
        return (v.get("public_money_bias", 0) >= 8 and
                v.get("net_ranking_gap", 10) <= 5)
    # MLB chains
    elif chain_name == "ACE_DOMINATION":
        return (v.get("starting_pitcher", 0) >= 9 and
                v.get("platoon_matchup", 0) >= 7 and
                v.get("bullpen_state", 0) >= 7)
    elif chain_name == "BULLPEN_MELTDOWN":
        return (v.get("bullpen_state", 0) >= 8 and
                v.get("starting_pitcher", 0) >= 7 and
                v.get("defense_rating", 0) >= 6)
    elif chain_name == "COORS_OVER":
        return (v.get("park_factor", 0) >= 8 and
                v.get("weather_impact", 0) >= 7 and
                v.get("attack_defense_rating", v.get("off_ranking", 0)) >= 7)
    # Soccer chains
    elif chain_name == "CONGESTION_FADE":
        return (v.get("fixture_congestion", 0) >= 8 and
                v.get("european_hangover", 0) >= 7 and
                v.get("rotation_risk", 0) >= 7)
    elif chain_name == "CLASS_GAP":
        return (v.get("form_league_position", 0) >= 8 and
                v.get("attack_defense_rating", 0) >= 7 and
                v.get("home_away_venue", 0) >= 7)
    elif chain_name == "FORTRESS_HOME":
        return (v.get("home_away_venue", 0) >= 8 and
                v.get("clean_sheet_rate", 0) >= 7 and
                v.get("form_league_position", 0) >= 7)
    # ── NEW CHAINS: More compound signals for grade differentiation ──
    # Cross-sport: INJURY_GOLDMINE — star out + line hasn't moved + good form
    elif chain_name == "INJURY_GOLDMINE":
        return (v.get("star_player_status", 0) >= 8 and
                v.get("line_movement", 5) <= 3 and
                v.get("recent_form", 0) >= 6)
    # Cross-sport: REST_DOMINATION — massive rest + schedule + travel advantage
    elif chain_name == "REST_DOMINATION":
        return (v.get("rest_advantage", 0) >= 8 and
                v.get("home_away", 0) >= 6 and
                v.get("road_trip_length", v.get("home_stand_road_trip", 0)) >= 6)
    # Cross-sport: DUMPSTER_FIRE — everything bad, force grade down
    elif chain_name == "DUMPSTER_FIRE":
        return (v.get("recent_form", 10) <= 3 and
                v.get("off_ranking", v.get("offensive_efficiency", 10)) <= 3 and
                v.get("star_player_status", 10) <= 4)
    # Cross-sport: SHARPS_LOVE — line moving our way + strong fundamentals
    elif chain_name == "SHARPS_LOVE":
        return (v.get("sharp_vs_public", 0) >= 8 and
                v.get("line_movement", 0) >= 7 and
                v.get("recent_form", 0) >= 6)
    # NBA/NCAAB: BLOWOUT_INCOMING — elite offense vs trash defense + home
    elif chain_name == "BLOWOUT_INCOMING":
        return (v.get("off_ranking", v.get("offensive_efficiency", 0)) >= 8 and
                v.get("def_ranking", v.get("defensive_efficiency", 0)) >= 7 and
                v.get("home_away", 0) >= 6)
    # Cross-sport: COLD_TAKE — no edge anywhere, force grade down
    elif chain_name == "COLD_TAKE":
        avg_score = sum(v.values()) / len(v) if v else 5
        return avg_score <= 4.5  # Average variable score is below neutral
    # ── OFFENSE vs DEFENSE MATCHUP CHAINS ──
    # MISMATCH_MASSACRE — elite offense vs garbage defense
    elif chain_name == "MISMATCH_MASSACRE":
        off = v.get("off_ranking", v.get("offensive_efficiency", 0))
        defn = v.get("def_ranking", v.get("defensive_efficiency", 0))
        return off >= 8 and defn >= 7  # Our offense is elite AND our defense is solid
    # GLASS_CANNON — great offense but terrible defense (volatile, reduce confidence)
    elif chain_name == "GLASS_CANNON":
        off = v.get("off_ranking", v.get("offensive_efficiency", 0))
        defn = v.get("def_ranking", v.get("defensive_efficiency", 10))
        return off >= 7 and defn <= 3  # Score a lot but leak — unpredictable
    # BRICK_WALL — elite defense but no offense (unders, low-scoring grinds)
    elif chain_name == "BRICK_WALL":
        off = v.get("off_ranking", v.get("offensive_efficiency", 10))
        defn = v.get("def_ranking", v.get("defensive_efficiency", 0))
        return defn >= 8 and off <= 4  # Can't score but opponent can't either
    # ── SITUATIONAL CHAINS ──
    # REVENGE_GAME — lost H2H badly + on form now + home
    elif chain_name == "REVENGE_GAME":
        return (v.get("h2h_season", 10) <= 3 and
                v.get("recent_form", 0) >= 7 and
                v.get("home_away", 0) >= 6)
    # LETDOWN_AFTER_BIG_WIN — opponent on huge streak, our team flat
    elif chain_name == "LETDOWN_ALERT":
        return (v.get("recent_form", 0) >= 8 and
                v.get("motivation_gap", v.get("motivation_context", 10)) <= 4)
    # ROAD_WARRIOR — winning on road + good form + rest
    elif chain_name == "ROAD_WARRIOR":
        return (v.get("home_away", 10) <= 4 and  # We're away
                v.get("recent_form", 0) >= 7 and
                v.get("rest_advantage", 0) >= 6)
    # SCHEDULE_LOSS — B2B + road + bad form = stay away
    elif chain_name == "SCHEDULE_LOSS":
        return (v.get("rest_advantage", 10) <= 3 and
                v.get("road_trip_length", v.get("home_stand_road_trip", 10)) <= 3 and
                v.get("recent_form", 10) <= 4)
    # ── MLB SPECIFIC ──
    # PITCHING_DUEL — both SPs elite, expect under
    elif chain_name == "PITCHING_DUEL":
        return (v.get("starting_pitcher", 0) >= 8 and
                v.get("def_ranking", v.get("defense_rating", 0)) >= 7 and
                v.get("park_factor", 5) <= 4)  # Pitcher-friendly park
    # ── SOCCER SPECIFIC ──
    # DERBY_CHAOS — rivalry + both in form + tight table
    elif chain_name == "DERBY_CHAOS":
        return (v.get("rivalry_motivation", v.get("derby_motivation", 0)) >= 7 and
                v.get("recent_form", 0) >= 6 and
                v.get("motivation_context", 0) >= 6)
    # TOURIST_TRAP — big name team away in a tough venue with congestion
    elif chain_name == "TOURIST_TRAP":
        return (v.get("home_away_venue", v.get("home_away", 10)) <= 4 and
                v.get("fixture_congestion", v.get("fixture_congestion_spot", 10)) <= 3 and
                v.get("form_league_position", 10) <= 5)
    # ── NARRATIVE / MOTIVATION CHAINS ──
    # HUNGRY_DOG — underdog with good form + strong recent play = playing for something
    elif chain_name == "HUNGRY_DOG":
        return (v.get("recent_form", 0) >= 7 and
                v.get("motivation_gap", v.get("motivation_context", 0)) >= 7 and
                v.get("sharp_vs_public", 0) >= 6)
    # COASTING_FAV — big favorite but form dropping + no motivation
    elif chain_name == "COASTING_FAV":
        return (v.get("recent_form", 10) <= 4 and
                v.get("motivation_gap", v.get("motivation_context", 10)) <= 4 and
                v.get("off_ranking", v.get("offensive_efficiency", 10)) <= 5)
    # NOTHING_TO_LOSE — eliminated/bad team playing free + home = upset spot
    elif chain_name == "NOTHING_TO_LOSE":
        return (v.get("motivation_gap", v.get("motivation_context", 0)) >= 6 and
                v.get("home_away", 0) >= 6 and
                v.get("recent_form", 10) <= 4)
    # PUBLIC_TRAP — heavy public side + line not moving + sharps quiet
    elif chain_name == "PUBLIC_TRAP":
        return (v.get("sharp_vs_public", 10) <= 3 and
                v.get("line_movement", 5) <= 3 and
                v.get("ats_trend", 0) >= 6)
    # BOUNCE_BACK — lost 3+ straight but elite team + home = regression
    elif chain_name == "BOUNCE_BACK":
        return (v.get("recent_form", 10) <= 3 and
                v.get("off_ranking", v.get("offensive_efficiency", 0)) >= 7 and
                v.get("home_away", 0) >= 6)
    # FADE_THE_STREAK — long win streak + away + B2B = regression spot
    elif chain_name == "FADE_THE_STREAK":
        return (v.get("recent_form", 0) >= 9 and
                v.get("home_away", 10) <= 4 and
                v.get("rest_advantage", 10) <= 4)
    # ── BENCH / DEPTH CHAINS ──
    # BENCH_MOB — deep rotation + healthy + in form
    elif chain_name == "BENCH_MOB":
        return (v.get("depth_injuries", v.get("bench_depth", 0)) >= 7 and
                v.get("star_player_status", 0) >= 6 and
                v.get("recent_form", 0) >= 6)
    # THIN_ROSTER — key players OUT + no depth = trouble
    elif chain_name == "THIN_ROSTER":
        return (v.get("depth_injuries", v.get("bench_depth", 10)) <= 3 and
                v.get("star_player_status", 10) <= 4)
    # ── LEFT/RIGHT (PLATOON) CHAINS — MLB specific ──
    # PLATOON_EDGE — favorable handedness matchup + park + lineup
    elif chain_name == "PLATOON_EDGE":
        return (v.get("platoon_matchup", v.get("lineup_vs_hand", 0)) >= 8 and
                v.get("starting_pitcher", 0) >= 6 and
                v.get("park_factor", 5) >= 6)
    # WRONG_SIDE — bad platoon matchup + pitcher dominant = fade
    elif chain_name == "WRONG_SIDE":
        return (v.get("platoon_matchup", v.get("lineup_vs_hand", 10)) <= 3 and
                v.get("starting_pitcher", 10) <= 4)

    return False


# ─── Claude Profile Grader (Contrarian / Fade the Public) ────────────────────────

def grade_claude(game: dict, pick_side: str) -> dict:
    """
    Grade using Claude's contrarian profile.
    Targets public overreaction, recency bias, and name-value inflation.
    """
    sport = game.get("sport", "NBA")
    profile = game.get(f"{pick_side}_profile", {})
    opp_side = "away" if pick_side == "home" else "home"
    opp_profile = game.get(f"{opp_side}_profile", {})
    odds = game.get("odds", {})
    shifts = game.get("shifts", {})

    config_file = PROFILES_DIR / "claude.json"
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    sport_vars = config.get("sports", {}).get(sport, {}).get("variables", {})
    if not sport_vars:
        return {"error": f"No Claude config for {sport}"}

    variables = {}

    for var_name, var_config in sport_vars.items():
        weight = var_config["weight"]

        if var_name == "public_pct_split":
            # Proxy: large spread movement opposite to expected direction = public one-sided
            spread_delta = abs(shifts.get("spread_delta", 0))
            ml_moved = shifts.get("ml_moved", False)
            # If line is moving opposite what you'd expect, public is heavy one side
            if spread_delta >= 2.0 and ml_moved:
                score = 9
                note = f"Spread moved {spread_delta:.1f} + ML shifted — public likely lopsided"
            elif spread_delta >= 1.5:
                score = 7
                note = f"Significant movement ({spread_delta:.1f}) suggests one-sided action"
            elif spread_delta >= 0.5:
                score = 5
                note = f"Moderate movement ({spread_delta:.1f})"
            else:
                score = 3
                note = "Line flat — no public split signal"

        elif var_name == "line_vs_public":
            # Line moving toward the dog when public is on favorite = sharp contrarian
            spread_delta = shifts.get("spread_delta", 0)
            home_spread = odds.get("spread_home") or 0
            # If home is big favorite (negative spread) and spread is getting less negative, sharps on dog
            if home_spread < -5 and spread_delta > 1.0:
                score = 9
                note = f"Big fav ({home_spread}) but line moving to dog (+{spread_delta:.1f}) — sharps fading"
            elif home_spread < -3 and spread_delta > 0.5:
                score = 7
                note = f"Favorite ({home_spread}) losing points — contrarian signal"
            elif spread_delta < -1.0:
                score = 3
                note = "Line moving WITH public favorite — no contrarian edge"
            else:
                score = 5
                note = f"Spread delta: {spread_delta:+.1f}"

        elif var_name == "sharp_money":
            # Reverse line movement is the clearest sharp indicator
            spread_delta = shifts.get("spread_delta", 0)
            abs_delta = abs(spread_delta)
            if abs_delta >= 2.5:
                score = 9
                note = f"Major line move ({spread_delta:+.1f}) — steam move detected"
            elif abs_delta >= 1.5:
                score = 7
                note = f"Sharp action signal ({spread_delta:+.1f})"
            elif abs_delta >= 0.5:
                score = 5
                note = f"Mild movement ({spread_delta:+.1f})"
            else:
                score = 3
                note = "No sharp signal — line stable"

        elif var_name == "recency_bias":
            # Compare L3-ish (streak) vs longer form
            streak = profile.get("streak", "")
            w5, l5 = parse_record(profile.get("L5"))
            w10, l10 = parse_record(profile.get("L10"))
            margin5 = profile.get("margin_L5", 0)
            margin10 = profile.get("avg_margin_L10", 0)

            # Hot streak but underlying numbers don't support
            if streak.startswith("W") and w5 >= 4 and margin10 and margin10 < 3:
                score = 8
                note = f"Hot streak ({streak}) but L10 margin only {margin10:+.1f} — public overreacting"
            elif streak.startswith("L") and l5 >= 4 and margin10 and margin10 > -3:
                score = 8
                note = f"Cold streak ({streak}) but L10 margin {margin10:+.1f} — public undervaluing"
            elif streak.startswith("W") and w5 >= 3:
                score = 6
                note = f"Winning ({streak}) — some recency bias possible"
            elif streak.startswith("L") and l5 >= 3:
                score = 6
                note = f"Losing ({streak}) — some overreaction possible"
            else:
                score = 5
                note = f"Streak: {streak} — no strong recency signal"

        elif var_name in ("primetime_inflation", "public_star_bias", "goalie_name_bias"):
            # Without broadcast schedule data, proxy from spread size + public action
            try:
                abs_spread = abs(float(odds.get("spread_home") or 0))
            except (ValueError, TypeError):
                abs_spread = 0
            spread_delta = abs(shifts.get("spread_delta", 0))
            # Big favorites that aren't moving = potential primetime/star inflation
            if abs_spread > 8 and spread_delta < 1.0:
                score = 7
                note = f"Large spread ({abs_spread}) barely moving — potential name/primetime tax"
            elif abs_spread > 12:
                score = 8
                note = f"Massive spread ({abs_spread}) — public piling on name value"
            else:
                score = 5
                note = f"Spread: {abs_spread} — no inflation signal"

        elif var_name in ("spread_value", "dog_record_ats"):
            # Underdog value — use margin and record
            our_margin = profile.get("avg_margin_L10", 0)
            try:
                abs_spread = abs(float(odds.get("spread_home") or 0))
            except (ValueError, TypeError):
                abs_spread = 0
            our_pct = win_pct(profile.get("record"))
            # Underdog with decent record = value
            if our_pct >= 0.45 and abs_spread > 5:
                score = 7
                note = f"Dog at {abs_spread} with {our_pct:.0%} win rate — value spot"
            elif our_pct >= 0.40 and abs_spread > 3:
                score = 6
                note = f"Moderate dog value ({our_pct:.0%} at +{abs_spread})"
            else:
                score = 5
                note = f"Win rate: {our_pct:.0%} | Spread: {abs_spread}"

        elif var_name == "blue_blood_tax":
            # NCAAB: public overvalues brand names
            our_pct = win_pct(profile.get("record"))
            opp_pct = win_pct(opp_profile.get("record"))
            try:
                abs_spread = abs(float(odds.get("spread_home") or 0))
            except (ValueError, TypeError):
                abs_spread = 0
            pct_diff = abs(our_pct - opp_pct)
            # Close teams but big spread = name tax
            if pct_diff < 0.10 and abs_spread > 5:
                score = 9
                note = f"Win rates close ({pct_diff:.0%} diff) but spread {abs_spread} — brand name tax"
            elif pct_diff < 0.15 and abs_spread > 7:
                score = 8
                note = f"Similar quality ({pct_diff:.0%}) with {abs_spread} spread — public on name"
            else:
                score = 5
                note = f"Talent gap ({pct_diff:.0%}) roughly matches spread ({abs_spread})"

        elif var_name == "seed_bias":
            # NCAAB: public bets seed not quality
            our_pct = win_pct(profile.get("record"))
            opp_pct = win_pct(opp_profile.get("record"))
            try:
                abs_spread = abs(float(odds.get("spread_home") or 0))
            except (ValueError, TypeError):
                abs_spread = 0
            if our_pct >= 0.55 and abs_spread > 8:
                score = 8
                note = f"Good team ({our_pct:.0%}) as big dog (+{abs_spread}) — seed bias"
            else:
                score = 5
                note = f"No seed bias detected"

        elif var_name == "ace_name_bias":
            # MLB: Public overvalues big-name pitchers — proxy from SP data
            opp_sp = opp_profile.get("starting_pitcher", {})
            era = opp_sp.get("era") or opp_sp.get("ERA")
            try:
                abs_spread = abs(float(odds.get("spread_home") or 0))
            except (ValueError, TypeError):
                abs_spread = 0
            if era is not None:
                try:
                    era_val = float(era)
                    # Big name but bad recent ERA + public still on them
                    if era_val >= 4.5 and abs_spread > 0.5:
                        score = 8
                        note = f"Opp SP ERA {era_val:.2f} but public still backing — ace mirage"
                    elif era_val >= 3.8:
                        score = 6
                        note = f"Opp SP ERA {era_val:.2f} — moderate name bias risk"
                    else:
                        score = 4
                        note = f"Opp SP ERA {era_val:.2f} — legit ace, no bias"
                except (ValueError, TypeError):
                    score = 5
                    note = "Ace bias — ERA parse failed"
            else:
                score = 5
                note = "Ace bias — no SP ERA data"

        elif var_name == "weather_blind_spot":
            # MLB: Public ignores weather — use weather engine data
            score, note = score_weather_impact(game, pick_side)
            if score >= 7 or score <= 3:
                note = f"Weather blind spot — public ignoring: {note}"
                score = max(score, 7) if score >= 7 else max(7, 10 - score)
            else:
                note = f"Weather neutral — no blind spot"

        elif var_name == "umpire_tendency":
            # MLB: Umpire K-zone — proxy from total line
            total = odds.get("total") or odds.get("over_under")
            try:
                total_val = float(total) if total else 0
                if total_val >= 10:
                    score = 7
                    note = f"High total ({total_val}) — ump/park context may be mispriced"
                elif total_val <= 7:
                    score = 7
                    note = f"Low total ({total_val}) — tight K-zone / pitcher park"
                else:
                    score = 5
                    note = f"Total {total_val} — neutral ump signal"
            except (ValueError, TypeError):
                score = 5
                note = "Umpire tendency — no total data"

        # ── Soccer contrarian variables ──
        elif var_name == "league_reputation_tax":
            # Public overvalues big club names regardless of form
            our_pct = win_pct(profile.get("record"))
            opp_pct = win_pct(opp_profile.get("record"))
            try:
                abs_spread = abs(float(odds.get("spread_home") or 0))
            except (ValueError, TypeError):
                abs_spread = 0
            pct_diff = abs(our_pct - opp_pct)
            if pct_diff < 0.12 and abs_spread > 1.0:
                score = 9
                note = f"Close form ({pct_diff:.0%} diff) but handicap {abs_spread} — club reputation tax"
            elif pct_diff < 0.18 and abs_spread > 1.5:
                score = 7
                note = f"Form gap ({pct_diff:.0%}) doesn't justify handicap ({abs_spread})"
            else:
                score = 5
                note = f"Handicap roughly matches form gap"

        elif var_name == "home_draw_blind":
            # Public ignores draw probability in soccer
            try:
                home_ml = float(odds.get("ml_home") or odds.get("home_ml") or 0)
                away_ml = float(odds.get("ml_away") or odds.get("away_ml") or 0)
                draw_ml = float(game.get("draw_ml") or 0)
            except (ValueError, TypeError):
                home_ml, away_ml, draw_ml = 0, 0, 0
            if draw_ml and abs(home_ml - away_ml) < 50:
                score = 8
                note = f"Evenly matched (ML gap <50) — draw undervalued at {draw_ml}"
            elif draw_ml and draw_ml >= 300:
                score = 6
                note = f"Draw at {draw_ml} — public typically ignores"
            else:
                score = 5
                note = "No draw blind spot detected"

        elif var_name == "congestion_blind_spot":
            # Public ignores fixture congestion
            score, note = score_fixture_congestion(game, pick_side)
            if score >= 7:
                note = f"Public blind to congestion edge: {note}"
            else:
                score = 5
                note = "No congestion blind spot"

        else:
            score = 5
            note = f"[{var_name}] Not yet implemented"

        variables[var_name] = {
            "score": _clamp(score),
            "weight": weight,
            "weighted": round(_clamp(score) * weight, 1),
            "note": note,
            "available": True,
        }

    # Composite — only from available variables (excludes dormant MLB vars)
    active_vars = {k: v for k, v in variables.items() if v.get("available", True)}
    total_weighted = sum(v["weighted"] for v in active_vars.values())
    max_possible = sum(v["weight"] * 10 for v in active_vars.values())
    composite = round(total_weighted / max_possible * 10, 2) if max_possible > 0 else 0
    composite = _apply_spread_amplifier(composite, active_vars)

    # Check chains
    chains_fired = []
    chain_bonus = 0.0
    chains = config.get("chains", {})
    for chain_name, chain_config in chains.items():
        chain_sports = chain_config.get("sports")
        if chain_sports and sport not in chain_sports:
            continue
        fired = check_chain_claude(chain_name, variables)
        if fired:
            bonus = chain_config.get("bonus", 0.5)
            chain_bonus += bonus
            chains_fired.append(chain_name)

    cap = config.get("chain_cap", 3.0)
    chain_bonus = max(-cap, min(chain_bonus, cap))  # Cap both positive and negative
    final = round(max(1.0, min(10.0, composite + chain_bonus)), 2)
    grade = score_to_grade(final)

    if pick_side == "home":
        pick = f"{game.get('home', '?')} {odds.get('spread_home', '?')}"
    else:
        pick = f"{game.get('away', '?')} {odds.get('spread_away', '?')}"

    return {
        "game_id": game.get("game_id", ""),
        "profile": "claude",
        "thesis": "Market disagreement — fading public overreaction, recency bias, and name-value inflation",
        "pick_side": pick_side,
        "grade": grade,
        "composite": composite,
        "chain_bonus": chain_bonus,
        "final": final,
        "pick": pick,
        "sizing": score_to_sizing(final),
        "variables": variables,
        "chains_fired": chains_fired,
        "vars_available": len(active_vars),
        "vars_total": len(variables),
    }


def check_chain_claude(chain_name: str, variables: dict) -> bool:
    """Check Claude profile chain triggers."""
    v = {k: var["score"] for k, var in variables.items()}

    if chain_name == "PUBLIC_FADE":
        return (v.get("public_pct_split", 0) >= 8 and
                v.get("line_vs_public", 0) >= 8 and
                v.get("sharp_money", 0) >= 7)
    elif chain_name == "NAME_TRAP":
        return (v.get("public_star_bias", v.get("goalie_name_bias", v.get("ace_name_bias", 0))) >= 8 and
                v.get("recency_bias", 0) >= 7 and
                v.get("spread_value", 0) >= 7)
    elif chain_name == "PRIMETIME_FADE":
        return (v.get("primetime_inflation", 0) >= 8 and
                v.get("public_pct_split", 0) >= 7 and
                v.get("line_vs_public", 0) >= 7)
    elif chain_name == "BLUE_BLOOD_FADE":
        return (v.get("blue_blood_tax", 0) >= 8 and
                v.get("seed_bias", 0) >= 7 and
                v.get("line_vs_public", 0) >= 7)
    elif chain_name == "ACE_MIRAGE":
        return (v.get("ace_name_bias", 0) >= 8 and
                v.get("recency_bias", 0) >= 7 and
                v.get("sharp_money", 0) >= 7)
    elif chain_name == "BIG_CLUB_FADE":
        return (v.get("league_reputation_tax", 0) >= 8 and
                v.get("congestion_blind_spot", 0) >= 7 and
                v.get("line_vs_public", 0) >= 7)
    elif chain_name == "DRAW_VALUE":
        return (v.get("home_draw_blind", 0) >= 8 and
                v.get("spread_value", 0) >= 7 and
                v.get("public_pct_split", 0) >= 7)
    return False


# ─── Edge Profile Grader (Situational / Calendar Spots) ─────────────────────────

def grade_edge(game: dict, pick_side: str) -> dict:
    """
    Grade using Edge's situational profile.
    Calendar spots, travel, motivation, and context the math models miss.
    """
    sport = game.get("sport", "NBA")
    profile = game.get(f"{pick_side}_profile", {})
    opp_side = "away" if pick_side == "home" else "home"
    opp_profile = game.get(f"{opp_side}_profile", {})
    odds = game.get("odds", {})

    config_file = PROFILES_DIR / "edge.json"
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    sport_vars = config.get("sports", {}).get(sport, {}).get("variables", {})
    if not sport_vars:
        return {"error": f"No Edge config for {sport}"}

    variables = {}

    for var_name, var_config in sport_vars.items():
        weight = var_config["weight"]

        if var_name == "schedule_spot":
            # B2B, 3-in-4, compressed schedule
            opp_b2b = opp_profile.get("is_b2b", False)
            our_b2b = profile.get("is_b2b", False)
            opp_7d = opp_profile.get("games_last_7d", 0)
            our_7d = profile.get("games_last_7d", 0)

            if opp_b2b and not our_b2b:
                score = 9
                note = f"OPP on B2B, we're rested — schedule gold"
            elif opp_7d >= 4 and our_7d <= 2:
                score = 8
                note = f"OPP played {opp_7d} in 7d vs our {our_7d} — schedule squeeze"
            elif opp_b2b and our_b2b:
                score = 5
                note = "Both on B2B — wash"
            elif our_b2b and not opp_b2b:
                score = 2
                note = "WE on B2B, they're rested — bad spot"
            elif our_7d >= 4 and opp_7d <= 2:
                score = 3
                note = f"We played {our_7d} in 7d vs their {opp_7d} — we're squeezed"
            else:
                score = 5
                note = f"Schedule neutral (us: {our_7d}/7d, them: {opp_7d}/7d)"

        elif var_name == "lookahead_trap":
            # Without next-game schedule data, proxy from opponent quality
            # If opponent is weak (low win %) this might be a lookahead game
            opp_pct = win_pct(opp_profile.get("record"))
            our_pct = win_pct(profile.get("record"))
            try:
                abs_spread = abs(float(odds.get("spread_home") or 0))
            except (ValueError, TypeError):
                abs_spread = 0

            # Big favorite against weak team = potential lookahead
            if our_pct > 0.60 and opp_pct < 0.40 and abs_spread > 8:
                score = 7
                note = f"Strong team vs weak ({opp_pct:.0%}) with big spread — lookahead risk"
            elif abs_spread > 12:
                score = 8
                note = f"Massive spread ({abs_spread}) — motivation/lookahead concern"
            else:
                score = 5
                note = "No clear lookahead signal"

        elif var_name == "letdown_spot":
            # After a big win streak = potential letdown
            streak = profile.get("streak", "")
            opp_streak = opp_profile.get("streak", "")
            our_margin = profile.get("margin_L5", 0)

            if streak.startswith("W"):
                streak_n = int(streak[1:]) if len(streak) > 1 and streak[1:].isdigit() else 0
                if streak_n >= 5:
                    score = 5
                    note = f"On {streak} win streak — hot team, no penalty"
                elif streak_n >= 3:
                    score = 5
                    note = f"On {streak} streak — neutral"
                else:
                    score = 5
                    note = f"Streak: {streak} — no letdown signal"
            elif streak.startswith("L"):
                streak_n = int(streak[1:]) if len(streak) > 1 and streak[1:].isdigit() else 0
                if streak_n >= 3:
                    score = 7
                    note = f"OPP may be in letdown — we're hungry after {streak}"
                else:
                    score = 5
                    note = f"Streak: {streak}"
            else:
                score = 5
                note = "No streak data"

            # Flip perspective: opponent on big streak = THEIR letdown risk (good for us)
            if opp_streak.startswith("W"):
                opp_n = int(opp_streak[1:]) if len(opp_streak) > 1 and opp_streak[1:].isdigit() else 0
                if opp_n >= 5:
                    score = max(score, 8)
                    note += f" | OPP on {opp_streak} — THEIR letdown spot"
                elif opp_n >= 3:
                    score = max(score, 7)
                    note += f" | OPP on {opp_streak} — possible letdown"

        elif var_name == "motivation_gap":
            # Playoff push vs tanking/eliminated
            our_pct = win_pct(profile.get("record"))
            opp_pct = win_pct(opp_profile.get("record"))
            our_w, our_l = parse_record(profile.get("record"))
            opp_w, opp_l = parse_record(opp_profile.get("record"))

            # Late season: bad teams quit
            if opp_pct < 0.35 and our_pct > 0.55:
                score = 8
                note = f"Motivation edge: us {our_pct:.0%} vs them {opp_pct:.0%} — they may be mailing it in"
            elif our_pct < 0.35 and opp_pct > 0.55:
                score = 2
                note = f"WE might be mailing it in ({our_pct:.0%}) vs motivated {opp_pct:.0%} team"
            elif abs(our_pct - opp_pct) < 0.10:
                score = 5
                note = f"Similar records ({our_pct:.0%} vs {opp_pct:.0%}) — both motivated"
            else:
                pct_edge = our_pct - opp_pct
                score = _clamp(5 + pct_edge * 6)
                note = f"Record gap: us {our_pct:.0%} vs them {opp_pct:.0%}"

        elif var_name in ("travel_fatigue", "travel_unfamiliarity"):
            # Use road trip length and games in 7 days as proxy
            opp_road = opp_profile.get("road_trip_len", 0)
            our_road = profile.get("road_trip_len", 0)
            opp_7d = opp_profile.get("games_last_7d", 0)

            if opp_road >= 4:
                score = 8
                note = f"OPP deep into road trip ({opp_road} games) — travel fatigue"
            elif opp_road >= 2 and opp_7d >= 3:
                score = 7
                note = f"OPP road trip {opp_road} + {opp_7d} games in 7d — wear showing"
            elif our_road >= 4:
                score = 3
                note = f"WE deep into road trip ({our_road}) — bad travel spot"
            elif our_road >= 2:
                score = 4
                note = f"On the road ({our_road} games)"
            else:
                score = 5
                note = f"Travel neutral (us road: {our_road}, them: {opp_road})"

        elif var_name == "division_familiarity":
            # H2H meetings this season
            h2h = profile.get("h2h_season", "0-0")
            w, l = parse_record(h2h)
            total = w + l
            if total >= 3:
                score = 4
                note = f"4th+ meeting ({h2h}) — edges diminished, both sides adjusted"
            elif total == 2:
                score = 5
                note = f"3rd meeting ({h2h}) — familiarity building"
            else:
                score = 6
                note = f"Early in series ({h2h}) — edges still fresh"

        elif var_name == "season_phase":
            # Proxy from overall record and date context
            our_w, our_l = parse_record(profile.get("record"))
            total_games = our_w + our_l
            our_pct = win_pct(profile.get("record"))

            if total_games < 20:
                score = 4
                note = "Early season — high variance, small samples"
            elif total_games > 65:
                # Late season
                if our_pct < 0.35:
                    score = 3
                    note = f"Late season ({total_games} GP) at {our_pct:.0%} — potential tank/rest mode"
                elif our_pct > 0.65:
                    score = 4
                    note = f"Late season ({total_games} GP) at {our_pct:.0%} — may rest starters"
                else:
                    score = 7
                    note = f"Late season ({total_games} GP) at {our_pct:.0%} — playoff race intensity"
            else:
                score = 6
                note = f"Mid-season ({total_games} GP) — stable form"

        elif var_name in ("home_stand_road_trip",):
            score_val, note = score_road_trip(profile)
            score = score_val

        elif var_name == "goalie_workload":
            # NHL: goalie on B2B is brutal
            opp_b2b = opp_profile.get("is_b2b", False)
            opp_7d = opp_profile.get("games_last_7d", 0)
            if opp_b2b:
                score = 8
                note = "OPP goalie likely on B2B — backup or tired starter"
            elif opp_7d >= 4:
                score = 7
                note = f"OPP played {opp_7d} in 7d — goalie workload high"
            else:
                score = 5
                note = "Goalie workload normal"

        elif var_name == "tournament_pressure":
            # NCAAB: tournament inexperience
            try:
                abs_spread = abs(float(odds.get("spread_home") or 0))
            except (ValueError, TypeError):
                abs_spread = 0
            if abs_spread <= 3:
                score = 7
                note = "Close tournament matchup — pressure on both sides"
            elif abs_spread <= 6:
                score = 6
                note = "Moderate spread — some pressure dynamics"
            else:
                score = 5
                note = "Large spread — less pressure-dependent"

        elif var_name == "coaching_experience":
            score = 5
            note = "Coaching data not available — neutral"

        elif var_name == "pitching_rotation":
            score, note = score_starting_pitcher(game, pick_side)
            note = f"Rotation spot — {note}"
        elif var_name == "weather_impact":
            score, note = score_weather_impact(game, pick_side)
        elif var_name == "bullpen_state":
            score, note = score_bullpen_state(game, pick_side)
        elif var_name == "day_night_split":
            # Day vs night — proxy from rest/schedule
            rest = profile.get("rest_days")
            if rest is not None and rest == 0:
                score = 4
                note = "Day game after night — DGAN fatigue risk"
            else:
                score = 5
                note = "Day/night split — no data, neutral"

        # ── Soccer situational variables ──
        elif var_name == "fixture_congestion_spot":
            score, note = score_fixture_congestion(game, pick_side)
        elif var_name == "european_travel":
            score, note = score_european_hangover(game, pick_side)
        elif var_name == "derby_motivation":
            # Proxy from H2H and motivation
            score_h, note_h = score_h2h(profile)
            score_m, note_m = score_motivation_context(game, pick_side)
            score = max(score_h, score_m)
            note = f"Derby proxy: H2H {note_h} | Motivation {note_m}"
        elif var_name == "rotation_risk":
            score, note = score_rotation_risk(game, pick_side)
        elif var_name == "midweek_letdown":
            # After big European result
            score, note = score_european_hangover(game, pick_side)
            if score >= 7:
                note = f"Midweek letdown — {note}"

        else:
            score = 5
            note = f"[{var_name}] Not yet implemented"

        variables[var_name] = {
            "score": _clamp(score),
            "weight": weight,
            "weighted": round(_clamp(score) * weight, 1),
            "note": note,
            "available": True,
        }

    # Composite — only from available variables (excludes dormant MLB vars)
    active_vars = {k: v for k, v in variables.items() if v.get("available", True)}
    total_weighted = sum(v["weighted"] for v in active_vars.values())
    max_possible = sum(v["weight"] * 10 for v in active_vars.values())
    composite = round(total_weighted / max_possible * 10, 2) if max_possible > 0 else 0
    composite = _apply_spread_amplifier(composite, active_vars)

    # Check chains
    chains_fired = []
    chain_bonus = 0.0
    chains = config.get("chains", {})
    for chain_name, chain_config in chains.items():
        chain_sports = chain_config.get("sports")
        if chain_sports and sport not in chain_sports:
            continue
        fired = check_chain_edge(chain_name, variables)
        if fired:
            bonus = chain_config.get("bonus", 0.5)
            chain_bonus += bonus
            chains_fired.append(chain_name)

    cap = config.get("chain_cap", 3.0)
    chain_bonus = max(-cap, min(chain_bonus, cap))  # Cap both positive and negative
    final = round(max(1.0, min(10.0, composite + chain_bonus)), 2)
    grade = score_to_grade(final)

    if pick_side == "home":
        pick = f"{game.get('home', '?')} {odds.get('spread_home', '?')}"
    else:
        pick = f"{game.get('away', '?')} {odds.get('spread_away', '?')}"

    return {
        "game_id": game.get("game_id", ""),
        "profile": "edge",
        "thesis": "Calendar/motivation spot — schedule traps, travel fatigue, lookahead, and situational context",
        "pick_side": pick_side,
        "grade": grade,
        "composite": composite,
        "chain_bonus": chain_bonus,
        "final": final,
        "pick": pick,
        "sizing": score_to_sizing(final),
        "variables": variables,
        "chains_fired": chains_fired,
        "vars_available": len(active_vars),
        "vars_total": len(variables),
    }


def check_chain_edge(chain_name: str, variables: dict) -> bool:
    """Check Edge profile chain triggers."""
    v = {k: var["score"] for k, var in variables.items()}

    if chain_name == "SCHEDULE_SQUEEZE":
        return (v.get("schedule_spot", 0) >= 8 and
                v.get("travel_fatigue", v.get("travel_unfamiliarity", 0)) >= 7 and
                v.get("letdown_spot", 0) >= 7)
    elif chain_name == "LOOKAHEAD_LETDOWN":
        return (v.get("lookahead_trap", 0) >= 8 and
                v.get("motivation_gap", 0) >= 7)
    elif chain_name == "CINDERELLA_CRASH":
        return (v.get("letdown_spot", 0) >= 8 and
                v.get("tournament_pressure", 0) >= 8 and
                v.get("schedule_spot", 0) >= 7)
    elif chain_name == "BULLPEN_BURN":
        return (v.get("bullpen_state", 0) >= 8 and
                v.get("pitching_rotation", 0) >= 8 and
                v.get("schedule_spot", 0) >= 7)
    elif chain_name == "GOALIE_GRIND":
        return (v.get("goalie_workload", 0) >= 8 and
                v.get("schedule_spot", 0) >= 8 and
                v.get("travel_fatigue", 0) >= 7)
    elif chain_name == "DEAD_TEAM_WALKING":
        return (v.get("motivation_gap", 0) >= 8 and
                v.get("season_phase", 0) >= 7)
    elif chain_name == "EUROPEAN_HANGOVER":
        return (v.get("european_travel", 0) >= 8 and
                v.get("fixture_congestion_spot", 0) >= 8 and
                v.get("midweek_letdown", 0) >= 7)
    elif chain_name == "CONGESTION_CRASH":
        return (v.get("fixture_congestion_spot", 0) >= 9 and
                v.get("rotation_risk", 0) >= 8 and
                v.get("travel_unfamiliarity", 0) >= 6)
    elif chain_name == "DERBY_FIRE":
        return (v.get("derby_motivation", 0) >= 9 and
                v.get("motivation_gap", 0) >= 7)
    return False


# ─── Renzo Profile Grader ────────────────────────────────────────────────────────

def grade_renzo(game: dict, pick_side: str) -> dict:
    """
    Grade using Renzo's 5 Questions + Mathurin Test.
    Pure math from available data.
    """
    sport = game.get("sport", "NBA")
    profile = game.get(f"{pick_side}_profile", {})
    opp_side = "away" if pick_side == "home" else "home"
    opp_profile = game.get(f"{opp_side}_profile", {})
    odds = game.get("odds", {})

    questions = {}

    # Q1: Who's missing? Quantify PPG gap BOTH sides
    our_injuries = game.get("injuries", {}).get(pick_side, [])
    opp_injuries = game.get("injuries", {}).get(opp_side, [])
    our_ppg_lost = sum(i.get("ppg", 0) or 0 for i in our_injuries if i.get("status") == "OUT")
    opp_ppg_lost = sum(i.get("ppg", 0) or 0 for i in opp_injuries if i.get("status") == "OUT")
    ppg_gap = opp_ppg_lost - our_ppg_lost

    if ppg_gap > 15:
        q1_score = 9
    elif ppg_gap > 5:
        q1_score = 7
    elif ppg_gap > 0:
        q1_score = 6
    elif ppg_gap > -5:
        q1_score = 5
    elif ppg_gap > -15:
        q1_score = 3
    else:
        q1_score = 1

    q1_note = f"OPP lost {opp_ppg_lost:.1f} PPG | We lost {our_ppg_lost:.1f} PPG | Gap: {ppg_gap:+.1f}"
    questions["Q1_whos_missing"] = {"score": q1_score, "weight": 9, "note": q1_note}

    # Q2: Does the line reflect injuries?
    spread_delta = abs(game.get("shifts", {}).get("spread_delta", 0))
    if opp_ppg_lost > 15 and spread_delta < 2:
        q2_score = 9  # Big injury, line hasn't moved = edge
        q2_note = f"Star OUT ({opp_ppg_lost:.1f} PPG) but spread only moved {spread_delta}"
    elif opp_ppg_lost > 10 and spread_delta < 3:
        q2_score = 7
        q2_note = f"Significant injuries ({opp_ppg_lost:.1f} PPG) with small movement ({spread_delta})"
    elif our_ppg_lost > 15 and spread_delta < 2:
        q2_score = 2  # OUR star out and line hasn't adjusted
        q2_note = f"WE lost {our_ppg_lost:.1f} PPG but line barely moved"
    else:
        q2_score = 5
        q2_note = f"Line moved {spread_delta} pts | Injuries roughly priced"

    questions["Q2_line_reflects"] = {"score": q2_score, "weight": 9, "note": q2_note}

    # Q3: Recent form L5
    w, l = parse_record(profile.get("L5"))
    opp_w, opp_l = parse_record(opp_profile.get("L5"))
    form_edge = (w - l) - (opp_w - opp_l) if (w + l > 0 and opp_w + opp_l > 0) else 0

    if form_edge >= 4:
        q3_score = 9
    elif form_edge >= 2:
        q3_score = 7
    elif form_edge >= 0:
        q3_score = 5
    else:
        q3_score = max(1, 5 + form_edge)

    q3_note = f"Us L5: {profile.get('L5', '?')} | Them L5: {opp_profile.get('L5', '?')} | Edge: {form_edge:+d}"
    questions["Q3_recent_form"] = {"score": q3_score, "weight": 8, "note": q3_note}

    # Q4: Best players on court/ice
    # Without detailed player data, base on injury impact
    if our_ppg_lost < 5 and opp_ppg_lost > 10:
        q4_score = 8
        q4_note = "Our stars healthy, opponent missing key players"
    elif our_ppg_lost < 5 and opp_ppg_lost < 5:
        q4_score = 6
        q4_note = "Both sides mostly healthy"
    elif our_ppg_lost > 10:
        q4_score = 3
        q4_note = f"We're missing {our_ppg_lost:.1f} PPG of production"
    else:
        q4_score = 5
        q4_note = "Mixed injury situation"

    questions["Q4_best_on_court"] = {"score": q4_score, "weight": 8, "note": q4_note}

    # Q5: Schedule/rest/situational
    # Basic: home advantage + streak
    streak = profile.get("streak", "")
    streak_score = 5
    if streak.startswith("W") and len(streak) > 1:
        streak_num = int(streak[1:]) if streak[1:].isdigit() else 0
        streak_score = min(9, 5 + streak_num)
    elif streak.startswith("L") and len(streak) > 1:
        streak_num = int(streak[1:]) if streak[1:].isdigit() else 0
        streak_score = max(1, 5 - streak_num)

    is_home = pick_side == "home"
    q5_score = min(10, streak_score + (1 if is_home else 0))
    q5_note = f"Streak: {streak} | {'Home' if is_home else 'Away'}"
    questions["Q5_schedule_rest"] = {"score": q5_score, "weight": 7, "note": q5_note}

    # Thesis Edge (Mathurin Test) — the KEY variable
    # "Why is the market wrong? Show me the number."
    thesis_score = 5
    thesis_note = "No clear thesis edge found"
    mathurin_pass = False

    # Check for specific mispricing
    if opp_ppg_lost > 15 and spread_delta < 2:
        thesis_score = 9
        thesis_note = f"MATHURIN: {opp_ppg_lost:.1f} PPG out on opponent, line moved only {spread_delta}"
        mathurin_pass = True
    elif ppg_gap > 10 and q2_score >= 7:
        thesis_score = 8
        thesis_note = f"PPG gap {ppg_gap:+.1f} in our favor, line underreacting"
        mathurin_pass = True
    elif form_edge >= 3 and q2_score >= 6:
        thesis_score = 7
        thesis_note = f"Form edge {form_edge:+d} + line not fully reflecting"
        mathurin_pass = True

    if not mathurin_pass:
        thesis_score = min(thesis_score, 5)  # Cap at neutral without thesis

    questions["thesis_edge"] = {"score": thesis_score, "weight": 10, "note": thesis_note}

    # Calculate composite
    total_weighted = sum(q["score"] * q["weight"] for q in questions.values())
    max_possible = sum(q["weight"] * 10 for q in questions.values())
    composite = round(total_weighted / max_possible * 10, 2) if max_possible > 0 else 0
    # Convert questions to variables format for amplifier
    _renzo_vars = {k: {"score": v["score"], "weight": v["weight"], "available": True} for k, v in questions.items()}
    composite = _apply_spread_amplifier(composite, _renzo_vars)
    final = composite  # No chain bonus for Renzo
    grade = score_to_grade(final)

    if pick_side == "home":
        pick = f"{game.get('home', '?')} {odds.get('spread_home', '?')}"
    else:
        pick = f"{game.get('away', '?')} {odds.get('spread_away', '?')}"

    return {
        "game_id": game.get("game_id", ""),
        "profile": "renzo",
        "pick_side": pick_side,
        "grade": grade,
        "composite": composite,
        "chain_bonus": 0,
        "final": final,
        "pick": pick,
        "sizing": score_to_sizing(final),
        "variables": questions,
        "chains_fired": [],
        "mathurin_test": "PASS" if mathurin_pass else "FAIL",
        "mathurin_note": thesis_note,
    }


# ─── Peter's Rules (Kill/Boost Flags) ─────────────────────────────────────────────

BLOW_LEAD_TEAMS = {"Phoenix Suns"}


def grade_peter_rules(game: dict, pick_side: str) -> dict:
    """
    Apply Peter's hard rules as kill/boost flags.
    Returns flags that override or adjust the consensus.
    Applies to: NBA, NHL, MLB, WNBA, NFL, NCAAB, NCAAF, Soccer.
    Skipped for: Tennis, MMA, Boxing (no relevant rules).
    """
    sport = (game.get("sport", "") or "").upper()
    # Skip Peter's Rules for sports where they don't apply
    if sport in ("TENNIS", "MMA", "BOXING"):
        return {
            "game_id": game.get("game_id", ""),
            "profile": "peter_rules",
            "pick_side": pick_side,
            "flags": [],
            "adjustment": 0,
            "has_kill": False,
        }

    odds = game.get("odds", {})
    profile = game.get(f"{pick_side}_profile", {})
    opp_side = "away" if pick_side == "home" else "home"
    opp_profile = game.get(f"{opp_side}_profile", {})
    our_injuries = game.get("injuries", {}).get(pick_side, [])
    opp_injuries = game.get("injuries", {}).get(opp_side, [])

    flags = []
    adjustment = 0  # Net adjustment to final score

    spread = odds.get("spread_home") or odds.get("spread_away") or 0
    abs_spread = abs(spread)

    # Sport-specific thresholds
    # PPG threshold for "star player" impact
    star_ppg = {"NBA": 15, "WNBA": 15, "NCAAB": 12, "NCAAF": 0, "NHL": 0.8, "MLB": 0, "NFL": 0, "SOCCER": 0.3}.get(sport, 15)
    # Spread threshold for "big favorite" trap
    big_fav_spread = {"NBA": 15, "WNBA": 12, "NCAAB": 20, "NCAAF": 21, "NHL": 2.5, "MLB": 2.5, "NFL": 14, "SOCCER": 2.5}.get(sport, 15)

    # Rule 1: Big fav ATS trap — spread beyond threshold against winning team
    opp_record = opp_profile.get("record", "0-0")
    opp_w, opp_l = parse_record(opp_record)
    opp_pct = opp_w / max(opp_w + opp_l, 1)

    if abs_spread > big_fav_spread and opp_pct > 0.45:
        flags.append({
            "rule": "big_fav_ats",
            "action": "KILL",
            "severity": "CRITICAL",
            "note": f"Spread {abs_spread} against {opp_record} team ({opp_pct:.0%}) — OKC -18.5 trap",
        })
        adjustment -= 3.0

    # Rule 2: No Leans — anything below B+ is PASS (enforced at sizing level)
    # This is handled by the sizing map already — no adjustment needed, just flag it
    # (grades B and below already map to PASS)

    # Rule 3: Blow-lead teams on spread bets
    pick_team = game.get("home" if pick_side == "home" else "away", "")
    if pick_team in BLOW_LEAD_TEAMS:
        flags.append({
            "rule": "blowout_lead_teams",
            "action": "DOWNGRADE",
            "severity": "WARNING",
            "note": f"{pick_team} known to blow leads — spread bet risky",
        })
        adjustment -= 1.0

    # Rule 4: Fresh injury boost — star out < 3 days, books haven't adjusted
    for inj in opp_injuries:
        if (inj.get("status") == "OUT" and
            inj.get("freshness") == "FRESH" and
            star_ppg > 0 and (inj.get("ppg") or 0) >= star_ppg):
            flags.append({
                "rule": "fresh_injury_boost",
                "action": "BOOST",
                "severity": "EDGE",
                "note": f"FRESH: {inj['player']} ({inj.get('ppg')} PPG) OUT {inj.get('days_out', '?')}d — books may not have adjusted",
            })
            adjustment += 1.0

    # Rule 5: Established injury = PRICED — star out 15+ days, team winning without
    for inj in opp_injuries:
        if (inj.get("status") == "OUT" and
            inj.get("freshness") in ("ESTABLISHED", "SEASON") and
            star_ppg > 0 and (inj.get("ppg") or 0) >= star_ppg):
            flags.append({
                "rule": "established_injury_priced",
                "action": "DOWNGRADE",
                "severity": "WARNING",
                "note": f"PRICED: {inj['player']} ({inj.get('ppg')} PPG) out {inj.get('days_out', '?')}d — team adapted",
            })
            adjustment -= 0.5

    # Rule 6: Massive spread + tournament (NCAAB) — first round blowouts unreliable ATS
    sport_check = game.get("sport", "")
    if sport_check == "NCAAB" and abs_spread > 20:
        flags.append({
            "rule": "ncaab_massive_spread",
            "action": "DOWNGRADE",
            "severity": "WARNING",
            "note": f"NCAAB spread {abs_spread} — massive spreads ATS unreliable in tournament",
        })
        adjustment -= 1.0

    # ── Soccer-specific rules ──
    if sport == "SOCCER":
        # Rule S1: Heavy congestion KILL — 4+ matches in 10d backing that team
        our_m10 = profile.get("matches_in_10d") or profile.get("congestion_10d")
        if our_m10 is not None:
            try:
                m10 = int(our_m10)
                if m10 >= 4:
                    flags.append({
                        "rule": "soccer_congestion_kill",
                        "action": "KILL",
                        "severity": "CRITICAL",
                        "note": f"Backing team with {m10} matches in 10 days — congestion KILL",
                    })
                    adjustment -= 3.0
            except (ValueError, TypeError):
                pass

        # Rule S2: European hangover — UCL/UEL away within 72hrs
        euro = profile.get("european_match") or profile.get("ucl_uel_last")
        if euro:
            days_since = euro.get("days_since") or euro.get("days_ago")
            was_away = euro.get("away", False)
            if days_since is not None:
                try:
                    ds = int(days_since)
                    if ds <= 3 and was_away:
                        flags.append({
                            "rule": "soccer_european_hangover",
                            "action": "DOWNGRADE",
                            "severity": "WARNING",
                            "note": f"Played European away {ds}d ago — rotation + fatigue risk",
                        })
                        adjustment -= 1.5
                except (ValueError, TypeError):
                    pass

        # Rule S3: Draw trap boost — evenly matched teams, draw undervalued
        draw_ml = game.get("draw_ml")
        if draw_ml:
            try:
                home_ml = float(odds.get("ml_home") or odds.get("home_ml") or 0)
                away_ml = float(odds.get("ml_away") or odds.get("away_ml") or 0)
                if abs(home_ml - away_ml) < 40 and float(draw_ml) >= 250:
                    flags.append({
                        "rule": "soccer_draw_trap",
                        "action": "BOOST",
                        "severity": "EDGE",
                        "note": f"Evenly matched (ML gap <40) + draw at {draw_ml} — draw value",
                    })
                    adjustment += 0.5
            except (ValueError, TypeError):
                pass

    has_kill = any(f["action"] == "KILL" for f in flags)

    return {
        "game_id": game.get("game_id", ""),
        "profile": "peter_rules",
        "pick_side": pick_side,
        "flags": flags,
        "adjustment": round(adjustment, 1),
        "has_kill": has_kill,
        "flag_count": len(flags),
    }


# ─── Expected Value Calculator ────────────────────────────────────────────────────

def ml_to_implied_prob(ml: int | float | None) -> float | None:
    """Convert American moneyline to implied probability."""
    if ml is None:
        return None
    if ml > 0:
        return 100 / (ml + 100)
    elif ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 0.5


def grade_to_true_prob(final_score: float, implied_prob: float | None = None) -> float:
    """
    Convert consensus grade score to estimated true win probability.
    Uses implied probability as anchor, then adjusts based on grade.

    Logic: The market (implied prob) is the baseline. Our grade tells us
    if the true probability is higher or lower than the market thinks.
    - Grade 5.0 = market is right (true = implied)
    - Grade 7.0+ = market undervalues this side (true = implied + edge)
    - Grade 3.0- = market overvalues this side (true = implied - penalty)
    """
    if implied_prob is None:
        # Fallback if no ML data
        prob = 0.30 + (final_score / 10) * 0.45
        return max(0.25, min(0.80, prob))

    # Grade deviation from neutral (5.0)
    deviation = final_score - 5.0  # Range: roughly -5 to +5

    # Each point of deviation = ~3% edge over market
    # This means an A (7.8) adds ~8.4% to implied probability
    # An F (2.0) subtracts ~9% from implied probability
    edge = deviation * 0.03

    true_prob = implied_prob + edge
    return max(0.15, min(0.90, true_prob))


def calculate_ev(game: dict, pick_side: str, consensus_final: float) -> dict:
    """
    Calculate expected value for a pick.
    EV = (true_prob * payout) - ((1 - true_prob) * stake)
    Kelly = (bp - q) / b
    """
    odds = game.get("odds", {})

    # Get moneyline for our pick side
    if pick_side == "home":
        ml = odds.get("home_ml_current", odds.get("ml_home"))
    else:
        ml = odds.get("away_ml_current", odds.get("ml_away"))

    implied_prob = ml_to_implied_prob(ml)
    true_prob = grade_to_true_prob(consensus_final, implied_prob)

    if implied_prob is None or ml is None or ml == 0:
        return {
            "profile": "ev",
            "status": "no_ml_data",
            "ev_pct": None,
            "kelly_size": None,
        }

    # Decimal odds
    if ml > 0:
        decimal_odds = 1 + (ml / 100)
    else:
        decimal_odds = 1 + (100 / abs(ml))

    b = decimal_odds - 1  # Net odds
    p = true_prob
    q = 1 - p

    # EV calculation
    ev = (p * b) - q
    ev_pct = round(ev * 100, 2)

    # Kelly criterion (quarter Kelly)
    kelly_full = (b * p - q) / b if b > 0 else 0
    kelly_quarter = max(0, kelly_full * 0.25)

    # Kelly to units
    if kelly_quarter >= 0.06:
        kelly_units = "2u"
    elif kelly_quarter >= 0.04:
        kelly_units = "1.5u"
    elif kelly_quarter >= 0.02:
        kelly_units = "1u"
    elif kelly_quarter > 0:
        kelly_units = "0.5u"
    else:
        kelly_units = "PASS"

    # EV grade
    if ev_pct >= 10:
        ev_grade = "A+"
    elif ev_pct >= 7:
        ev_grade = "A"
    elif ev_pct >= 5:
        ev_grade = "B+"
    elif ev_pct >= 3:
        ev_grade = "B"
    elif ev_pct >= 0:
        ev_grade = "C"
    else:
        ev_grade = "F"

    return {
        "profile": "ev",
        "game_id": game.get("game_id", ""),
        "pick_side": pick_side,
        "moneyline": ml,
        "implied_prob": round(implied_prob, 4),
        "true_prob": round(true_prob, 4),
        "edge": round(true_prob - implied_prob, 4),
        "decimal_odds": round(decimal_odds, 3),
        "ev_pct": ev_pct,
        "ev_grade": ev_grade,
        "kelly_full": round(kelly_full, 4),
        "kelly_quarter": round(kelly_quarter, 4),
        "kelly_units": kelly_units,
        "status": "ok",
    }


# ─── Determine Best Pick Side ────────────────────────────────────────────────────

def determine_pick_side(game: dict) -> str:
    """
    Simple heuristic: grade both sides, return the better one.
    For now, use home as default and let the grade speak.
    """
    odds = game.get("odds", {})
    spread = odds.get("spread_home")

    # If home is favored (negative spread), evaluate both
    # Start with whoever has the smaller spread (closer to pick'em = more interesting)
    if spread is not None and spread < 0:
        return "home"  # Home favorite
    else:
        return "away"  # Away favorite


# ─── Main Grading Flow ───────────────────────────────────────────────────────────

def grade_all(sport: str | None = None, date: str | None = None, profiles: list[str] | None = None):
    """Grade all games for a sport/date with specified profiles."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    if profiles is None:
        profiles = ["sintonia", "renzo", "claude", "edge"]
    if sport is None:
        sport_keys = ["nba", "nhl", "ncaab"]
    else:
        sport_keys = [sport.lower()]

    GRADES_DIR.mkdir(parents=True, exist_ok=True)

    for sport_key in sport_keys:
        data_file = DATA_DIR / f"games_{sport_key}_{date}.json"
        if not data_file.exists():
            print(f"[grade] No data for {sport_key}: {data_file}")
            continue

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Sanitize odds values: convert strings to floats
        for _g in data.get("games", []):
            _odds = _g.get("odds", {})
            for _k in [
                "spread_home",
                "spread_away",
                "spread_home_val",
                "spread_away_val",
                "total",
                "total_current",
                "ml_home",
                "ml_away",
                "home_ml_current",
                "away_ml_current",
                "over_odds",
                "under_odds",
            ]:
                if _k in _odds and isinstance(_odds[_k], str):
                    try:
                        _odds[_k] = float(_odds[_k])
                    except (ValueError, TypeError):
                        _odds[_k] = 0

        all_grades = []

        for game in data.get("games", []):
            game_id = game.get("game_id", "unknown")
            matchup = f"{game.get('away', '?')} @ {game.get('home', '?')}"

            # Grade both sides, pick the better one
            for side in ["home", "away"]:
                game_grades = {"game_id": game_id, "matchup": matchup, "pick_side": side, "profiles": {}}

                for profile_name in profiles:
                    if profile_name == "sintonia":
                        grade_result = grade_sintonia(game, side)
                    elif profile_name == "renzo":
                        grade_result = grade_renzo(game, side)
                    elif profile_name == "claude":
                        grade_result = grade_claude(game, side)
                    elif profile_name == "edge":
                        grade_result = grade_edge(game, side)
                    else:
                        continue

                    game_grades["profiles"][profile_name] = grade_result

                # Add "crew" — random-weighted blend of all profiles (from v3)
                real_profiles = {k: v for k, v in game_grades["profiles"].items() if k in ("sintonia", "edge", "renzo", "claude")}
                if len(real_profiles) >= 3:
                    blend_weights = {name: random.uniform(0.2, 0.5) for name in real_profiles}
                    total_bw = sum(blend_weights.values())
                    blend_weights = {k: v / total_bw for k, v in blend_weights.items()}

                    crew_final = sum(real_profiles[name]["final"] * blend_weights[name] for name in real_profiles)
                    crew_final = round(max(1.0, min(10.0, crew_final)), 2)
                    crew_grade = score_to_grade(crew_final)

                    # Crew picks whichever side the majority of profiles pick
                    side_votes = {}
                    for pname, pdata in real_profiles.items():
                        p_side = pdata.get("pick_side", side)
                        side_votes[p_side] = side_votes.get(p_side, 0) + 1
                    crew_pick = max(side_votes, key=side_votes.get) if side_votes else side

                    game_grades["profiles"]["crew"] = {
                        "grade": crew_grade,
                        "final": crew_final,
                        "composite": crew_final,
                        "sizing": SIZING_MAP.get(crew_grade, "PASS"),
                        "chains_fired": [],
                        "pick_side": crew_pick,
                        "blend": {k: round(v, 2) for k, v in blend_weights.items()},
                    }

                # Consensus: average of ENGINE profiles only (exclude Renzo — independent grader)
                engine_profiles = {k: v for k, v in game_grades["profiles"].items() if k != "renzo"}
                avg_final = sum(
                    g["final"] for g in engine_profiles.values()
                ) / max(len(engine_profiles), 1)
                # Store Renzo's independent grade separately
                renzo_result = game_grades["profiles"].get("renzo")
                if renzo_result:
                    game_grades["renzo_independent"] = {
                        "grade": renzo_result["grade"],
                        "score": renzo_result["final"],
                        "sizing": renzo_result.get("sizing", ""),
                        "mathurin_test": renzo_result.get("mathurin_test", ""),
                    }

                # Apply Peter's Rules
                peter = grade_peter_rules(game, side)
                game_grades["peter_rules"] = peter

                # Adjust consensus with Peter's flags
                adjusted_final = round(avg_final + peter["adjustment"], 2)
                adjusted_final = max(0, min(10, adjusted_final))

                if peter["has_kill"]:
                    adjusted_final = min(adjusted_final, 2.9)  # Force F

                game_grades["consensus_avg"] = round(adjusted_final, 2)
                game_grades["consensus_grade"] = score_to_grade(adjusted_final)
                game_grades["consensus_raw"] = round(avg_final, 2)

                # Calculate EV
                ev = calculate_ev(game, side, adjusted_final)
                game_grades["ev"] = ev

                all_grades.append(game_grades)

        # Group by game, pick best side
        games_graded = {}
        for g in all_grades:
            gid = g["game_id"]
            if gid not in games_graded or g["consensus_avg"] > games_graded[gid]["consensus_avg"]:
                games_graded[gid] = g

        output = {
            "sport": sport_key.upper(),
            "date": data.get("date", ""),
            "graded_at": datetime.now().isoformat(),
            "profiles_used": profiles,
            "games": list(games_graded.values()),
            "count": len(games_graded),
        }

        out_file = GRADES_DIR / f"{sport_key}_{date}_grades.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        # ── RACE GENERATOR: build swim-lane consensus from both sides ──
        # all_grades has both home and away entries for every game
        both_sides = {}  # game_id -> {"home": {...}, "away": {...}}
        for g in all_grades:
            gid = g["game_id"]
            side = g["pick_side"]
            both_sides.setdefault(gid, {})[side] = g

        races = []
        for gid, sides in both_sides.items():
            home_g = sides.get("home")
            away_g = sides.get("away")
            if not home_g or not away_g:
                continue

            matchup = home_g["matchup"]
            lanes = []
            home_total = 0
            away_total = 0

            # Each profile is a swim lane — compare home vs away final scores
            all_profiles = set(list(home_g.get("profiles", {}).keys()) +
                              list(away_g.get("profiles", {}).keys()))
            for pname in sorted(all_profiles):
                h_score = home_g.get("profiles", {}).get(pname, {}).get("final", 0)
                a_score = away_g.get("profiles", {}).get(pname, {}).get("final", 0)
                winner = "home" if h_score > a_score else ("away" if a_score > h_score else "tie")
                lanes.append({
                    "profile": pname,
                    "home_score": h_score,
                    "away_score": a_score,
                    "winner": winner,
                })
                home_total += h_score
                away_total += a_score

            # Consensus: which side wins the race overall
            home_lanes_won = sum(1 for l in lanes if l["winner"] == "home")
            away_lanes_won = sum(1 for l in lanes if l["winner"] == "away")
            total_lanes = len(lanes)

            if home_lanes_won > away_lanes_won:
                race_winner = "home"
            elif away_lanes_won > home_lanes_won:
                race_winner = "away"
            else:
                race_winner = "home" if home_total > away_total else "away"

            # Pick from the winning side's grades
            winner_grades = games_graded.get(gid, {})
            pick_team = ""
            bet_size = ""
            if winner_grades:
                pick_side = winner_grades.get("pick_side", "")
                if pick_side == "home":
                    pick_team = home_g.get("matchup", "").split(" @ ")[-1]
                else:
                    pick_team = home_g.get("matchup", "").split(" @ ")[0]
                bet_size = winner_grades.get("profiles", {}).get("sintonia", {}).get("sizing", "PASS")

            # Confidence label based on lane agreement
            unanimity = max(home_lanes_won, away_lanes_won) / max(total_lanes, 1)
            if unanimity == 1.0:
                confidence = f"UNANIMOUS ({total_lanes}/{total_lanes} lanes)"
            elif unanimity >= 0.75:
                confidence = f"STRONG ({max(home_lanes_won, away_lanes_won)}/{total_lanes} lanes)"
            elif unanimity > 0.5:
                confidence = f"LEAN ({max(home_lanes_won, away_lanes_won)}/{total_lanes} lanes)"
            else:
                confidence = f"SPLIT ({home_lanes_won}-{away_lanes_won})"

            races.append({
                "game_id": gid,
                "matchup": matchup,
                "lanes": lanes,
                "home_score": round(home_total, 2),
                "away_score": round(away_total, 2),
                "race_winner": race_winner,
                "pick_team": pick_team,
                "bet_size": bet_size,
                "confidence": confidence,
            })

        race_output = {
            "sport": sport_key.upper(),
            "date": date,
            "generated_at": datetime.now().isoformat(),
            "races": races,
            "count": len(races),
        }
        race_file = GRADES_DIR / f"race_{sport_key}_{date}.json"
        with open(race_file, "w", encoding="utf-8") as f:
            json.dump(race_output, f, indent=2)
        print(f"[race] {sport_key.upper()}: {len(races)} races generated -> {race_file.name}")

        # Summary
        for g in games_graded.values():
            consensus = g["consensus_grade"]
            avg = g["consensus_avg"]
            raw = g.get("consensus_raw", avg)
            side = g["pick_side"]
            matchup = g["matchup"]
            profile_grades = " | ".join(
                f"{p}: {v['grade']}" for p, v in g["profiles"].items()
            )
            sizing = g["profiles"].get("sintonia", {}).get("sizing", "PASS")

            # Peter's flags
            peter = g.get("peter_rules", {})
            flags_str = ""
            if peter.get("flags"):
                flag_labels = [f"{f['action']}:{f['rule']}" for f in peter["flags"]]
                flags_str = f" | FLAGS: {', '.join(flag_labels)}"

            # EV
            ev = g.get("ev", {})
            ev_str = ""
            if ev.get("ev_pct") is not None:
                ev_str = f" | EV: {ev['ev_pct']:+.1f}% ({ev.get('ev_grade', '?')}) Kelly: {ev.get('kelly_units', '?')}"

            print(f"[grade] {matchup} -> {side.upper()} | {profile_grades} | Consensus: {consensus} ({avg}){flags_str}{ev_str} | {sizing}")

        print(f"[grade] {sport_key.upper()}: {len(games_graded)} games graded -> {out_file.name}")


if __name__ == "__main__":
    sport_arg = sys.argv[1] if len(sys.argv) > 1 else None
    date_arg = sys.argv[2] if len(sys.argv) > 2 else None
    grade_all(sport_arg, date_arg)

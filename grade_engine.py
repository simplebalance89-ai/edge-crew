"""
Layer 2: Grade Engine
Applies grading profiles to game data. Pure math — no AI.
Reads from data/, writes to grades/.
"""

import json
import math
import sys
from datetime import datetime

from paths import DATA_DIR, GRADES_DIR, PROFILES_DIR


# ─── Grade Thresholds ───────────────────────────────────────────────────────────

GRADE_THRESHOLDS = [
    (8.5, "A+"), (7.8, "A"), (7.0, "A-"),
    (6.5, "B+"), (6.0, "B"), (5.5, "B-"),
    (5.0, "C+"), (4.0, "C"), (3.0, "D"), (0.0, "F"),
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

def _clamp(val: int | float, lo: int = 1, hi: int = 10) -> int:
    return max(lo, min(hi, int(round(val))))


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
    base = {5: 9, 4: 7.5, 3: 5.5, 2: 4, 1: 2.5, 0: 1}.get(w, 5)

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
        base = 6  # Home court/ice inherent advantage
    else:
        w, l = parse_record(profile.get("away_record"))
        base = 4  # Away inherent disadvantage

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
        base = 3
        note = "Line flat — no sharp signal"

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
        # NBA average ~112 PPG
        if ppg >= 120:
            base = 9
        elif ppg >= 115:
            base = 8
        elif ppg >= 110:
            base = 7
        elif ppg >= 105:
            base = 5.5
        elif ppg >= 100:
            base = 4
        else:
            base = 3

        # Matchup boost: our offense vs their defense
        if opp_def and opp_def >= 115:
            base += 1  # They allow a lot
        elif opp_def and opp_def <= 100:
            base -= 1  # They're stingy
    elif sport == "NHL":
        if ppg >= 4.0: base = 9
        elif ppg >= 3.5: base = 8
        elif ppg >= 3.0: base = 6.5
        elif ppg >= 2.5: base = 5
        else: base = 3

        if opp_def and opp_def >= 3.5: base += 1
        elif opp_def and opp_def <= 2.5: base -= 1
    else:  # NCAAB
        if ppg >= 82: base = 9
        elif ppg >= 78: base = 8
        elif ppg >= 74: base = 7
        elif ppg >= 70: base = 5.5
        elif ppg >= 65: base = 4
        else: base = 3

    return _clamp(base), f"PPG L5: {ppg} | OPP allows: {opp_def} | Margin L5: {margin:+.1f}"


def score_def_ranking(profile: dict, opp_profile: dict, sport: str) -> tuple[int, str]:
    """Score defensive output relative to opponent's offense."""
    opp_ppg = profile.get("opp_ppg_L5", 0)  # How many we allow
    their_ppg = opp_profile.get("ppg_L5", 0)  # How many they score

    if not opp_ppg:
        return 5, "No OPP PPG data"

    if sport == "NBA":
        if opp_ppg <= 100: base = 9
        elif opp_ppg <= 105: base = 8
        elif opp_ppg <= 110: base = 6.5
        elif opp_ppg <= 115: base = 5
        elif opp_ppg <= 120: base = 3.5
        else: base = 2

        # Matchup: opponent scores a lot but we're stingy = edge
        if their_ppg and their_ppg >= 115 and opp_ppg <= 108:
            base += 1  # Matchup advantage
    elif sport == "NHL":
        if opp_ppg <= 2.0: base = 9
        elif opp_ppg <= 2.5: base = 7.5
        elif opp_ppg <= 3.0: base = 6
        elif opp_ppg <= 3.5: base = 4
        else: base = 2
    else:
        if opp_ppg <= 62: base = 9
        elif opp_ppg <= 66: base = 8
        elif opp_ppg <= 70: base = 6.5
        elif opp_ppg <= 75: base = 5
        else: base = 3

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
            score = 7  # Both fast = high scoring
            note = "FAST matchup — high scoring expected"
        elif our_pace <= 210 and opp_pace <= 210:
            score = 6  # Both slow = grind
            note = "SLOW matchup — grind game"
        elif pace_diff >= 20:
            score = 4  # Big mismatch = unpredictable
            note = f"PACE MISMATCH: {pace_diff:.0f} pt difference"
        else:
            score = 6
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
        score = 7
        note = f"Home stand: {home_len} games — well rested at home"
    elif home_len >= 2:
        score = 6
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
            note = "Coaching data not available — neutral"
        elif var_name in ("net_ranking_gap",):
            # Use season win % differential as proxy
            our_pct = win_pct(profile.get("record"))
            opp_pct = win_pct(opp_profile.get("record"))
            pct_diff = our_pct - opp_pct
            score = _clamp(5 + pct_diff * 8)
            note = f"NET proxy: us {our_pct:.0%} vs them {opp_pct:.0%} (diff {pct_diff:+.0%})"
        elif var_name in ("neutral_site",):
            score = 5
            note = "Neutral site detection not implemented — neutral"
        elif var_name in ("conference_strength",):
            score = 5
            note = "Conference strength data not available — neutral"
        elif var_name in ("ft_shooting",):
            score = 5
            note = "FT data not available — neutral"
        elif var_name in ("rivalry_motivation",):
            score, note = score_h2h(profile)
            note = f"Rivalry proxy from H2H — {note}"
        else:
            score = 5
            note = f"[{var_name}] Not yet implemented"

        variables[var_name] = {
            "score": score,
            "weight": weight,
            "weighted": round(score * weight, 1),
            "note": note,
        }

    # Calculate composite
    total_weighted = sum(v["weighted"] for v in variables.values())
    max_possible = sum(v["weight"] * 10 for v in variables.values())
    composite = round(total_weighted / max_possible * 10, 2) if max_possible > 0 else 0

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

    chain_bonus = min(chain_bonus, config.get("chain_cap", 3.0))
    final = round(min(10.0, composite + chain_bonus), 2)
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
    }


def check_chain(chain_name: str, variables: dict) -> bool:
    """Check if a chain bonus triggers based on variable scores."""
    v = {k: var["score"] for k, var in variables.items()}

    if chain_name == "THE_MISPRICING":
        return (v.get("star_player_status", 0) >= 8 and
                v.get("sharp_vs_public", 10) <= 4 and  # Line flat
                True)  # Simplified — full implementation needs sharp data
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

        elif var_name in ("ace_name_bias", "weather_blind_spot", "umpire_tendency"):
            # MLB-specific — will be populated when MLB collectors exist
            score = 5
            note = f"[{var_name}] MLB data not yet available — excluded from composite"
            variables[var_name] = {
                "score": _clamp(score), "weight": weight, "weighted": 0,
                "note": note, "available": False,
            }
            continue

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

    chain_bonus = min(chain_bonus, config.get("chain_cap", 3.0))
    final = round(min(10.0, composite + chain_bonus), 2)
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
                    score = 3
                    note = f"On {streak} win streak — letdown risk HIGH"
                elif streak_n >= 3:
                    score = 4
                    note = f"On {streak} streak — some letdown risk"
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

        elif var_name in ("pitching_rotation", "weather_impact", "bullpen_state", "day_night_split"):
            # MLB-specific — populated when MLB collectors exist
            score = 5
            note = f"[{var_name}] MLB data not yet available — excluded from composite"
            variables[var_name] = {
                "score": _clamp(score), "weight": weight, "weighted": 0,
                "note": note, "available": False,
            }
            continue

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

    chain_bonus = min(chain_bonus, config.get("chain_cap", 3.0))
    final = round(min(10.0, composite + chain_bonus), 2)
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
        thesis_score = min(thesis_score, 4)  # Cap at D without thesis

    questions["thesis_edge"] = {"score": thesis_score, "weight": 10, "note": thesis_note}

    # Calculate composite
    total_weighted = sum(q["score"] * q["weight"] for q in questions.values())
    max_possible = sum(q["weight"] * 10 for q in questions.values())
    composite = round(total_weighted / max_possible * 10, 2) if max_possible > 0 else 0
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
    Applies to: NBA, NHL, MLB, WNBA, NFL, NCAAB, NCAAF.
    Skipped for: Soccer, Tennis, MMA, Boxing (no relevant rules).
    """
    sport = (game.get("sport", "") or "").upper()
    # Skip Peter's Rules for sports where they don't apply
    if sport in ("SOCCER", "TENNIS", "MMA", "BOXING"):
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
    star_ppg = {"NBA": 15, "WNBA": 15, "NCAAB": 12, "NCAAF": 0, "NHL": 0.8, "MLB": 0, "NFL": 0}.get(sport, 15)
    # Spread threshold for "big favorite" trap
    big_fav_spread = {"NBA": 15, "WNBA": 12, "NCAAB": 20, "NCAAF": 21, "NHL": 2.5, "MLB": 2.5, "NFL": 14}.get(sport, 15)

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
    sport = game.get("sport", "")
    if sport == "NCAAB" and abs_spread > 20:
        flags.append({
            "rule": "ncaab_massive_spread",
            "action": "DOWNGRADE",
            "severity": "WARNING",
            "note": f"NCAAB spread {abs_spread} — massive spreads ATS unreliable in tournament",
        })
        adjustment -= 1.0

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

                # Consensus: best side based on average final score
                avg_final = sum(
                    g["final"] for g in game_grades["profiles"].values()
                ) / max(len(game_grades["profiles"]), 1)

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

"""
Layer 1.75: Player Profile Engine
Scores individual players on a 3-layer chain:
  Layer 1: Player form (L5 stats vs book's prop line)
  Layer 2: Team context (from roster_profile grades)
  Layer 3: Matchup (opponent's defensive profile at this position)

Fires PLAYER_CHAIN when all 3 layers align.
Calls all 4 Azure models in parallel per player.
Output: grades/player_chains_{sport}_{date}.json
"""

import json
import os
import sys
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path

import requests
from azure_config import build_client, get_model_spec

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
GRADES_DIR = BASE_DIR / "grades"
GRADES_DIR.mkdir(parents=True, exist_ok=True)

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
BDL_KEY = os.environ.get("BALLDONTLIE_API_KEY")

MODELS = {
    "grok":      (get_model_spec("grok")["deployment"], "ai_services"),
    "grok_fast": (get_model_spec("grok_fast")["deployment"], "ai_services"),
    "deepseek":  (get_model_spec("deepseek")["deployment"], "ai_services"),
    "gpt41":     (get_model_spec("gpt41")["deployment"], "openai"),
}

GRADE_THRESHOLDS = [
    (8.5, "A+"), (7.8, "A"), (7.0, "A-"),
    (6.5, "B+"), (6.0, "B"), (5.5, "B-"),
    (5.0, "C+"), (4.0, "C"), (3.0, "D"), (0.0, "F"),
]

# Minimum player impact to profile (PPG or equiv)
MIN_PPG = 10.0

# Chain fire thresholds
PLAYER_CHAIN_THRESHOLDS = {
    "player_score": 7.0,   # Player grade must be >= B+
    "team_grade_score": 6.5,  # Team Sinton.ia grade >= B+
    "matchup_score": 6.5,  # Matchup must be >= B+
}

# Prop gap thresholds to qualify as an edge
PROP_GAP_THRESHOLDS = {
    "points": 2.5,
    "assists": 1.0,
    "rebounds": 1.0,
    "three_pt": 0.5,
    "blocks": 0.3,
    "steals": 0.3,
    "pra": 3.5,    # points+rebounds+assists combo
}


# ─── Grade Helpers ──────────────────────────────────────────────────────────────

def score_to_grade(score: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


def grade_to_score(grade: str) -> float:
    mapping = {
        "A+": 9.0, "A": 7.9, "A-": 7.2,
        "B+": 6.7, "B": 6.2, "B-": 5.7,
        "C+": 5.2, "C": 4.5, "D": 3.5, "F": 1.0,
    }
    return mapping.get(grade, 5.0)


# ─── Tank01 Player Stats ─────────────────────────────────────────────────────────

def fetch_player_l5_nba(player_name: str) -> dict:
    """Fetch last 5 game stats for an NBA player via Tank01."""
    host = "tank01-fantasy-stats.p.rapidapi.com"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": host,
    }

    # Search for player
    try:
        search_url = f"https://{host}/getNBAPlayerInfo"
        r = requests.get(search_url, headers=headers,
                         params={"playerName": player_name, "statsToGet": "averages"},
                         timeout=10)
        r.raise_for_status()
        data = r.json()
        players = data.get("body", [])
        if not players:
            return {}
        player = players[0]
        player_id = player.get("playerID", "")

        # Get game logs
        logs_url = f"https://{host}/getNBAGamesForPlayer"
        r2 = requests.get(logs_url, headers=headers,
                          params={"playerID": player_id, "numberOfGames": "5"},
                          timeout=10)
        r2.raise_for_status()
        logs_data = r2.json()
        games = logs_data.get("body", {}).get("recentGames", [])

        if not games:
            return {}

        # Calculate L5 averages
        pts = [float(g.get("pts", 0) or 0) for g in games]
        ast = [float(g.get("ast", 0) or 0) for g in games]
        reb = [float(g.get("reb", 0) or 0) for g in games]
        tpm = [float(g.get("tptfgm", 0) or 0) for g in games]  # 3PM
        blk = [float(g.get("blk", 0) or 0) for g in games]
        stl = [float(g.get("stl", 0) or 0) for g in games]
        min_played = [float(g.get("mins", 0) or 0) for g in games]

        def avg(lst): return round(sum(lst) / len(lst), 1) if lst else 0

        return {
            "player_id": player_id,
            "pts_l5": avg(pts),
            "ast_l5": avg(ast),
            "reb_l5": avg(reb),
            "tpm_l5": avg(tpm),
            "blk_l5": avg(blk),
            "stl_l5": avg(stl),
            "min_l5": avg(min_played),
            "pts_max": max(pts) if pts else 0,
            "pts_min": min(pts) if pts else 0,
            "games": len(games),
            "raw_logs": games[:5],
        }

    except Exception as e:
        print(f"  [player] Tank01 error for {player_name}: {e}")
        return {}


# ─── BDL Player Props ────────────────────────────────────────────────────────────

def fetch_player_props_bdl(player_name: str, game_date: str) -> dict:
    """Fetch player props from BallDontLie for tonight."""
    headers = {"Authorization": BDL_KEY}
    url = "https://api.balldontlie.io/v1/props"
    try:
        r = requests.get(url, headers=headers,
                         params={"date": game_date, "player_name": player_name},
                         timeout=10)
        r.raise_for_status()
        data = r.json()
        props_raw = data.get("data", [])

        props = {}
        for p in props_raw:
            stat = p.get("stat_type", "").lower().replace(" ", "_")
            line = p.get("line_score", p.get("line"))
            if line is not None:
                props[stat] = float(line)
        return props
    except Exception:
        return {}


# ─── Layer 1: Player Form Score ──────────────────────────────────────────────────

def score_player_form(l5_stats: dict, props: dict) -> tuple[float, dict]:
    """
    Score player's recent form 1-10.
    Compares L5 averages vs book's prop line.
    Returns (score, details).
    """
    if not l5_stats:
        return 5.0, {"note": "No L5 data available"}

    pts = l5_stats.get("pts_l5", 0)
    ast = l5_stats.get("ast_l5", 0)
    reb = l5_stats.get("reb_l5", 0)
    min_played = l5_stats.get("min_l5", 0)
    pts_max = l5_stats.get("pts_max", 0)
    pts_min = l5_stats.get("pts_min", 0)

    # Consistency bonus: tight range = reliable
    pts_range = pts_max - pts_min
    consistency = 1.0 if pts_range <= 10 else (0.5 if pts_range <= 20 else 0.0)

    # Minutes floor — if <20 min, discount
    if min_played < 20:
        return 3.0, {"note": f"Low minutes ({min_played} min/g) — role player, skip"}

    # Base score from PPG L5
    if pts >= 30:
        base = 9.5
    elif pts >= 25:
        base = 8.5
    elif pts >= 20:
        base = 7.5
    elif pts >= 15:
        base = 6.5
    elif pts >= 12:
        base = 5.5
    elif pts >= 10:
        base = 4.5
    else:
        base = 3.0

    base += consistency * 0.5

    # Prop gap scoring
    prop_gaps = {}
    book_pts = props.get("points", props.get("player_points", 0))
    if book_pts and pts:
        gap = pts - book_pts
        prop_gaps["points"] = {"our": pts, "book": book_pts, "gap": round(gap, 1)}
        if gap >= PROP_GAP_THRESHOLDS["points"]:
            base += 1.0  # Book underpricing
        elif gap <= -PROP_GAP_THRESHOLDS["points"]:
            base -= 0.5  # We're below book

    book_ast = props.get("assists", props.get("player_assists", 0))
    if book_ast and ast:
        gap = ast - book_ast
        prop_gaps["assists"] = {"our": ast, "book": book_ast, "gap": round(gap, 1)}
        if gap >= PROP_GAP_THRESHOLDS["assists"]:
            base += 0.5

    score = max(1.0, min(10.0, base))
    return round(score, 1), {
        "pts_l5": pts, "ast_l5": ast, "reb_l5": reb,
        "min_l5": min_played, "consistency": consistency,
        "prop_gaps": prop_gaps,
        "note": f"L5 avg: {pts}pts {ast}ast {reb}reb | Consistency: {'tight' if consistency == 1.0 else 'moderate' if consistency == 0.5 else 'volatile'}",
    }


def _group_prefetched_props(props: list[dict]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for prop in props or []:
        player = prop.get("player", "")
        if not player:
            continue
        player_entry = grouped.setdefault(player, {
            "player": player,
            "team": prop.get("team", ""),
            "pos": prop.get("pos", ""),
            "prop_count": 0,
            "lines": {},
        })
        player_entry["prop_count"] += 1
        stat = prop.get("stat", "")
        if stat and prop.get("line") is not None:
            player_entry["lines"][stat] = prop["line"]
        if not player_entry.get("pos") and prop.get("pos"):
            player_entry["pos"] = prop["pos"]
    return grouped


# ─── Layer 3: Matchup Score ───────────────────────────────────────────────────────

def score_matchup(player_pos: str, opp_profile: dict, sport: str = "NBA") -> tuple[float, str]:
    """
    Score the matchup between player's role and opponent's defense.
    Uses opponent's opp_ppg_L5 and defensive profile as proxy.
    """
    opp_ppg_allowed = opp_profile.get("opp_ppg_L5", 112)  # Lower = tighter defense
    opp_l5 = opp_profile.get("L5", "3-2")

    # Parse opponent's defensive strength from opp_ppg_allowed
    # NBA avg ~112 PPG allowed
    if sport == "NBA":
        if opp_ppg_allowed >= 120:
            def_score = 9    # They're giving up everything
            def_note = f"OPP allows {opp_ppg_allowed} PPG — porous defense"
        elif opp_ppg_allowed >= 115:
            def_score = 7
            def_note = f"OPP allows {opp_ppg_allowed} PPG — below avg defense"
        elif opp_ppg_allowed >= 112:
            def_score = 5.5
            def_note = f"OPP allows {opp_ppg_allowed} PPG — average defense"
        elif opp_ppg_allowed >= 108:
            def_score = 4
            def_note = f"OPP allows {opp_ppg_allowed} PPG — solid defense"
        elif opp_ppg_allowed >= 105:
            def_score = 3
            def_note = f"OPP allows {opp_ppg_allowed} PPG — elite defense"
        else:
            def_score = 2
            def_note = f"OPP allows {opp_ppg_allowed} PPG — lockdown defense"

    # Position-specific adjustment
    pos_adj = 0
    pos_note = ""
    pos_upper = (player_pos or "").upper()

    if pos_upper in ("PG", "SG", "G"):
        # Guards thrive vs teams with poor perimeter D
        opp_3pt_def = opp_profile.get("three_pt_def_pct", None)
        if opp_3pt_def and opp_3pt_def >= 0.37:
            pos_adj += 0.5
            pos_note = "OPP weak perimeter defense — guard favorable"
    elif pos_upper in ("C", "PF", "F"):
        # Bigs thrive vs teams with poor interior D
        opp_paint = opp_profile.get("paint_allowed_l5", None)
        if opp_paint and opp_paint >= 50:
            pos_adj += 0.5
            pos_note = "OPP weak interior defense — big favorable"

    total = max(1, min(10, def_score + pos_adj))
    note = def_note + (f" | {pos_note}" if pos_note else "")
    return round(total, 1), note


# ─── PLAYER_CHAIN Fire Logic ─────────────────────────────────────────────────────

def evaluate_player_chain(
    player_score: float,
    team_grade: str,
    matchup_score: float,
    prop_gaps: dict,
) -> dict:
    """
    Fire PLAYER_CHAIN when all 3 layers align.
    Returns chain result with bonus and play recommendation.
    """
    team_score = grade_to_score(team_grade)

    thresholds = PLAYER_CHAIN_THRESHOLDS
    player_ok = player_score >= thresholds["player_score"]
    team_ok = team_score >= thresholds["team_grade_score"]
    matchup_ok = matchup_score >= thresholds["matchup_score"]

    fired = player_ok and team_ok and matchup_ok

    # Determine bonus based on how many layers are strong
    strong_count = sum([
        player_score >= 8.5,
        team_score >= 7.8,
        matchup_score >= 8.0,
    ])

    if not fired:
        return {
            "fired": False,
            "reason": f"Layers: player={player_ok}({player_score:.1f}) team={team_ok}({team_score:.1f}) matchup={matchup_ok}({matchup_score:.1f})",
            "bonus": 0.0,
        }

    # Bonus: 1.5 base + 0.5 per additional strong layer
    bonus = 1.5 + (strong_count * 0.5)
    bonus = min(bonus, 3.0)

    # Best prop play
    best_prop = None
    best_gap = 0
    for stat, threshold in PROP_GAP_THRESHOLDS.items():
        gap_data = prop_gaps.get(stat)
        if gap_data and abs(gap_data.get("gap", 0)) > best_gap:
            best_gap = abs(gap_data["gap"])
            best_prop = {
                "type": stat,
                "direction": "OVER" if gap_data["gap"] > 0 else "UNDER",
                "book": gap_data.get("book"),
                "our": gap_data.get("our"),
                "gap": gap_data["gap"],
            }

    # Play type: prop if we have a line gap, otherwise side
    play_type = "prop" if best_prop and best_gap >= 2.0 else "side"

    # Confidence
    if strong_count >= 3:
        confidence = "Very High"
    elif strong_count == 2:
        confidence = "High"
    else:
        confidence = "Moderate"

    return {
        "fired": True,
        "bonus": round(bonus, 1),
        "play_type": play_type,
        "best_prop": best_prop,
        "confidence": confidence,
        "layers": {
            "player": {"score": player_score, "ok": player_ok},
            "team": {"grade": team_grade, "score": round(team_score, 1), "ok": team_ok},
            "matchup": {"score": matchup_score, "ok": matchup_ok},
        },
        "strong_count": strong_count,
    }


# ─── Azure Model Call ─────────────────────────────────────────────────────────────

def build_player_prompt(player_name: str, team: str, pos: str,
                        l5_stats: dict, matchup_note: str,
                        opp_team: str, team_grade: str,
                        props: dict) -> str:
    pts = l5_stats.get("pts_l5", "?")
    ast = l5_stats.get("ast_l5", "?")
    reb = l5_stats.get("reb_l5", "?")
    tpm = l5_stats.get("tpm_l5", "?")
    min_played = l5_stats.get("min_l5", "?")

    prop_lines = ""
    for stat, val in props.items():
        prop_lines += f"  {stat}: {val}\n"
    if not prop_lines:
        prop_lines = "  (No prop lines available — estimate from L5)\n"

    return f"""You are a sharp sports betting player prop analyst using the Sinton.ia chain model.

## PLAYER TO PROFILE
Player: {player_name} ({pos}) — {team}
Opponent tonight: {opp_team}
Team Sinton.ia grade (tonight): {team_grade}

## LAST 5 GAMES (L5)
Points avg: {pts}
Assists avg: {ast}
Rebounds avg: {reb}
3PM avg: {tpm}
Minutes avg: {min_played}

## MATCHUP CONTEXT
{matchup_note}

## BOOK'S PROP LINES (tonight)
{prop_lines}

## YOUR TASK

Score this player on the 3-layer Sinton.ia chain:

LAYER 1 — PLAYER FORM (1-10):
Score the player's recent form. Is their L5 trending UP or DOWN? Consistent or volatile?

LAYER 2 — TEAM CONTEXT (already graded: {team_grade}):
How does this player benefit from or hurt the team's current setup? Are they the primary option or secondary?

LAYER 3 — MATCHUP (1-10):
Does tonight's opponent allow this player to thrive? Check their defensive weaknesses at this position.

CHAIN FIRE:
Does the chain fire? All 3 layers must point the same direction (all positive or all negative).
If fired: what's the best prop play? OVER or UNDER on which stat? What's our number vs the book's line?

RESPOND IN THIS EXACT JSON FORMAT:
{{
  "player": "{player_name}",
  "team": "{team}",
  "pos": "{pos}",
  "layer1_player_score": <1-10>,
  "layer1_note": "<brief note>",
  "layer2_team_score": <1-10>,
  "layer2_note": "<brief note>",
  "layer3_matchup_score": <1-10>,
  "layer3_note": "<brief note>",
  "chain_fired": <true/false>,
  "play_type": "<prop|side|none>",
  "best_prop": "<points|assists|rebounds|three_pt|pra|none>",
  "direction": "<OVER|UNDER|none>",
  "our_number": <float or null>,
  "book_line": <float or null>,
  "gap": <float or null>,
  "grade": "<A+|A|A-|B+|B|B-|C+|C|D|F>",
  "confidence": "<Very High|High|Moderate|Low>",
  "notes": "<1-2 sentence sharp take>"
}}"""


def call_model(model_key: str, prompt: str) -> dict:
    model_name, endpoint_type = MODELS[model_key]
    client, _ = build_client(model_key)

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        raw = resp.choices[0].message.content.strip()

        # Parse JSON
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])
            parsed["model"] = model_key
            return parsed
        return {"model": model_key, "error": "No JSON in response", "raw": raw[:200]}

    except Exception as e:
        return {"model": model_key, "error": str(e)}


def call_all_models(prompt: str) -> dict:
    """Call all 4 models in parallel."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(call_model, k, prompt): k for k in MODELS}
        for fut in concurrent.futures.as_completed(futures):
            key = futures[fut]
            try:
                results[key] = fut.result()
            except Exception as e:
                results[key] = {"model": key, "error": str(e)}
    return results


# ─── Consensus Across Models ─────────────────────────────────────────────────────

def build_consensus(model_results: dict, math_chain: dict) -> dict:
    """Build consensus grade from 4 models + math chain."""
    grades_breakdown = {}
    grade_scores = []

    for model_key, result in model_results.items():
        if "error" in result:
            continue
        grade = result.get("grade", "C")
        grades_breakdown[model_key] = grade
        grade_scores.append(grade_to_score(grade))

    if not grade_scores:
        return {"grade": "F", "score": 1.0, "confidence": "Low", "note": "All models failed"}

    avg_score = sum(grade_scores) / len(grade_scores)

    # Apply math chain bonus if fired
    if math_chain.get("fired"):
        avg_score = min(10.0, avg_score + math_chain["bonus"] * 0.3)

    consensus_grade = score_to_grade(avg_score)

    # Agreement check
    unique_grades = set(grades_breakdown.values())
    agreement = "Full" if len(unique_grades) == 1 else ("High" if len(unique_grades) == 2 else "Split")

    # Confidence from chain + agreement
    confidences = [r.get("confidence", "Moderate") for r in model_results.values() if "error" not in r]
    conf_map = {"Very High": 4, "High": 3, "Moderate": 2, "Low": 1}
    avg_conf = sum(conf_map.get(c, 2) for c in confidences) / max(len(confidences), 1)
    consensus_conf = (
        "Very High" if avg_conf >= 3.5 else
        "High" if avg_conf >= 2.5 else
        "Moderate" if avg_conf >= 1.5 else "Low"
    )

    return {
        "grade": consensus_grade,
        "score": round(avg_score, 2),
        "agreement": agreement,
        "confidence": consensus_conf,
        "grades_breakdown": grades_breakdown,
    }


# ─── Profile One Player ───────────────────────────────────────────────────────────

def profile_player(
    player_name: str,
    pos: str,
    team: str,
    team_grade: str,
    opp_team: str,
    opp_profile: dict,
    game_date: str,
    sport: str = "NBA",
    prefetched_props: dict | None = None,
) -> dict:
    print(f"    [{sport}] {player_name} ({pos}) — {team} vs {opp_team} | Team grade: {team_grade}")

    # Layer 1: Fetch player L5 stats
    l5_stats = fetch_player_l5_nba(player_name) if sport == "NBA" else {}
    if not l5_stats:
        print(f"      No L5 data — skipping {player_name}")
        return {}

    pts_l5 = l5_stats.get("pts_l5", 0)
    if pts_l5 < MIN_PPG:
        print(f"      PPG {pts_l5} < threshold {MIN_PPG} — skip")
        return {}

    # Fetch prop lines
    game_date_fmt = f"{game_date[:4]}-{game_date[4:6]}-{game_date[6:]}"
    props = prefetched_props or fetch_player_props_bdl(player_name, game_date_fmt)
    if not prefetched_props:
        time.sleep(0.3)  # Rate limit

    # Score layer 1 (player form)
    player_score, form_details = score_player_form(l5_stats, props)
    prop_gaps = form_details.get("prop_gaps", {})

    # Score layer 3 (matchup)
    matchup_score, matchup_note = score_matchup(pos, opp_profile, sport)

    # Fire math chain
    math_chain = evaluate_player_chain(player_score, team_grade, matchup_score, prop_gaps)

    # Build prompt and call all 4 models
    prompt = build_player_prompt(
        player_name, team, pos, l5_stats, matchup_note,
        opp_team, team_grade, props,
    )
    model_results = call_all_models(prompt)

    # Build consensus
    consensus = build_consensus(model_results, math_chain)

    status = "CHAIN FIRED" if math_chain["fired"] else "chain not fired"
    print(f"      Grade: {consensus['grade']} | {status} | Confidence: {consensus['confidence']}")

    return {
        "player": player_name,
        "pos": pos,
        "team": team,
        "opp_team": opp_team,
        "sport": sport,
        "l5_stats": l5_stats,
        "props_book": props,
        "layer1_player": {"score": player_score, **form_details},
        "layer2_team": {"grade": team_grade, "score": grade_to_score(team_grade)},
        "layer3_matchup": {"score": matchup_score, "note": matchup_note},
        "math_chain": math_chain,
        "models": model_results,
        "consensus": consensus,
        "profiled_at": datetime.now().isoformat(),
    }


# ─── Load Roster for Team ─────────────────────────────────────────────────────────

def get_team_roster_from_game(game: dict, side: str) -> list[dict]:
    """Build player candidates from attached props first, then injury feed fallback."""
    team_name = game.get(side, "")
    roster = []

    grouped_props = _group_prefetched_props(game.get("props", []))
    for player_data in grouped_props.values():
        if player_data.get("team") != team_name:
            continue
        roster.append({
            "player": player_data["player"],
            "pos": player_data.get("pos", ""),
            "ppg": player_data["lines"].get("points", 0),
            "status": "ACTIVE",
            "prop_count": player_data.get("prop_count", 0),
            "prefetched_props": player_data.get("lines", {}),
        })

    if roster:
        roster.sort(key=lambda x: (x.get("prop_count", 0), x.get("ppg", 0)), reverse=True)
        return roster[:8]

    injuries = game.get("injuries", {}).get(side, [])
    for inj in injuries:
        player = inj.get("player", "")
        pos = inj.get("pos", "")
        ppg = inj.get("ppg") or 0
        status = inj.get("status", "")
        if player and ppg >= MIN_PPG and status and status not in ("OUT", "IR", "SUSPENDED"):
            roster.append({"player": player, "pos": pos, "ppg": ppg, "status": status, "prefetched_props": {}})

    roster.sort(key=lambda x: x["ppg"], reverse=True)
    return roster[:5]


def load_team_grades(sport: str, date: str) -> dict:
    """Load team grades from roster_profile output."""
    profile_file = GRADES_DIR / f"{sport.lower()}_roster_profiles_{date}.json"
    if not profile_file.exists():
        print(f"  [player] No roster profiles found: {profile_file}")
        return {}

    with open(profile_file, encoding="utf-8") as f:
        data = json.load(f)

    grades = {}
    for team_data in data.get("teams", []):
        team_name = team_data.get("team", "")
        grade = team_data.get("consensus", {}).get("grade", "C")
        grades[team_name] = grade
    return grades


# ─── Main: Profile All Players for a Game ────────────────────────────────────────

def profile_game(game: dict, team_grades: dict, date: str, sport: str = "NBA") -> list[dict]:
    """Profile top players for both sides of a game."""
    if sport != "NBA":
        return []

    home = game.get("home", "")
    away = game.get("away", "")
    home_profile = game.get("home_profile", {})
    away_profile = game.get("away_profile", {})

    home_grade = team_grades.get(home, "C")
    away_grade = team_grades.get(away, "C")

    print(f"\n  Game: {away} @ {home}")
    print(f"  Team grades: {away}={away_grade} | {home}={home_grade}")

    results = []

    # Profile home team's top players (matchup = away's defense)
    home_roster = get_team_roster_from_game(game, "home")
    for p in home_roster:
        result = profile_player(
            player_name=p["player"],
            pos=p["pos"],
            team=home,
            team_grade=home_grade,
            opp_team=away,
            opp_profile=away_profile,
            game_date=date,
            sport=sport,
            prefetched_props=p.get("prefetched_props"),
        )
        if result:
            results.append(result)

    # Profile away team's top players (matchup = home's defense)
    away_roster = get_team_roster_from_game(game, "away")
    for p in away_roster:
        result = profile_player(
            player_name=p["player"],
            pos=p["pos"],
            team=away,
            team_grade=away_grade,
            opp_team=home,
            opp_profile=home_profile,
            game_date=date,
            sport=sport,
            prefetched_props=p.get("prefetched_props"),
        )
        if result:
            results.append(result)

    return results


def run_all(sport: str = "NBA", date: str | None = None):
    """Profile players for all games on the slate."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    sport_key = sport.lower()
    data_file = DATA_DIR / f"games_{sport_key}_{date}.json"

    if not data_file.exists():
        print(f"[player] No game data: {data_file}")
        return

    with open(data_file, encoding="utf-8") as f:
        game_data = json.load(f)

    games = game_data.get("games", [])
    if not games:
        print("[player] No games found")
        return

    # Load team grades
    team_grades = load_team_grades(sport, date)
    if not team_grades:
        print("[player] WARNING: No team grades loaded — using C for all teams")

    print(f"\n{'='*60}")
    print(f"  PLAYER PROFILE ENGINE — {sport} {date}")
    print(f"  {len(games)} games | Team grades loaded: {len(team_grades)}")
    print(f"{'='*60}")

    all_chains = []
    for game in games:
        chains = profile_game(game, team_grades, date, sport)
        all_chains.extend(chains)

    # Sort by chain bonus + consensus score
    all_chains.sort(key=lambda x: (
        x.get("math_chain", {}).get("fired", False),
        x.get("consensus", {}).get("score", 0)
    ), reverse=True)

    # Summary
    fired = [c for c in all_chains if c.get("math_chain", {}).get("fired")]
    print(f"\n{'='*60}")
    print(f"  CHAINS FIRED: {len(fired)} / {len(all_chains)} players profiled")
    for c in fired:
        mc = c.get("math_chain", {})
        best = mc.get("best_prop", {})
        prop_str = ""
        if best:
            prop_str = f" | {best.get('type','?')} {best.get('direction','?')} {best.get('book','?')} (our: {best.get('our','?')})"
        print(f"    {c['player']} ({c['team']}) — {c['consensus']['grade']} | bonus +{mc.get('bonus',0)}{prop_str}")
    print(f"{'='*60}\n")

    # Save
    output = {
        "sport": sport,
        "date": date,
        "profiled_at": datetime.now().isoformat(),
        "chains_fired": len(fired),
        "total_profiled": len(all_chains),
        "players": all_chains,
    }

    out_file = GRADES_DIR / f"player_chains_{sport_key}_{date}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[player] Saved: {out_file.name}")
    return output


if __name__ == "__main__":
    sport_arg = sys.argv[1].upper() if len(sys.argv) > 1 else "NBA"
    date_arg = sys.argv[2] if len(sys.argv) > 2 else None
    run_all(sport_arg, date_arg)

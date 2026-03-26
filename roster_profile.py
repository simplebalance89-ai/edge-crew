"""
Roster Profile Engine
Builds Sinton.ia matrix team profile from L5 data.
Calls all 4 Azure models in parallel for graded analysis.
Usage: python roster_profile.py [team1] [team2] [team3]
Default: Atlanta Hawks, Los Angeles Lakers, Boston Celtics
"""

import json
import sys
import concurrent.futures
from datetime import datetime
from pathlib import Path
from azure_config import build_client, get_model_spec

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROFILES_DIR = BASE_DIR / "profiles"

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


# ─── Load Data ─────────────────────────────────────────────────────────────────

def load_games(sport: str = "nba", date: str | None = None) -> list[dict]:
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    data_file = DATA_DIR / f"games_{sport}_{date}.json"
    if not data_file.exists():
        print(f"No data file: {data_file}")
        return []
    with open(data_file, encoding="utf-8") as f:
        return json.load(f).get("games", [])


def find_team(games: list[dict], team_name: str) -> tuple[dict, str] | None:
    """Find team profile from games. Returns (profile, side)."""
    team_lower = team_name.lower()
    for game in games:
        if team_lower in game["away"].lower():
            return game["away_profile"], "away", game["away"], game["injuries"].get("away", []), game
        if team_lower in game["home"].lower():
            return game["home_profile"], "home", game["home"], game["injuries"].get("home", []), game
    return None


# ─── Sinton.ia Matrix Scoring (standalone team) ────────────────────────────────

def score_sintonia_matrix(profile: dict, injuries: list[dict], team_name: str) -> dict:
    """
    Score a team on Sinton.ia's 15-variable NBA matrix.
    Standalone — no opponent context for matchup vars.
    """
    scores = {}

    # 1. star_player_status (weight 9) — based on injury count + severity
    out_count = sum(1 for i in injuries if i.get("status") == "OUT")
    gtd_count = sum(1 for i in injuries if i.get("status") == "GTD")
    if out_count == 0 and gtd_count == 0:
        star_score = 9
    elif out_count >= 3:
        star_score = 3
    elif out_count == 2:
        star_score = 5
    elif out_count == 1:
        star_score = 6
    else:
        star_score = 7
    scores["star_player_status"] = {"score": star_score, "weight": 9,
        "note": f"{out_count} OUT, {gtd_count} GTD"}

    # 2. rest_advantage (weight 9) — days rest
    rest = profile.get("rest_days", 2)
    is_b2b = profile.get("is_b2b", False)
    if is_b2b:
        rest_score = 3
    elif rest >= 3:
        rest_score = 9
    elif rest == 2:
        rest_score = 7
    else:
        rest_score = 5
    scores["rest_advantage"] = {"score": rest_score, "weight": 9,
        "note": f"{rest} days rest | B2B: {is_b2b}"}

    # 3. off_ranking (weight 8) — PPG L5 vs league avg ~115
    ppg_l5 = profile.get("ppg_L5", 115)
    if ppg_l5 >= 125:
        off_score = 10
    elif ppg_l5 >= 120:
        off_score = 8
    elif ppg_l5 >= 115:
        off_score = 6
    elif ppg_l5 >= 110:
        off_score = 4
    else:
        off_score = 2
    scores["off_ranking"] = {"score": off_score, "weight": 8,
        "note": f"{ppg_l5} PPG L5"}

    # 4. def_ranking (weight 8) — OPP PPG L5 vs league avg ~115 (lower = better)
    opp_ppg_l5 = profile.get("opp_ppg_L5", 115)
    if opp_ppg_l5 <= 105:
        def_score = 10
    elif opp_ppg_l5 <= 110:
        def_score = 8
    elif opp_ppg_l5 <= 115:
        def_score = 6
    elif opp_ppg_l5 <= 120:
        def_score = 4
    else:
        def_score = 2
    scores["def_ranking"] = {"score": def_score, "weight": 8,
        "note": f"{opp_ppg_l5} OPP PPG L5"}

    # 5. three_pt_matchup (weight 7) — N/A standalone; use margin as proxy
    margin_l5 = profile.get("margin_L5", 0)
    if margin_l5 >= 15:
        three_score = 9
    elif margin_l5 >= 10:
        three_score = 8
    elif margin_l5 >= 5:
        three_score = 7
    elif margin_l5 >= 0:
        three_score = 5
    elif margin_l5 >= -5:
        three_score = 4
    else:
        three_score = 2
    scores["three_pt_matchup"] = {"score": three_score, "weight": 7,
        "note": f"Proxy: {margin_l5:+.1f} margin L5 (no opp 3PT data)"}

    # 6. pace_matchup (weight 7) — pace L5 vs league avg ~225
    pace_l5 = profile.get("pace_L5", 225)
    # Score based on pace consistency (tighter = more controlled)
    pace_l10 = profile.get("pace_L10", 225)
    pace_delta = abs(pace_l5 - pace_l10)
    if pace_delta <= 3:
        pace_score = 8  # Consistent pace
    elif pace_delta <= 8:
        pace_score = 6
    else:
        pace_score = 4  # Volatile pace = less predictable
    scores["pace_matchup"] = {"score": pace_score, "weight": 7,
        "note": f"Pace L5: {pace_l5} | L10: {pace_l10} | delta: {pace_delta:.1f}"}

    # 7. recent_form (weight 7) — L5 W-L
    l5 = profile.get("L5", "0-5")
    try:
        l5_w = int(l5.split("-")[0])
    except (ValueError, IndexError):
        l5_w = 0
    form_map = {5: 10, 4: 8, 3: 6, 2: 4, 1: 2, 0: 0}
    form_score = form_map.get(l5_w, 5)
    scores["recent_form"] = {"score": form_score, "weight": 7,
        "note": f"L5: {l5} | streak: {profile.get('streak', '?')}"}

    # 8. road_trip_length (weight 7) — games into road trip
    road_len = profile.get("road_trip_len", 0)
    home_len = profile.get("home_stand_len", 0)
    if home_len >= 3:
        road_score = 9  # Deep home stand = comfortable
    elif home_len >= 1:
        road_score = 7
    elif road_len == 0:
        road_score = 6  # Just started road trip
    elif road_len <= 2:
        road_score = 5
    else:
        road_score = 3  # Deep road trip
    scores["road_trip_length"] = {"score": road_score, "weight": 7,
        "note": f"Road trip: {road_len}g | Home stand: {home_len}g"}

    # 9. h2h_season (weight 6) — season record
    record = profile.get("record", "0-0")
    try:
        w, l = map(int, record.split("-"))
        win_pct = w / (w + l) if (w + l) > 0 else 0.5
    except (ValueError, IndexError):
        win_pct = 0.5
    if win_pct >= 0.65:
        h2h_score = 9
    elif win_pct >= 0.55:
        h2h_score = 7
    elif win_pct >= 0.45:
        h2h_score = 5
    else:
        h2h_score = 3
    scores["h2h_season"] = {"score": h2h_score, "weight": 6,
        "note": f"Season: {record} ({win_pct:.0%} win rate)"}

    # 10. paint_scoring (weight 6) — proxy: PPG L5 vs season PPG delta
    ppg_season = profile.get("ppg_season", 115)
    ppg_delta = ppg_l5 - ppg_season
    if ppg_delta >= 5:
        paint_score = 9  # Scoring surge
    elif ppg_delta >= 2:
        paint_score = 7
    elif ppg_delta >= -2:
        paint_score = 6
    elif ppg_delta >= -5:
        paint_score = 4
    else:
        paint_score = 2
    scores["paint_scoring"] = {"score": paint_score, "weight": 6,
        "note": f"PPG L5: {ppg_l5} vs season: {ppg_season} (delta: {ppg_delta:+.1f})"}

    # 11. fast_break (weight 6) — proxy: margin L10
    margin_l10 = profile.get("margin_L10", 0)
    if margin_l10 >= 10:
        fb_score = 9
    elif margin_l10 >= 5:
        fb_score = 7
    elif margin_l10 >= 0:
        fb_score = 5
    elif margin_l10 >= -5:
        fb_score = 4
    else:
        fb_score = 2
    scores["fast_break"] = {"score": fb_score, "weight": 6,
        "note": f"Margin L10: {margin_l10:+.1f} (proxy)"}

    # 12. ats_trend (weight 6) — null in data, use blowout/close ratio
    blowouts = profile.get("blowouts_L10", 0)
    close = profile.get("close_games_L10", 0)
    l10 = profile.get("L10", "0-10")
    try:
        l10_w = int(l10.split("-")[0])
    except (ValueError, IndexError):
        l10_w = 0
    ats_score = min(10, max(1, int(l10_w * 1.2 - close * 0.5)))
    scores["ats_trend"] = {"score": ats_score, "weight": 6,
        "note": f"L10: {l10} | Blowouts: {blowouts} | Close: {close} (ATS data unavailable)"}

    # 13. sharp_vs_public (weight 5) — N/A standalone; use games_last_7d fatigue
    games_7d = profile.get("games_last_7d", 3)
    if games_7d <= 2:
        sharp_score = 8
    elif games_7d == 3:
        sharp_score = 6
    elif games_7d == 4:
        sharp_score = 5
    else:
        sharp_score = 3  # 5+ games in 7 days
    scores["sharp_vs_public"] = {"score": sharp_score, "weight": 5,
        "note": f"{games_7d} games in last 7 days (fatigue proxy — no line data)"}

    # 14. home_away (weight 5)
    home_record = profile.get("home_record", "0-0")
    away_record = profile.get("away_record", "0-0")
    try:
        hw, hl = map(int, home_record.split("-"))
        home_pct = hw / (hw + hl) if (hw + hl) > 0 else 0.5
    except (ValueError, IndexError):
        home_pct = 0.5
    try:
        aw, al = map(int, away_record.split("-"))
        away_pct = aw / (aw + al) if (aw + al) > 0 else 0.5
    except (ValueError, IndexError):
        away_pct = 0.5
    combined_pct = (home_pct + away_pct) / 2
    if combined_pct >= 0.65:
        ha_score = 9
    elif combined_pct >= 0.55:
        ha_score = 7
    elif combined_pct >= 0.45:
        ha_score = 5
    else:
        ha_score = 3
    scores["home_away"] = {"score": ha_score, "weight": 5,
        "note": f"Home: {home_record} ({home_pct:.0%}) | Away: {away_record} ({away_pct:.0%})"}

    # 15. depth_injuries (weight 4)
    if out_count == 0:
        depth_score = 10
    elif out_count == 1:
        depth_score = 8
    elif out_count == 2:
        depth_score = 6
    elif out_count == 3:
        depth_score = 4
    else:
        depth_score = 2
    scores["depth_injuries"] = {"score": depth_score, "weight": 4,
        "note": f"{out_count} players OUT"}

    # ─── Compute Weighted Composite ────────────────────────────────────────────
    total_weighted = 0
    total_weight = 0
    for var, data in scores.items():
        weighted = data["score"] * data["weight"]
        data["weighted"] = weighted
        total_weighted += weighted
        total_weight += data["weight"]

    composite = total_weighted / total_weight if total_weight else 0
    grade = score_to_grade(composite)
    sizing = SIZING_MAP.get(grade, "PASS")

    return {
        "variables": scores,
        "composite": round(composite, 2),
        "total_weighted": total_weighted,
        "total_weight": total_weight,
        "grade": grade,
        "sizing": sizing,
    }


# ─── Build Model Prompt ────────────────────────────────────────────────────────

def build_roster_prompt(team_name: str, profile: dict, injuries: list[dict], matrix: dict) -> str:
    out_players = [i["player"] for i in injuries if i.get("status") == "OUT"]
    gtd_players = [i["player"] for i in injuries if i.get("status") == "GTD"]

    # Top matrix scores/concerns
    vars_sorted = sorted(matrix["variables"].items(), key=lambda x: x[1]["score"])
    concerns = [f"{k}: {v['score']}/10 ({v['note']})" for k, v in vars_sorted[:3]]
    strengths = [f"{k}: {v['score']}/10 ({v['note']})" for k, v in reversed(vars_sorted[-3:])]

    return f"""You are a sharp sports betting analyst. Analyze this NBA team's current form and provide a roster profile grade.

## TEAM: {team_name}
## DATE: {datetime.now().strftime("%Y-%m-%d")}

## ROSTER STATUS
- OUT: {', '.join(out_players) if out_players else 'None'}
- GTD: {', '.join(gtd_players) if gtd_players else 'None'}

## L5 PERFORMANCE
| Metric | Value |
|--------|-------|
| L5 Record | {profile.get('L5', '?')} |
| L10 Record | {profile.get('L10', '?')} |
| Current Streak | {profile.get('streak', '?')} |
| PPG L5 | {profile.get('ppg_L5', '?')} |
| OPP PPG L5 | {profile.get('opp_ppg_L5', '?')} |
| Avg Margin L5 | {profile.get('margin_L5') or 0:+.1f} |
| Avg Margin L10 | {profile.get('avg_margin_L10') or profile.get('margin_L10') or 0:+.1f} |
| Season Record | {profile.get('record', '?')} |
| Home Record | {profile.get('home_record', '?')} |
| Away Record | {profile.get('away_record', '?')} |
| Rest Days | {profile.get('rest_days', '?')} |
| B2B Tonight | {profile.get('is_b2b', False)} |
| Road Trip Game # | {profile.get('road_trip_len', 0)} |
| Home Stand Game # | {profile.get('home_stand_len', 0)} |
| Games Last 7 Days | {profile.get('games_last_7d', '?')} |

## SINTON.IA MATRIX (Math Grade: {matrix['grade']} | Composite: {matrix['composite']}/10)

**Top Strengths:**
{chr(10).join(f'- {s}' for s in strengths)}

**Top Concerns:**
{chr(10).join(f'- {c}' for c in concerns)}

## YOUR TASK
Provide a roster profile assessment for {team_name} as a BETTING ASSET right now.

1. **Form Assessment**: Is this team a buy or fade based on L5 data? Why?
2. **Injury Impact**: How does the current injury situation affect their ceiling?
3. **Situational Flags**: Any schedule/fatigue concerns or advantages?
4. **Grade**: Confirm or override the math grade of {matrix['grade']} ({matrix['composite']}/10). Your grade should reflect team's current BETTING VALUE as a potential pick.
5. **Key Edge**: In one line — what's the single biggest factor for or against betting this team right now?

Respond in this exact JSON format:
{{
  "team": "{team_name}",
  "form_assessment": "Buy/Fade — 2-3 sentences with specific numbers",
  "injury_impact": "One sentence on injury situation",
  "situational_flags": "One sentence on schedule/fatigue",
  "math_grade": "{matrix['grade']}",
  "math_composite": {matrix['composite']},
  "model_grade": "A+/A/A-/B+/B/B-/C+/C/D/F",
  "model_score": 7.5,
  "grade_reasoning": "Why you agree/disagree with math grade",
  "sizing": "2u/1.5u/1u/PASS",
  "key_edge": "Single most important factor in one line",
  "confidence": "high/medium/low"
}}"""


# ─── Call Model ────────────────────────────────────────────────────────────────

def call_model(model_key: str, prompt: str, team_name: str) -> dict:
    model_name, endpoint_type = MODELS[model_key]
    client, _ = build_client(model_key)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a sharp sports betting analyst. Respond only with valid JSON. No markdown fences."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
        result = json.loads(raw)
        return {
            "model": model_key,
            "model_name": model_name,
            "status": "ok",
            "grade": result.get("model_grade", "?"),
            "score": float(result.get("model_score", 0)),
            "sizing": result.get("sizing", "PASS"),
            "form_assessment": result.get("form_assessment", ""),
            "injury_impact": result.get("injury_impact", ""),
            "situational_flags": result.get("situational_flags", ""),
            "grade_reasoning": result.get("grade_reasoning", ""),
            "key_edge": result.get("key_edge", ""),
            "confidence": result.get("confidence", "medium"),
        }
    except json.JSONDecodeError as e:
        return {"model": model_key, "model_name": model_name, "status": "error", "error": f"JSON: {e}"}
    except Exception as e:
        return {"model": model_key, "model_name": model_name, "status": "error", "error": str(e)}


# ─── Profile One Team ──────────────────────────────────────────────────────────

def profile_team(team_name: str, games: list[dict]) -> dict:
    print(f"\n{'='*60}")
    print(f"  Profiling: {team_name}")
    print(f"{'='*60}")

    result = find_team(games, team_name)
    if not result:
        return {"team": team_name, "error": f"Team not found in today's games"}

    profile, side, full_name, injuries, game = result
    print(f"  Found: {full_name} ({side}) | L5: {profile.get('L5')} | Streak: {profile.get('streak')}")

    # Score Sinton.ia matrix
    print(f"  Scoring Sinton.ia matrix...")
    matrix = score_sintonia_matrix(profile, injuries, full_name)
    print(f"  Math grade: {matrix['grade']} ({matrix['composite']}/10)")

    # Build prompt
    prompt = build_roster_prompt(full_name, profile, injuries, matrix)

    # Call all 4 models in parallel
    print(f"  Calling all 4 Azure models in parallel...")
    model_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(call_model, model_key, prompt, full_name): model_key
            for model_key in MODELS
        }
        for future in concurrent.futures.as_completed(futures):
            model_key = futures[future]
            result_data = future.result()
            model_results[model_key] = result_data
            status = result_data.get("status", "error")
            if status == "ok":
                print(f"    {model_key}: {result_data.get('grade', '?')} ({result_data.get('score', '?')}) — {result_data.get('key_edge', '')[:60]}")
            else:
                print(f"    {model_key}: ERROR — {result_data.get('error', 'unknown')}")

    # Consensus
    ok_results = [r for r in model_results.values() if r.get("status") == "ok"]
    if ok_results:
        avg_score = sum(r["score"] for r in ok_results) / len(ok_results)
        consensus_grade = score_to_grade(avg_score)
        grades_list = [r["grade"] for r in ok_results]
        print(f"  Consensus: {consensus_grade} (avg score: {avg_score:.1f}) | Models: {', '.join(grades_list)}")
    else:
        avg_score = matrix["composite"]
        consensus_grade = matrix["grade"]

    return {
        "team": full_name,
        "side": side,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "opponent": game["home"] if side == "away" else game["away"],
        "game_time": game.get("time", ""),
        "l5_profile": {
            "record": profile.get("record"),
            "l5": profile.get("L5"),
            "l10": profile.get("L10"),
            "streak": profile.get("streak"),
            "ppg_l5": profile.get("ppg_L5"),
            "opp_ppg_l5": profile.get("opp_ppg_L5"),
            "margin_l5": profile.get("margin_L5"),
            "margin_l10": profile.get("margin_L10"),
            "rest_days": profile.get("rest_days"),
            "is_b2b": profile.get("is_b2b"),
            "games_last_7d": profile.get("games_last_7d"),
            "road_trip_len": profile.get("road_trip_len"),
            "home_stand_len": profile.get("home_stand_len"),
        },
        "injuries": {
            "out": [i["player"] for i in injuries if i.get("status") == "OUT"],
            "gtd": [i["player"] for i in injuries if i.get("status") == "GTD"],
            "count_out": sum(1 for i in injuries if i.get("status") == "OUT"),
        },
        "sintonia_matrix": matrix,
        "model_grades": model_results,
        "consensus": {
            "grade": consensus_grade,
            "avg_score": round(avg_score, 2),
            "sizing": SIZING_MAP.get(consensus_grade, "PASS"),
            "models_ok": len(ok_results),
            "grades": [r["grade"] for r in ok_results],
        },
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    date = datetime.now().strftime("%Y%m%d")
    games = load_games("nba", date)

    if not games:
        print("No NBA games found for today.")
        sys.exit(1)

    # Teams to profile — default or from args
    if len(sys.argv) > 1:
        teams = sys.argv[1:]
    else:
        teams = ["Atlanta Hawks", "Los Angeles Lakers", "Boston Celtics"]

    print(f"\n{'#'*60}")
    print(f"  EDGE CREW V3 — NBA ROSTER PROFILES")
    print(f"  Date: {date} | Teams: {len(teams)}")
    print(f"  Models: Grok-3, Grok-Fast, DeepSeek-V3.2, GPT-4.1")
    print(f"{'#'*60}")

    results = []
    for team in teams:
        profile = profile_team(team, games)
        results.append(profile)

    # Save output
    output = {
        "type": "nba_roster_profiles",
        "date": date,
        "generated_at": datetime.now().isoformat(),
        "teams": results,
    }

    out_file = BASE_DIR / f"grades/nba_roster_profiles_{date}.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if "error" in r:
            print(f"  ❌ {r['team']}: {r['error']}")
        else:
            c = r["consensus"]
            matrix_grade = r["sintonia_matrix"]["grade"]
            print(f"  {r['team']}")
            print(f"     Math (Sinton.ia): {matrix_grade} | Consensus: {c['grade']} ({c['avg_score']}) | Sizing: {c['sizing']}")
            print(f"     Models: {' | '.join(c['grades'])}")
            print(f"     Opponent tonight: {r['opponent']}")

    print(f"\n  Saved: {out_file}")
    return output


if __name__ == "__main__":
    main()

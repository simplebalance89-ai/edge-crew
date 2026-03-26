"""
Layer 4: Model Caller
On-demand model grading — one card at a time.
Sends clean structured data to Grok via Azure AI Services.
Returns model grade displayed next to math grades.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from azure_config import build_client, get_model_spec

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
GRADES_DIR = BASE_DIR / "grades"

# Available models
MODELS = {
    "grok": get_model_spec("grok")["deployment"],
    "grok-fast": get_model_spec("grok-fast")["deployment"],
    "deepseek": get_model_spec("deepseek")["deployment"],
    "gpt41": get_model_spec("gpt41")["deployment"],
    "gpt41mini": get_model_spec("gpt41mini")["deployment"],
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


def build_prompt(game: dict, grade_info: dict | None = None) -> str:
    """Build a clean structured prompt for the model. No raw data dumps — organized tables."""
    odds = game.get("odds", {})
    away = game.get("away", "?")
    home = game.get("home", "?")
    sport = game.get("sport", "?")
    away_prof = game.get("away_profile", {})
    home_prof = game.get("home_profile", {})
    away_inj = game.get("injuries", {}).get("away", [])
    home_inj = game.get("injuries", {}).get("home", [])
    shifts = game.get("shifts", {})

    # Build injury summaries
    def fmt_injuries(injuries):
        if not injuries:
            return "None reported"
        lines = []
        for inj in injuries:
            if inj.get("status") in ("OUT", "GTD"):
                ppg = inj.get("ppg")
                ppg_str = f" ({ppg} PPG)" if ppg else ""
                lines.append(f"  - {inj['player']} ({inj.get('pos', '?')}) — {inj['status']} — {inj.get('injury', '?')}{ppg_str} [{inj.get('freshness', '?')}, {inj.get('days_out', '?')}d]")
        return "\n".join(lines) if lines else "None significant"

    # Build math grades summary if available
    math_grades = ""
    if grade_info and grade_info.get("profiles"):
        math_grades = "\n\n## MATH GRADES (from our models — for comparison only)\n"
        for pname, pdata in grade_info["profiles"].items():
            math_grades += f"- {pdata.get('profile', pname)}: {pdata.get('grade', '?')} ({pdata.get('final', '?')}) — Pick: {pdata.get('pick', '?')} {pdata.get('sizing', '')}\n"
            if pdata.get("chains_fired"):
                math_grades += f"  Chains: {', '.join(pdata['chains_fired'])}\n"
            if pdata.get("mathurin_test"):
                math_grades += f"  Mathurin Test: {pdata['mathurin_test']} — {pdata.get('mathurin_note', '')}\n"

    prompt = f"""You are a sharp sports betting analyst. Grade this {sport} game for betting value.

## MATCHUP
{away} @ {home}
Game Time: {game.get('time', '?')}

## ODDS
| Line | Open | Current | Move |
|------|------|---------|------|
| Spread | {odds.get('spread_open', '?')} | {home} {odds.get('spread_home', '?')} | {shifts.get('spread_delta', 0):+.1f} |
| Total | {odds.get('total_open', '?')} | {odds.get('total_current', '?')} | {shifts.get('total_delta', 0):+.1f} |
| ML {away} | {odds.get('away_ml_open', '?')} | {odds.get('away_ml_current', '?')} | |
| ML {home} | {odds.get('home_ml_open', '?')} | {odds.get('home_ml_current', '?')} | |

## TEAM PROFILES
| Stat | {away} | {home} |
|------|--------|--------|
| Record | {away_prof.get('record', '?')} | {home_prof.get('record', '?')} |
| Home | {away_prof.get('home_record', '?')} | {home_prof.get('home_record', '?')} |
| Away | {away_prof.get('away_record', '?')} | {home_prof.get('away_record', '?')} |
| L5 | {away_prof.get('L5', '?')} | {home_prof.get('L5', '?')} |
| L10 | {away_prof.get('L10', '?')} | {home_prof.get('L10', '?')} |
| PPG L5 | {away_prof.get('ppg_L5', '?')} | {home_prof.get('ppg_L5', '?')} |
| OPP PPG L5 | {away_prof.get('opp_ppg_L5', '?')} | {home_prof.get('opp_ppg_L5', '?')} |
| Streak | {away_prof.get('streak', '?')} | {home_prof.get('streak', '?')} |

## INJURIES — {away}
{fmt_injuries(away_inj)}

## INJURIES — {home}
{fmt_injuries(home_inj)}
{math_grades}

## YOUR TASK
1. Identify the BEST bet in this game (spread, total, or ML). One pick only.
2. Grade it on a 1-10 scale (use the same thresholds: 8.5+=A+, 7.8+=A, 7.0+=A-, 6.5+=B+, 6.0+=B, 5.5+=B-, 5.0+=C+, 4.0+=C, 3.0+=D, <3.0=F).
3. Size it: A+ = 2u, A = 1.5u, A- or B+ = 1u, below B+ = PASS.
4. Explain WHY in 2-3 sentences. Be specific with numbers.

Respond in this exact JSON format:
{{
  "pick": "TEAM SPREAD/TOTAL/ML",
  "score": 7.5,
  "grade": "A-",
  "sizing": "1u",
  "reasoning": "Your 2-3 sentence explanation with specific numbers.",
  "key_factor": "Single most important factor in one line.",
  "confidence": "high/medium/low"
}}"""

    return prompt


def call_model(game: dict, grade_info: dict | None = None, model_key: str = "grok") -> dict:
    """Call the model with structured game data. Returns parsed grade."""
    model_name = MODELS.get(model_key, MODELS["grok"])
    client, _ = build_client(model_key)

    prompt = build_prompt(game, grade_info)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a sharp sports betting analyst. Respond only with valid JSON. No markdown fences."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        result = json.loads(raw)

        return {
            "model": model_name,
            "model_key": model_key,
            "pick": result.get("pick", "?"),
            "score": float(result.get("score", 0)),
            "grade": result.get("grade", "?"),
            "sizing": result.get("sizing", "PASS"),
            "reasoning": result.get("reasoning", ""),
            "key_factor": result.get("key_factor", ""),
            "confidence": result.get("confidence", "medium"),
            "raw_response": raw,
            "called_at": datetime.now().isoformat(),
            "status": "ok",
        }

    except json.JSONDecodeError as e:
        return {
            "model": model_name,
            "model_key": model_key,
            "status": "error",
            "error": f"JSON parse error: {e}",
            "raw_response": raw if 'raw' in dir() else "",
            "called_at": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "model": model_name,
            "model_key": model_key,
            "status": "error",
            "error": str(e),
            "called_at": datetime.now().isoformat(),
        }


def grade_game_with_model(game_id: str, sport: str | None = None, date: str | None = None, model_key: str = "grok") -> dict:
    """Find a game by ID, call model, save result."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    search_sports = [sport.lower()] if sport else ["nba", "nhl", "ncaab"]

    for sport_key in search_sports:
        data_file = DATA_DIR / f"games_{sport_key}_{date}.json"
        if not data_file.exists():
            continue

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for game in data.get("games", []):
            if game.get("game_id") == game_id:
                # Load existing grades for comparison
                grade_info = None
                grade_file = GRADES_DIR / f"{sport_key}_{date}_grades.json"
                if grade_file.exists():
                    with open(grade_file, "r", encoding="utf-8") as gf:
                        grades_data = json.load(gf)
                    for graded in grades_data.get("games", []):
                        if graded.get("game_id") == game_id:
                            grade_info = graded
                            break

                # Call model
                result = call_model(game, grade_info, model_key)
                result["game_id"] = game_id
                result["matchup"] = f"{game.get('away', '?')} @ {game.get('home', '?')}"

                # Save model grade back into grades file
                if grade_file.exists() and result.get("status") == "ok":
                    with open(grade_file, "r", encoding="utf-8") as gf:
                        grades_data = json.load(gf)

                    for graded in grades_data.get("games", []):
                        if graded.get("game_id") == game_id:
                            if "model_grades" not in graded:
                                graded["model_grades"] = {}
                            graded["model_grades"][model_key] = result
                            break

                    with open(grade_file, "w", encoding="utf-8") as gf:
                        json.dump(grades_data, gf, indent=2)

                return result

    return {"status": "error", "error": f"Game {game_id} not found"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model_caller.py <game_id> [model_key]")
        sys.exit(1)

    game_id = sys.argv[1]
    model_key = sys.argv[2] if len(sys.argv) > 2 else "grok"
    result = grade_game_with_model(game_id, model_key=model_key)
    print(json.dumps(result, indent=2))

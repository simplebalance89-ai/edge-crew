"""
Prop Analyzer — sends game data to models with prop-specific prompt.
Asks for top 3-5 player prop edges based on injuries, matchups, recent form.
"""
import json, sys
from datetime import datetime
from azure_config import build_client, get_model_spec
from paths import DATA_DIR

def build_prop_prompt(game):
    odds = game.get("odds", {})
    away = game.get("away", "?")
    home = game.get("home", "?")
    away_prof = game.get("away_profile", {})
    home_prof = game.get("home_profile", {})
    away_inj = game.get("injuries", {}).get("away", [])
    home_inj = game.get("injuries", {}).get("home", [])

    def fmt_inj(injuries):
        if not injuries: return "None"
        lines = []
        for i in injuries:
            if i.get("status") in ("OUT", "GTD"):
                lines.append(f"  {i['player']} ({i.get('pos','?')}) — {i['status']} — {i.get('ppg','?')} PPG — {i.get('freshness','?')} {i.get('days_out','?')}d")
        return "\n".join(lines) if lines else "None significant"

    return f"""You are a sharp sports betting prop analyst. Identify the TOP 5 player prop edges for this game.

## MATCHUP
{away} @ {home}
Spread: {home} {odds.get('spread_home', '?')} | Total: {odds.get('total_current', '?')}

## {away} (L5: {away_prof.get('L5','?')} | PPG: {away_prof.get('ppg_L5','?')} | OPP PPG: {away_prof.get('opp_ppg_L5','?')} | Streak: {away_prof.get('streak','?')})
Injuries:
{fmt_inj(away_inj)}

## {home} (L5: {home_prof.get('L5','?')} | PPG: {home_prof.get('ppg_L5','?')} | OPP PPG: {home_prof.get('opp_ppg_L5','?')} | Streak: {home_prof.get('streak','?')})
Injuries:
{fmt_inj(home_inj)}

## YOUR TASK
Find the 5 best player prop bets based on:
1. Opposing team injuries creating usage/shot boosts
2. Players on hot streaks getting favorable matchups
3. Total line suggesting pace/scoring environment
4. Defensive weaknesses the opponent allows

For each prop give: player, stat (PTS/REB/AST/3PM), over/under, why (1 sentence with numbers), and confidence (high/medium/low).

Respond in this exact JSON format:
{{
  "props": [
    {{"player": "Name", "team": "TEAM", "stat": "PTS", "direction": "OVER", "reasoning": "Why with numbers.", "confidence": "high"}},
    ...
  ],
  "game_environment": "One sentence on expected pace/scoring."
}}"""


def analyze_props(game_id, model_key="grok"):
    models = {
        "grok": get_model_spec("grok")["deployment"],
        "deepseek": get_model_spec("deepseek")["deployment"],
        "grok-fast": get_model_spec("grok-fast")["deployment"],
    }
    model_name = models.get(model_key, models["grok"])
    client, _ = build_client(model_key)

    # Find game
    for sport_key in ["nba", "nhl", "ncaab"]:
        data_file = DATA_DIR / f"games_{sport_key}_{datetime.now().strftime('%Y%m%d')}.json"
        if not data_file.exists(): continue
        with open(data_file) as f: data = json.load(f)
        for game in data.get("games", []):
            if game.get("game_id") == game_id:
                prompt = build_prop_prompt(game)
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a sharp prop analyst. Respond only with valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3, max_tokens=800,
                )
                raw = resp.choices[0].message.content.strip()
                if raw.startswith("```"): raw = raw.split("\n",1)[1].rsplit("```",1)[0].strip()
                result = json.loads(raw)
                result["game_id"] = game_id
                result["matchup"] = f"{game['away']} @ {game['home']}"
                result["model"] = model_key
                return result
    return {"error": f"Game {game_id} not found"}

if __name__ == "__main__":
    gid = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "grok"
    r = analyze_props(gid, model)
    print(json.dumps(r, indent=2))

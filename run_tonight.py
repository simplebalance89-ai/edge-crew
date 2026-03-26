"""Run tonight's 8-stage hurdle system on real NBA slate."""
import asyncio
from datetime import datetime, timedelta, timezone
import httpx

REST_DATA = {
    "Hawks": {"rest_days": 1, "is_b2b": False, "games_7d": 2, "streak": 1},
    "Nets": {"rest_days": 0, "is_b2b": True, "games_7d": 2, "streak": -2},
    "Bulls": {"rest_days": 3, "is_b2b": False, "games_7d": 0, "streak": 0},
    "Mavericks": {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": -1},
    "Nuggets": {"rest_days": 0, "is_b2b": True, "games_7d": 2, "streak": 2},
    "Pistons": {"rest_days": 2, "is_b2b": False, "games_7d": 1, "streak": 1},
    "Warriors": {"rest_days": 1, "is_b2b": False, "games_7d": 2, "streak": -2},
    "Rockets": {"rest_days": 1, "is_b2b": False, "games_7d": 2, "streak": 2},
    "Pacers": {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": -1},
    "Clippers": {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": 1},
    "Lakers": {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": 1},
    "Grizzlies": {"rest_days": 1, "is_b2b": False, "games_7d": 2, "streak": -2},
    "Heat": {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": -1},
    "Bucks": {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": 1},
    "Pelicans": {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": -1},
    "Knicks": {"rest_days": 0, "is_b2b": True, "games_7d": 2, "streak": 2},
    "Thunder": {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": 1},
    "Magic": {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": -1},
    "76ers": {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": 1},
    "Suns": {"rest_days": 0, "is_b2b": True, "games_7d": 2, "streak": 1},
    "Trail Blazers": {"rest_days": 0, "is_b2b": True, "games_7d": 2, "streak": -1},
    "Spurs": {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": 1},
    "Raptors": {"rest_days": 0, "is_b2b": True, "games_7d": 2, "streak": -2},
    "Jazz": {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": -1},
}


def get_rest(team_name):
    for key, data in REST_DATA.items():
        if key.lower() in team_name.lower() or team_name.lower() in key.lower():
            return data
    return {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": 0}


async def run_sport(sport_key, sport_label, rest_data_map=None, ai_fn=None):
    from engines.stage_models import HurdleGame
    from engines.stage0_filter import run_stage0
    from engines.stage1_fixture import run_stage1
    from engines.stage2_origin import run_stage2
    from engines.stage3_team_dna import run_stage3
    from engines.stage4_h2h import run_stage4
    from engines.stage5_structural import run_stage5
    from engines.stage6_market import run_stage6
    from engines.stage7_lineup import run_stage7
    from engines.stage8_final import run_stage8

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/",
            params={
                "apiKey": "eca3a06babb5d86e9317960ea4eb2661",
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
            },
        )
        raw_games = resp.json()

    sport_name = sport_label.lower()
    # Filter to TODAY only (March 23 UTC = games starting before March 24 ~8am UTC)
    today_cutoff_start = "2026-03-23T"
    today_cutoff_end = "2026-03-24T08:00:00Z"  # Captures late West Coast games
    todays_games = [
        g for g in raw_games
        if g.get("commence_time", "") >= today_cutoff_start
        and g.get("commence_time", "") <= today_cutoff_end
    ]
    print(f"  Filtered to today: {len(todays_games)} of {len(raw_games)} games")

    games = []
    for g in todays_games:
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        spread = total = home_ml = away_ml = None
        for bm in g.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt["key"] == "spreads":
                    for o in mkt["outcomes"]:
                        if o["name"] == home and spread is None:
                            spread = o.get("point")
                elif mkt["key"] == "totals":
                    for o in mkt["outcomes"]:
                        if o["name"] == "Over" and total is None:
                            total = o.get("point")
                elif mkt["key"] == "h2h":
                    for o in mkt["outcomes"]:
                        if o["name"] == home and home_ml is None:
                            home_ml = o.get("price")
                        elif o["name"] == away and away_ml is None:
                            away_ml = o.get("price")

        # Rest data
        if rest_data_map:
            hr = get_rest(home)
            ar = get_rest(away)
        else:
            hr = {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": 0}
            ar = {"rest_days": 1, "is_b2b": False, "games_7d": 1, "streak": 0}

        hg = HurdleGame(
            game_id=g.get("id", ""),
            sport=sport_name,
            home_team=home,
            away_team=away,
            commence_time=g.get("commence_time", ""),
            home_spread=spread,
            total=total,
            home_ml=home_ml,
            away_ml=away_ml,
            home_rest_days=hr["rest_days"],
            away_rest_days=ar["rest_days"],
            home_b2b=hr["is_b2b"],
            away_b2b=ar["is_b2b"],
            home_games_7d=hr["games_7d"],
            away_games_7d=ar["games_7d"],
            home_streak=hr["streak"],
            away_streak=ar["streak"],
        )
        games.append(hg)

    print("=" * 60)
    print(f"8-STAGE HURDLE — {sport_label} — {len(games)} GAMES")
    print("=" * 60)

    # Stage 0
    s0 = await run_stage0(games)
    survivors = s0["passed"]
    print(f"S0 Morning Filter | Profile: {s0['slate_profile']}")
    print(f"   PASSED ({len(survivors)}):")
    for g in survivors:
        r = g.stage_results[-1]
        factors = ", ".join(r.factors.keys())
        print(f"     {g.away_team} @ {g.home_team} | Score {r.score} | {factors}")
    print(f"   KILLED ({len(s0['killed'])}): ", end="")
    print(", ".join(f"{g.away_team}@{g.home_team}" for g in s0["killed"]))
    print()

    if not survivors:
        print("No games survived Stage 0.")
        return []

    # Stage 1-7
    survivors = run_stage1(survivors)
    print(f"S1 Fixture: {len(survivors)} pass")
    s2p, _, s2k = await run_stage2(survivors, sport=sport_name)
    survivors = s2p
    print(f"S2 Origin: {len(survivors)} pass, {len(s2k)} kill")
    s3i = len(survivors)
    survivors = await run_stage3(survivors, ai_caller=ai_fn)
    print(f"S3 DNA: {len(survivors)} pass")
    s4i = len(survivors)
    survivors = run_stage4(survivors)
    print(f"S4 H2H: {len(survivors)} pass")
    s5i = len(survivors)
    survivors = run_stage5(survivors)
    print(f"S5 Structural: {len(survivors)} pass, {s5i - len(survivors)} kill")
    s6i = len(survivors)
    survivors = await run_stage6(survivors, ai_caller=ai_fn)
    print(f"S6 Market: {len(survivors)} pass, {s6i - len(survivors)} kill")
    s7p, s7i = await run_stage7(survivors)
    survivors = s7p
    print(f"S7 Lineup: {len(survivors)} pass")

    # Stage 8
    final = run_stage8(survivors)
    print(f"S8 Final: {len(final)} picks")
    print()

    print("-" * 60)
    print(f"{sport_label} PICKS")
    print("-" * 60)
    for p in final:
        print(f"  {p['grade']} | {p['matchup']}")
        print(f"       {p['bet_type']} {p['bet_line']} | {p['sizing']} | Edge: {p['edge_side']}")
        print(f"       {p['bet_reasoning']}")
        if p["degraded"]:
            print(f"       [DEGRADED x{p['degrade_factor']:.2f}]")
        if p["crowdsource_flag"]:
            print(f"       CROWDSOURCE: {p['crowdsource_reason']}")
        print()

    return final


async def main():
    from engines.ai_caller import call_ai, test_ai

    # Test AI connectivity
    ai_fn = None
    print("Testing AI connection (DeepSeek-V3.2)...")
    ai_ok = await test_ai()
    if ai_ok:
        print("AI: CONNECTED — Stage 3 + Stage 6 will use blinded AI analysis")
        ai_fn = call_ai
    else:
        print("AI: OFFLINE — using math-only fallback")
    print()

    # NBA
    nba = await run_sport("basketball_nba", "NBA", rest_data_map=REST_DATA, ai_fn=ai_fn)
    print()

    # NHL
    nhl = await run_sport("icehockey_nhl", "NHL", ai_fn=ai_fn)
    print()

    # NCAAB
    ncaab = await run_sport("basketball_ncaab", "NCAAB", ai_fn=ai_fn)
    print()

    # Summary
    print("=" * 60)
    print("TONIGHT'S CARD — ALL SPORTS")
    print("=" * 60)
    all_picks = nba + nhl + ncaab
    if not all_picks:
        print("  System found no picks tonight.")
    for p in all_picks:
        print(f"  {p['grade']} | {p['sport'].upper()} | {p['matchup']} | {p['bet_type']} {p['bet_line']} | {p['sizing']}")


asyncio.run(main())

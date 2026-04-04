"""
STAGE 1: SCHEDULE & REST — Display and score fixture/rest/travel data.

DATA DISPLAYED:
  - Rest days per team (home_rest_days, away_rest_days)
  - Back-to-back detection (home_b2b, away_b2b)
  - Games in last 7 days (home_games_7d, away_games_7d)
  - Rest differential between teams
  - Schedule density / fixture congestion
  - Travel context (road trip length, distance if available)

VERDICT: Always PASS — stages 0-5 never kill. Scoring informs downstream
aggregation and the final Stage 6 gate.

Prerequisite: If no odds exist the game still passes, but score is 0
and a note flags the missing market data.
"""

import logging
from typing import List

from engines.stage_models import HurdleGame, StageResult, Verdict

logger = logging.getLogger("edge-crew")

# ---------------------------------------------------------------------------
# Sport-specific weight overrides (default base weights below)
# ---------------------------------------------------------------------------
_SPORT_WEIGHTS = {
    # key: sport  ->  dict of factor weight overrides
    "nba": {
        "b2b_one_side": 2.5,     # B2B matters a lot in NBA
    },
    "nhl": {
        "b2b_one_side": 3.0,     # B2B road team is biggest fade in hockey
    },
    "soccer": {
        "congestion": 2.5,       # European midweek + weekend fixture pile-up
    },
    "mlb": {
        "dgan": 1.5,             # Day game after night — MLB's key rest signal
        "rest_differential": 0.0,  # Daily games, rest diff is noise
        "b2b_one_side": 0.0,       # Not applicable
    },
    "ncaab": {
        "away_travel": 1.5,      # Tournament travel pressure
    },
}

# Base weights (used when sport doesn't override)
_BASE = {
    "rest_differential": 1.5,   # rest diff >= 2 days
    "b2b_one_side": 2.0,        # one team B2B, other rested
    "b2b_both": -1.0,           # both B2B = chaos
    "congestion": 1.0,          # 4+ games in 7 days for one team
    "road_trip": 1.0,           # 4+ consecutive road games
    "cross_country": 0.5,       # >1500 mi travel
    "dgan": 0.0,                # only MLB uses this
    "away_travel": 0.0,
}


def _weight(sport: str, key: str) -> float:
    """Return the weight for *key*, respecting sport-specific overrides."""
    overrides = _SPORT_WEIGHTS.get(sport, {})
    if key in overrides:
        return overrides[key]
    return _BASE.get(key, 0.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_stage1(games: List[HurdleGame]) -> List[HurdleGame]:
    """
    Score every game on schedule/rest data and return ALL games.
    No kills — verdict is always PASS.
    """
    for game in games:
        result = _score_schedule(game)
        game.stage_results.append(result)
        logger.info(
            f"[STAGE 1] {game.away_team} @ {game.home_team}  "
            f"score={result.score:.1f}  notes={result.notes}"
        )

    logger.info(f"[STAGE 1] {len(games)} games scored (0 killed)")
    return games


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def _has_odds(game: HurdleGame) -> bool:
    """Quick prerequisite — does the game have any market data?"""
    return game.home_ml is not None and game.away_ml is not None


def _score_schedule(game: HurdleGame) -> StageResult:
    """Build factors dict and compute a 0-10 schedule/rest score."""
    sport = game.sport.lower()
    factors: dict = {}
    adjustments: list = []   # (label, delta) tuples for transparency
    score = 5.0              # neutral baseline

    # ------------------------------------------------------------------
    # Prerequisite: market check (2-line gate, no kill)
    # ------------------------------------------------------------------
    has_market = _has_odds(game)
    factors["has_market"] = has_market
    if not has_market:
        return StageResult(
            stage=1,
            name="Schedule & Rest",
            game_id=game.game_id,
            score=0.0,
            threshold=0.0,
            verdict=Verdict.PASS,
            confidence=0.0,
            next_stage=2,
            factors={"has_market": False},
            notes="No market data — score zeroed, game still passes",
        )

    # ------------------------------------------------------------------
    # Rest days
    # ------------------------------------------------------------------
    home_rest = game.home_rest_days
    away_rest = game.away_rest_days
    factors["home_rest_days"] = home_rest
    factors["away_rest_days"] = away_rest

    rest_known = home_rest >= 0 and away_rest >= 0
    rest_diff = abs(home_rest - away_rest) if rest_known else 0
    factors["rest_differential"] = rest_diff if rest_known else "unknown"

    if rest_known and rest_diff >= 2:
        w = _weight(sport, "rest_differential")
        rested_side = "home" if home_rest > away_rest else "away"
        adjustments.append((f"rest_diff_{rest_diff}d_favors_{rested_side}", w))
        score += w
        factors["rest_edge"] = rested_side

    # ------------------------------------------------------------------
    # Back-to-back
    # ------------------------------------------------------------------
    factors["home_b2b"] = game.home_b2b
    factors["away_b2b"] = game.away_b2b

    if game.home_b2b and game.away_b2b:
        w = _BASE["b2b_both"]  # always the same across sports
        adjustments.append(("both_b2b_chaos", w))
        score += w
    elif game.home_b2b or game.away_b2b:
        w = _weight(sport, "b2b_one_side")
        tired_side = "home" if game.home_b2b else "away"
        fresh_side = "away" if game.home_b2b else "home"
        adjustments.append((f"b2b_{tired_side}_favors_{fresh_side}", w))
        score += w
        factors["b2b_edge"] = fresh_side

    # ------------------------------------------------------------------
    # Games in last 7 days — schedule density / congestion
    # ------------------------------------------------------------------
    factors["home_games_7d"] = game.home_games_7d
    factors["away_games_7d"] = game.away_games_7d

    congestion_diff = abs(game.home_games_7d - game.away_games_7d)
    busy_home = game.home_games_7d >= 4
    busy_away = game.away_games_7d >= 4

    if busy_home and not busy_away:
        w = _weight(sport, "congestion")
        adjustments.append(("home_congested_favors_away", w))
        score += w
        factors["congestion_edge"] = "away"
    elif busy_away and not busy_home:
        w = _weight(sport, "congestion")
        adjustments.append(("away_congested_favors_home", w))
        score += w
        factors["congestion_edge"] = "home"

    # ------------------------------------------------------------------
    # Travel / road-trip context (from metadata if available)
    # ------------------------------------------------------------------
    meta = game.metadata or {}

    # Road trip length
    home_road_trip = meta.get("home_road_trip_len", 0)
    away_road_trip = meta.get("away_road_trip_len", 0)
    factors["home_road_trip_len"] = home_road_trip
    factors["away_road_trip_len"] = away_road_trip

    if away_road_trip >= 4:
        w = _weight(sport, "road_trip")
        adjustments.append(("away_long_road_trip_favors_home", w))
        score += w

    # Cross-country travel distance
    travel_miles = meta.get("away_travel_miles", 0)
    factors["away_travel_miles"] = travel_miles
    if travel_miles > 1500:
        w = _weight(sport, "cross_country")
        adjustments.append(("cross_country_travel_favors_home", w))
        score += w

    # ------------------------------------------------------------------
    # MLB-specific: Day Game After Night (DGAN)
    # ------------------------------------------------------------------
    if sport == "mlb":
        home_dgan = meta.get("home_dgan", False)
        away_dgan = meta.get("away_dgan", False)
        factors["home_dgan"] = home_dgan
        factors["away_dgan"] = away_dgan
        if home_dgan and not away_dgan:
            w = _weight(sport, "dgan")
            adjustments.append(("home_dgan_favors_away", w))
            score += w
        elif away_dgan and not home_dgan:
            w = _weight(sport, "dgan")
            adjustments.append(("away_dgan_favors_home", w))
            score += w

    # ------------------------------------------------------------------
    # Soccer-specific: European hangover / rotation risk
    # ------------------------------------------------------------------
    if sport == "soccer":
        euro_hangover = meta.get("european_hangover", False)
        rotation_risk = meta.get("rotation_risk", False)
        factors["european_hangover"] = euro_hangover
        factors["rotation_risk"] = rotation_risk
        if euro_hangover:
            adjustments.append(("european_hangover", 1.0))
            score += 1.0

    # ------------------------------------------------------------------
    # NCAAB-specific: away travel pressure
    # ------------------------------------------------------------------
    if sport == "ncaab":
        if travel_miles > 500:
            w = _weight(sport, "away_travel")
            adjustments.append(("ncaab_away_travel_pressure", w))
            score += w

    # ------------------------------------------------------------------
    # Clamp and build result
    # ------------------------------------------------------------------
    score = max(0.0, min(10.0, round(score, 1)))
    factors["adjustments"] = adjustments
    factors["sport_weights_used"] = sport

    # Confidence: higher when we have more data points
    data_points = sum([
        rest_known,
        game.home_b2b or game.away_b2b,
        game.home_games_7d > 0 or game.away_games_7d > 0,
        travel_miles > 0,
    ])
    confidence = min(0.95, 0.4 + data_points * 0.15)

    # Build human-readable notes
    note_parts = []
    if rest_known:
        note_parts.append(f"rest {home_rest}d/{away_rest}d (H/A)")
    if game.home_b2b:
        note_parts.append("home B2B")
    if game.away_b2b:
        note_parts.append("away B2B")
    if congestion_diff:
        note_parts.append(
            f"7d games {game.home_games_7d}/{game.away_games_7d} (H/A)"
        )
    if travel_miles > 0:
        note_parts.append(f"travel {travel_miles}mi")
    if not note_parts:
        note_parts.append("no schedule data available")

    notes = "Schedule: " + "; ".join(note_parts)

    return StageResult(
        stage=1,
        name="Schedule & Rest",
        game_id=game.game_id,
        score=score,
        threshold=0.0,       # no threshold — S1 never kills
        verdict=Verdict.PASS,
        confidence=confidence,
        next_stage=2,
        factors=factors,
        notes=notes,
    )

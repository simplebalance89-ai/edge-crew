"""
STAGE 4: H2H CONTEXT — Historical patterns with decay weighting.

Pure math stage. No AI needed.

Decay Weights:
  - Last 3 months: 60%
  - 3-12 months: 30%
  - 12+ months: 10%

PASS: Score >4.0, no red flags
DEGRADE: Historical baggage but not disqualifying
KILL: Score <4.0 with red flags (e.g., 5-game losing streak at venue)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from engines.stage_models import HurdleGame, StageResult, Verdict

logger = logging.getLogger("edge-crew")

PASS_THRESHOLD = 4.0

# Decay windows
DECAY_RECENT = 0.60   # Last 3 months
DECAY_MEDIUM = 0.30   # 3-12 months
DECAY_OLD = 0.10      # 12+ months


def run_stage4(
    games: List[HurdleGame],
    scores_archive: dict = None,
    team_match_fn=None,
) -> List[HurdleGame]:
    """
    Run H2H context analysis on all games.

    Args:
        games: Survivors from Stage 3
        scores_archive: Dict of completed games from server.py's scores archive.
                       Keyed by game ID with home_team, away_team, scores, commence_time.
        team_match_fn: Function(team, game, sport) -> 'home'|'away'|'' for fuzzy matching.

    Returns: List of games that pass (includes degraded)
    """
    passed = []
    killed = 0

    for game in games:
        result = _evaluate_h2h(game, scores_archive or {}, team_match_fn)
        game.stage_results.append(result)

        if result.verdict in (Verdict.PASS, Verdict.DEGRADE):
            if result.verdict == Verdict.DEGRADE:
                game.degraded = True
                game.degrade_factor *= 0.85
            passed.append(game)
        else:
            killed += 1
            logger.info(
                f"[STAGE 4] KILL {game.away_team} @ {game.home_team}: {result.notes}"
            )

    logger.info(f"[STAGE 4] {len(passed)} passed, {killed} killed")
    return passed


def _evaluate_h2h(
    game: HurdleGame, archive: dict, team_match_fn=None
) -> StageResult:
    """Score a game's H2H history with decay weighting."""
    factors = {}
    red_flags = []

    # Find H2H games from archive
    h2h_games = _find_h2h_games(
        game.home_team, game.away_team, game.sport, archive, team_match_fn
    )

    factors["h2h_count"] = len(h2h_games)

    if not h2h_games:
        # No H2H data — pass with neutral score (don't kill for missing data)
        return StageResult(
            stage=4,
            name="H2H Context",
            game_id=game.game_id,
            score=5.0,
            threshold=PASS_THRESHOLD,
            verdict=Verdict.PASS,
            confidence=0.3,
            next_stage=5,
            factors={"h2h_count": 0, "method": "no_data_passthrough"},
            notes="No H2H history found — passing with neutral score",
        )

    # Calculate decay-weighted score
    now = datetime.utcnow()
    weighted_score = 0.0
    total_weight = 0.0
    home_wins_at_home = 0
    home_losses_at_home = 0
    away_wins_on_road = 0
    total_h2h = len(h2h_games)

    for match in h2h_games:
        match_date = match.get("date")
        if not match_date:
            continue

        try:
            if isinstance(match_date, str):
                match_dt = datetime.strptime(match_date[:10], "%Y-%m-%d")
            else:
                match_dt = match_date
        except (ValueError, TypeError):
            continue

        days_ago = (now - match_dt).days

        # Decay weight
        if days_ago < 90:
            weight = DECAY_RECENT
        elif days_ago < 365:
            weight = DECAY_MEDIUM
        else:
            weight = DECAY_OLD

        # Score based on who won and by how much
        home_won = match.get("home_won", False)
        margin = abs(match.get("margin", 0))

        if home_won:
            # Home team won this H2H
            result_score = 6.0 + min(margin * 0.2, 2.0)  # 6-8 range
            home_wins_at_home += 1 if match.get("at_home_venue", True) else 0
        else:
            # Away team won this H2H
            result_score = 4.0 - min(margin * 0.2, 2.0)  # 2-4 range
            home_losses_at_home += 1 if match.get("at_home_venue", True) else 0
            away_wins_on_road += 1

        weighted_score += result_score * weight
        total_weight += weight

    final_score = weighted_score / total_weight if total_weight > 0 else 5.0
    final_score = min(max(final_score, 0.0), 10.0)

    factors["decay_weighted_score"] = round(final_score, 1)
    factors["home_wins_at_home"] = home_wins_at_home
    factors["home_losses_at_home"] = home_losses_at_home
    factors["away_wins_on_road"] = away_wins_on_road

    # Red flag detection
    if home_losses_at_home >= 3:
        red_flags.append(f"{game.home_team} lost {home_losses_at_home} of last {total_h2h} at home vs {game.away_team}")
    if away_wins_on_road >= 3:
        red_flags.append(f"{game.away_team} won {away_wins_on_road} of last {total_h2h} on the road here")

    factors["red_flags"] = red_flags

    # Verdict
    if red_flags and final_score < PASS_THRESHOLD:
        verdict = Verdict.KILL
        notes = f"Score {final_score:.1f} + red flags: {'; '.join(red_flags)}"
    elif red_flags:
        verdict = Verdict.DEGRADE
        notes = f"Score {final_score:.1f} but flagged: {'; '.join(red_flags)}"
    elif final_score >= PASS_THRESHOLD:
        verdict = Verdict.PASS
        notes = f"H2H score {final_score:.1f} — {total_h2h} games analyzed"
    else:
        verdict = Verdict.KILL
        notes = f"H2H score {final_score:.1f} < {PASS_THRESHOLD} — unfavorable history"

    return StageResult(
        stage=4,
        name="H2H Context",
        game_id=game.game_id,
        score=round(final_score, 1),
        threshold=PASS_THRESHOLD,
        verdict=verdict,
        confidence=0.6 if total_h2h >= 3 else 0.4,
        next_stage=5 if verdict in (Verdict.PASS, Verdict.DEGRADE) else None,
        factors=factors,
        notes=notes,
    )


def _find_h2h_games(
    home: str, away: str, sport: str, archive: dict, match_fn=None
) -> List[Dict]:
    """Find historical matchups between these two teams from the archive."""
    matches = []
    home_lower = home.lower()
    away_lower = away.lower()

    for gid, game in archive.items():
        if not game.get("completed"):
            continue

        sport_key = game.get("sport_key", "")
        if sport.lower() not in sport_key:
            continue

        g_home = (game.get("home_team", "") or "").lower()
        g_away = (game.get("away_team", "") or "").lower()

        # Check if this game involves both teams (in either order)
        teams_match = False
        at_home_venue = False

        if match_fn:
            h_side = match_fn(home, game, sport)
            a_side = match_fn(away, game, sport)
            if h_side and a_side:
                teams_match = True
                at_home_venue = h_side == "home"
        else:
            # Simple substring matching
            h_in_home = any(p in g_home for p in home_lower.split() if len(p) > 2)
            h_in_away = any(p in g_away for p in home_lower.split() if len(p) > 2)
            a_in_home = any(p in g_home for p in away_lower.split() if len(p) > 2)
            a_in_away = any(p in g_away for p in away_lower.split() if len(p) > 2)

            if (h_in_home and a_in_away) or (h_in_away and a_in_home):
                teams_match = True
                at_home_venue = h_in_home

        if not teams_match:
            continue

        # Extract scores
        home_score = _get_score(game, "home")
        away_score = _get_score(game, "away")
        if home_score is None or away_score is None:
            continue

        # Determine winner relative to today's home team
        if at_home_venue:
            home_won = home_score > away_score
            margin = home_score - away_score
        else:
            home_won = away_score > home_score
            margin = away_score - home_score

        matches.append({
            "date": game.get("commence_time", "")[:10],
            "home_won": home_won,
            "margin": margin,
            "at_home_venue": at_home_venue,
            "score": f"{home_score}-{away_score}",
        })

    # Sort by date descending, cap at 10 most recent
    matches.sort(key=lambda x: x.get("date", ""), reverse=True)
    return matches[:10]


def _get_score(game: dict, side: str) -> Optional[int]:
    """Extract score from various game dict formats."""
    for key in [f"{side}_score", f"{side}Score"]:
        val = game.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass

    # Try scores dict
    scores = game.get("scores", {})
    if isinstance(scores, dict):
        val = scores.get(side)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass

    return None

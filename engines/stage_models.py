"""
Shared data models for the 8-Stage Hurdle System.
Verdict enum, StageResult dataclass, and HurdleGame wrapper.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class Verdict(Enum):
    PASS = "PASS"
    KILL = "KILL"
    SOFT_FAIL = "SOFT_FAIL"   # Retry later (e.g., questionable star)
    DEGRADE = "DEGRADE"       # Continue with reduced weight
    INCOMPLETE = "INCOMPLETE"  # Missing critical data


@dataclass
class StageResult:
    stage: int
    name: str
    game_id: str
    score: float
    threshold: float
    verdict: Verdict
    confidence: float
    next_stage: Optional[int]
    factors: Dict
    notes: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


@dataclass
class HurdleGame:
    """Wrapper around our existing game dict with hurdle-specific fields.
    Bridges the gap between server.py game format and the hurdle pipeline."""

    game_id: str
    sport: str
    home_team: str
    away_team: str
    commence_time: str

    # Odds data (from get_odds)
    home_spread: Optional[float] = None
    away_ml: Optional[int] = None
    home_ml: Optional[int] = None
    total: Optional[float] = None
    opening_spread: Optional[float] = None
    opening_total: Optional[float] = None
    spread_move: float = 0.0
    total_move: float = 0.0

    # Rest/schedule (from scores archive + _build_team_profile)
    home_rest_days: int = -1    # -1 = unknown
    away_rest_days: int = -1
    home_b2b: bool = False
    away_b2b: bool = False
    home_games_7d: int = 0
    away_games_7d: int = 0

    # Injury flags (from ESPN + RotoWire)
    injuries: List[Dict] = field(default_factory=list)
    home_star_out: int = 0
    away_star_out: int = 0
    home_star_questionable: int = 0
    away_star_questionable: int = 0

    # Streaks (from _build_team_profile)
    home_streak: int = 0
    away_streak: int = 0
    home_l5_record: str = ""
    away_l5_record: str = ""

    # Metadata (accumulates through stages)
    metadata: Dict = field(default_factory=dict)
    degraded: bool = False
    degrade_factor: float = 1.0

    # Stage chain
    stage_results: List[StageResult] = field(default_factory=list)

    @classmethod
    def from_odds_game(cls, game: Dict, sport: str) -> "HurdleGame":
        """Create a HurdleGame from the game dict returned by get_odds/odds API."""
        return cls(
            game_id=game.get("id", game.get("game_id", "")),
            sport=sport.lower(),
            home_team=game.get("home", game.get("home_team", "")),
            away_team=game.get("away", game.get("away_team", "")),
            commence_time=game.get("commence_time", ""),
            home_spread=_safe_float(game.get("home_spread")),
            away_ml=_safe_int(game.get("away_ml")),
            home_ml=_safe_int(game.get("home_ml")),
            total=_safe_float(game.get("total")),
        )

    def to_dict(self) -> Dict:
        return {
            "game_id": self.game_id,
            "sport": self.sport,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "commence_time": self.commence_time,
            "matchup": f"{self.away_team} @ {self.home_team}",
            "home_spread": self.home_spread,
            "total": self.total,
            "spread_move": self.spread_move,
            "total_move": self.total_move,
            "home_rest_days": self.home_rest_days,
            "away_rest_days": self.away_rest_days,
            "home_b2b": self.home_b2b,
            "away_b2b": self.away_b2b,
            "home_streak": self.home_streak,
            "away_streak": self.away_streak,
            "home_star_out": self.home_star_out,
            "away_star_out": self.away_star_out,
            "degraded": self.degraded,
            "degrade_factor": self.degrade_factor,
            "stage_chain": [r.to_dict() for r in self.stage_results],
        }


def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def _safe_int(v) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (ValueError, TypeError):
        return None

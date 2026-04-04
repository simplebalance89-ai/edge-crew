"""
Soccer Fixture Congestion & Rotation Engine — schedule-based fatigue analysis.

ESPN Soccer API: https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard
Free, public, no API key required.

Fixture congestion is the #1 edge in soccer betting. Teams playing midweek
Champions League and then weekend league games rotate 3-5 players on average
and consistently underperform. 3+ matches in 10 days = heavy congestion.
European away travel + domestic game within 72 hrs = strong FADE signal.

Used by: server.py analysis prompt (soccer congestion context injection)
Cache: 2 hours (schedules change with postponements)
"""

import httpx
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("edge-crew")

ESPN_SOCCER_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# ESPN league slugs for major competitions
ESPN_LEAGUES = {
    "eng.1": "English Premier League",
    "esp.1": "La Liga",
    "ger.1": "Bundesliga",
    "ita.1": "Serie A",
    "fra.1": "Ligue 1",
    "ned.1": "Eredivisie",
    "por.1": "Primeira Liga",
    "usa.1": "MLS",
    "uefa.champions": "UEFA Champions League",
    "uefa.europa": "UEFA Europa League",
    "uefa.europa.conf": "UEFA Conference League",
    "eng.fa": "FA Cup",
    "eng.league_cup": "EFL Cup",
    "esp.copa_del_rey": "Copa del Rey",
    "ita.coppa_italia": "Coppa Italia",
    "ger.dfb_pokal": "DFB-Pokal",
    "fra.coupe_de_france": "Coupe de France",
}

# European competitions (high intensity + travel)
EUROPEAN_COMPS = {"uefa.champions", "uefa.europa", "uefa.europa.conf"}

# Competition intensity tiers
COMP_INTENSITY = {
    "uefa.champions": "UCL",
    "uefa.europa": "UEL",
    "uefa.europa.conf": "UECL",
    "eng.1": "League",
    "esp.1": "League",
    "ger.1": "League",
    "ita.1": "League",
    "fra.1": "League",
    "ned.1": "League",
    "por.1": "League",
    "usa.1": "League",
    "eng.fa": "Cup",
    "eng.league_cup": "Cup",
    "esp.copa_del_rey": "Cup",
    "ita.coppa_italia": "Cup",
    "ger.dfb_pokal": "Cup",
    "fra.coupe_de_france": "Cup",
}

# Module-level cache
_congestion_cache = {}


def _get_cached(key: str, ttl: int = 7200):
    """Retrieve cached value if still within TTL (default 2 hours)."""
    if key in _congestion_cache:
        data, ts = _congestion_cache[key]
        if (datetime.now(timezone.utc) - ts).total_seconds() < ttl:
            return data
    return None


def _set_cache(key: str, data):
    """Store value in module-level cache with current timestamp."""
    _congestion_cache[key] = (data, datetime.now(timezone.utc))


def _normalize_team_name(name: str) -> str:
    """Normalize team name for fuzzy matching — lowercase, strip common suffixes."""
    n = name.lower().strip()
    for suffix in (" fc", " cf", " sc", " ac", " afc", " ssc", " bsc"):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


def _team_match(team_name: str, search_name: str) -> bool:
    """Check if two team names refer to the same club via substring matching."""
    a = _normalize_team_name(team_name)
    b = _normalize_team_name(search_name)
    if a == b:
        return True
    # Substring in either direction
    if a in b or b in a:
        return True
    # Handle common short names (e.g. "Arsenal" in "Arsenal London")
    a_parts = a.split()
    b_parts = b.split()
    if len(a_parts) >= 1 and len(b_parts) >= 1:
        if a_parts[0] == b_parts[0] and len(a_parts[0]) >= 4:
            return True
    return False


async def _fetch_espn_scoreboard(league: str, date_str: str) -> list:
    """Fetch ESPN scoreboard for a league on a specific date.

    Args:
        league: ESPN league slug (e.g. "eng.1", "uefa.champions")
        date_str: Date in YYYYMMDD format

    Returns:
        List of match dicts: [{"home": str, "away": str, "date": datetime,
                               "league": str, "status": str}, ...]
    """
    cache_key = f"espn_soccer:{league}:{date_str}"
    cached = _get_cached(cache_key, ttl=7200)
    if cached is not None:
        return cached

    url = f"{ESPN_SOCCER_BASE}/{league}/scoreboard"
    params = {"dates": date_str}

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[SOCCER CONGESTION] ESPN fetch failed for {league} {date_str}: {e}")
        return []

    matches = []
    for event in data.get("events", []):
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        if len(competitors) < 2:
            continue

        home_name = ""
        away_name = ""
        for comp in competitors:
            team_name = comp.get("team", {}).get("displayName", "")
            if comp.get("homeAway") == "home":
                home_name = team_name
            else:
                away_name = team_name

        match_date_str = event.get("date", "")
        try:
            match_date = datetime.fromisoformat(match_date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            match_date = None

        status = event.get("status", {}).get("type", {}).get("name", "")

        matches.append({
            "home": home_name,
            "away": away_name,
            "date": match_date,
            "league": league,
            "league_name": ESPN_LEAGUES.get(league, league),
            "comp_type": COMP_INTENSITY.get(league, "Other"),
            "is_european": league in EUROPEAN_COMPS,
            "status": status,
        })

    _set_cache(cache_key, matches)
    return matches


async def fetch_team_schedule(team_name: str, days_back: int = 14, days_forward: int = 7) -> list:
    """Fetch all recent and upcoming matches for a team across all competitions.

    Scans ESPN scoreboards across all tracked leagues for the past `days_back`
    days and next `days_forward` days, returning matches involving the team.

    Args:
        team_name: Team name to search for (fuzzy matched)
        days_back: How many days back to look (default 14)
        days_forward: How many days forward to look (default 7)

    Returns:
        Sorted list of match dicts involving the team, newest first for past,
        soonest first for future.
    """
    cache_key = f"soccer_schedule:{_normalize_team_name(team_name)}:{days_back}:{days_forward}"
    cached = _get_cached(cache_key, ttl=7200)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    all_matches = []

    # Scan each day across all leagues
    dates_to_check = []
    for offset in range(-days_back, days_forward + 1):
        d = now + timedelta(days=offset)
        dates_to_check.append(d.strftime("%Y%m%d"))

    # Fetch all league/date combos — prioritize domestic + European comps
    for league in ESPN_LEAGUES:
        for date_str in dates_to_check:
            day_matches = await _fetch_espn_scoreboard(league, date_str)
            for m in day_matches:
                if _team_match(m["home"], team_name) or _team_match(m["away"], team_name):
                    is_home = _team_match(m["home"], team_name)
                    all_matches.append({
                        **m,
                        "is_home": is_home,
                        "opponent": m["away"] if is_home else m["home"],
                        "venue": "home" if is_home else "away",
                    })

    # Deduplicate by date + opponent (same match could appear in multiple fetches)
    seen = set()
    unique = []
    for m in all_matches:
        key = f"{m.get('date', '')}:{_normalize_team_name(m.get('opponent', ''))}"
        if key not in seen:
            seen.add(key)
            unique.append(m)

    # Sort by date
    unique.sort(key=lambda x: x.get("date") or datetime.min.replace(tzinfo=timezone.utc))

    _set_cache(cache_key, unique)
    logger.info(f"[SOCCER CONGESTION] Found {len(unique)} matches for {team_name} ({days_back}d back, {days_forward}d fwd)")
    return unique


def _analyze_congestion(matches: list, team_name: str) -> dict:
    """Analyze fixture congestion from a team's match list.

    Returns dict with:
        - days_since_last: int or None
        - matches_in_14d: int
        - matches_in_10d: int
        - last_match: dict or None
        - next_match: dict or None
        - days_until_next: int or None
        - last_was_european: bool
        - last_was_european_away: bool
        - tag: str (FRESH, NORMAL, CONGESTED, EUROPEAN HANGOVER, ROTATION RISK)
        - rotation_risk: str (LOW, MODERATE, HIGH, VERY HIGH)
    """
    now = datetime.now(timezone.utc)

    past_matches = [m for m in matches if m.get("date") and m["date"] < now]
    future_matches = [m for m in matches if m.get("date") and m["date"] >= now]

    # Days since last match
    last_match = past_matches[-1] if past_matches else None
    days_since_last = None
    if last_match and last_match.get("date"):
        delta = now - last_match["date"]
        days_since_last = delta.days

    # Next match
    next_match = future_matches[0] if future_matches else None
    days_until_next = None
    if next_match and next_match.get("date"):
        delta = next_match["date"] - now
        days_until_next = delta.days

    # Congestion counts
    cutoff_14d = now - timedelta(days=14)
    cutoff_10d = now - timedelta(days=10)
    matches_in_14d = sum(1 for m in past_matches if m.get("date") and m["date"] >= cutoff_14d)
    matches_in_10d = sum(1 for m in past_matches if m.get("date") and m["date"] >= cutoff_10d)

    # European match analysis
    last_was_european = bool(last_match and last_match.get("is_european"))
    last_was_european_away = bool(last_was_european and last_match.get("venue") == "away")

    # Determine congestion tag
    tag = "NORMAL"
    if days_since_last is not None:
        if days_since_last >= 7 and not last_was_european:
            tag = "FRESH"
        elif days_since_last <= 3 and last_was_european:
            tag = "EUROPEAN HANGOVER"
        elif matches_in_10d >= 3:
            tag = "CONGESTED"
        elif days_since_last >= 4 and days_since_last <= 6:
            tag = "NORMAL"

    # Override: if congested AND important next match within 3 days, rotation risk
    if tag in ("CONGESTED", "EUROPEAN HANGOVER") and days_until_next is not None and days_until_next <= 3:
        if next_match and next_match.get("is_european"):
            tag = "ROTATION RISK"

    # Rotation risk level
    rotation_risk = "LOW"
    if matches_in_10d >= 3 and last_was_european:
        rotation_risk = "VERY HIGH"
    elif matches_in_10d >= 3 or (last_was_european_away and days_since_last is not None and days_since_last <= 3):
        rotation_risk = "HIGH"
    elif last_was_european and days_since_last is not None and days_since_last <= 4:
        rotation_risk = "MODERATE"

    return {
        "team": team_name,
        "days_since_last": days_since_last,
        "matches_in_14d": matches_in_14d,
        "matches_in_10d": matches_in_10d,
        "last_match": last_match,
        "next_match": next_match,
        "days_until_next": days_until_next,
        "last_was_european": last_was_european,
        "last_was_european_away": last_was_european_away,
        "tag": tag,
        "rotation_risk": rotation_risk,
    }


def _format_last_match(analysis: dict) -> str:
    """Format last match info for context output."""
    lm = analysis.get("last_match")
    if not lm:
        return "No recent match data"

    days = analysis["days_since_last"]
    comp = lm.get("comp_type", "?")
    venue = lm.get("venue", "?")
    opponent = lm.get("opponent", "?")

    european_note = ""
    if lm.get("is_european"):
        european_note = f" ({comp} {venue}, {opponent})"
    else:
        european_note = f" ({comp})"

    return f"Last match {days} days ago{european_note}"


def _format_next_match(analysis: dict) -> str:
    """Format next match info for context output."""
    nm = analysis.get("next_match")
    if not nm:
        return "No upcoming match data"

    days = analysis["days_until_next"]
    comp = nm.get("comp_type", "?")

    return f"Next match in {days}d ({comp})"


async def build_congestion_context(home_team: str, away_team: str) -> str:
    """Build the fixture congestion context block for the AI analysis prompt.

    Args:
        home_team: Home team name
        away_team: Away team name

    Returns formatted text like:
    === FIXTURE CONGESTION ===
    RULE: Fixture congestion is the #1 edge in soccer. ...

      Arsenal: Last match 3 days ago (UCL away, Milan) | 3 matches in 14d | ...
      Fulham: Last match 7 days ago (League) | 1 match in 14d | ...
      Congestion edge: Fulham (fresh vs congested European traveler)
    """
    # Fetch schedules for both teams
    home_schedule = await fetch_team_schedule(home_team)
    away_schedule = await fetch_team_schedule(away_team)

    # Analyze congestion
    home_analysis = _analyze_congestion(home_schedule, home_team)
    away_analysis = _analyze_congestion(away_schedule, away_team)

    lines = ["=== FIXTURE CONGESTION ==="]
    lines.append(
        "RULE: Fixture congestion is the #1 edge in soccer. Teams playing midweek "
        "UCL/UEL + weekend league rotate 3-5 players on average. 3+ matches in 10 "
        "days = heavy congestion, score fixture_congestion 8-9 for opponent. Travel "
        "from European away match + domestic game within 72hrs = FADE."
    )
    lines.append("")

    for analysis in [home_analysis, away_analysis]:
        team = analysis["team"]
        last_info = _format_last_match(analysis)
        next_info = _format_next_match(analysis)
        m14 = analysis["matches_in_14d"]
        m10 = analysis["matches_in_10d"]
        tag = analysis["tag"]
        rotation = analysis["rotation_risk"]

        travel_note = ""
        if analysis["last_was_european_away"]:
            travel_note = " (European travel)"

        lines.append(
            f"  {team}: {last_info} | {m14} matches in 14d ({m10} in 10d) | "
            f"{next_info} — {tag}{travel_note}"
        )
        if rotation != "LOW":
            lines.append(f"    Rotation risk: {rotation}")

    lines.append("")

    # Determine congestion edge
    edge = _determine_edge(home_analysis, away_analysis)
    if edge:
        lines.append(f"  Congestion edge: {edge}")
    else:
        lines.append("  Congestion edge: No significant congestion differential")

    lines.append("")
    return "\n".join(lines)


def _determine_edge(home: dict, away: dict) -> str:
    """Determine which team has the congestion edge (fresher team benefits)."""
    fresh_tags = {"FRESH", "NORMAL"}
    tired_tags = {"CONGESTED", "EUROPEAN HANGOVER", "ROTATION RISK"}

    home_tag = home["tag"]
    away_tag = away["tag"]

    # Clear edge: one fresh, one tired
    if home_tag in fresh_tags and away_tag in tired_tags:
        reason = _edge_reason(away)
        return f"{home['team']} (fresh vs {reason})"
    if away_tag in fresh_tags and home_tag in tired_tags:
        reason = _edge_reason(home)
        return f"{away['team']} (fresh vs {reason})"

    # Both congested but one has European travel
    if home["last_was_european_away"] and not away["last_was_european_away"]:
        return f"{away['team']} (opponent has European away travel fatigue)"
    if away["last_was_european_away"] and not home["last_was_european_away"]:
        return f"{home['team']} (opponent has European away travel fatigue)"

    # Slight edge from rest differential
    h_days = home.get("days_since_last")
    a_days = away.get("days_since_last")
    if h_days is not None and a_days is not None:
        diff = abs(h_days - a_days)
        if diff >= 3:
            fresher = home["team"] if h_days > a_days else away["team"]
            return f"{fresher} ({diff} extra days rest)"

    return ""


def _edge_reason(tired_team: dict) -> str:
    """Build a short reason string for the tired team."""
    parts = []
    if tired_team["tag"] == "EUROPEAN HANGOVER":
        parts.append("congested European traveler")
    elif tired_team["tag"] == "ROTATION RISK":
        parts.append("rotation risk, fixture pile-up")
    elif tired_team["tag"] == "CONGESTED":
        parts.append(f"congested — {tired_team['matches_in_10d']} matches in 10d")
    if tired_team["last_was_european_away"]:
        parts.append("European away travel")
    return ", ".join(parts) if parts else "congested schedule"

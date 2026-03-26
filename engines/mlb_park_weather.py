"""
MLB Park Factors + Weather Engine — Static park data + live weather for game context.

OpenWeatherMap API: https://api.openweathermap.org/data/2.5/weather
Free tier, requires API key. Rate limit: 60 req/min.

Used by: server.py analysis prompt (park + weather context injection)
Cache: 1 hour for weather data, park factors are static
"""

import os
import httpx
import logging
from datetime import datetime, timezone

logger = logging.getLogger("edge-crew")

OPENWEATHERMAP_KEY = os.environ.get("OPENWEATHERMAP_API_KEY", "")
OPENWEATHERMAP_BASE = "https://api.openweathermap.org/data/2.5/weather"

# Module-level cache (same pattern as mlb_pitcher_engine.py)
_weather_cache = {}


def _get_cached(key, ttl=3600):
    """Return cached data if within TTL, else None."""
    if key in _weather_cache:
        data, ts = _weather_cache[key]
        if (datetime.now(timezone.utc) - ts).total_seconds() < ttl:
            return data
    return None


def _set_cache(key, data):
    """Store data in module-level cache with current timestamp."""
    _weather_cache[key] = (data, datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# PARK FACTORS — All 30 MLB parks (2024 season baselines)
#
# runs_factor / hr_factor: 1.0 = league average
# type: "hitter" (>1.05), "pitcher" (<0.95), "neutral" (0.95-1.05)
# roof: True (fixed dome), "retractable", False (open air)
# ---------------------------------------------------------------------------

PARK_FACTORS = {
    "ARI": {
        "park_name": "Chase Field",
        "runs_factor": 1.06,
        "hr_factor": 1.12,
        "type": "hitter",
        "roof": "retractable",
        "altitude_ft": 1082,
        "lat": 33.4455,
        "lon": -112.0667,
    },
    "ATL": {
        "park_name": "Truist Park",
        "runs_factor": 1.01,
        "hr_factor": 1.04,
        "type": "neutral",
        "roof": False,
        "altitude_ft": 1050,
        "lat": 33.8908,
        "lon": -84.4678,
    },
    "BAL": {
        "park_name": "Camden Yards",
        "runs_factor": 1.05,
        "hr_factor": 1.10,
        "type": "hitter",
        "roof": False,
        "altitude_ft": 33,
        "lat": 39.2838,
        "lon": -76.6216,
    },
    "BOS": {
        "park_name": "Fenway Park",
        "runs_factor": 1.08,
        "hr_factor": 0.96,
        "type": "hitter",
        "roof": False,
        "altitude_ft": 20,
        "lat": 42.3467,
        "lon": -71.0972,
    },
    "CHC": {
        "park_name": "Wrigley Field",
        "runs_factor": 1.05,
        "hr_factor": 1.08,
        "type": "hitter",
        "roof": False,
        "altitude_ft": 600,
        "lat": 41.9484,
        "lon": -87.6553,
    },
    "CWS": {
        "park_name": "Guaranteed Rate Field",
        "runs_factor": 1.04,
        "hr_factor": 1.12,
        "type": "neutral",
        "roof": False,
        "altitude_ft": 595,
        "lat": 41.8299,
        "lon": -87.6338,
    },
    "CIN": {
        "park_name": "Great American Ball Park",
        "runs_factor": 1.15,
        "hr_factor": 1.22,
        "type": "hitter",
        "roof": False,
        "altitude_ft": 490,
        "lat": 39.0974,
        "lon": -84.5065,
    },
    "CLE": {
        "park_name": "Progressive Field",
        "runs_factor": 0.97,
        "hr_factor": 0.95,
        "type": "neutral",
        "roof": False,
        "altitude_ft": 653,
        "lat": 41.4962,
        "lon": -81.6852,
    },
    "COL": {
        "park_name": "Coors Field",
        "runs_factor": 1.28,
        "hr_factor": 1.39,
        "type": "hitter",
        "roof": False,
        "altitude_ft": 5200,
        "lat": 39.7559,
        "lon": -104.9942,
    },
    "DET": {
        "park_name": "Comerica Park",
        "runs_factor": 0.96,
        "hr_factor": 0.93,
        "type": "pitcher",
        "roof": False,
        "altitude_ft": 600,
        "lat": 42.3390,
        "lon": -83.0485,
    },
    "HOU": {
        "park_name": "Minute Maid Park",
        "runs_factor": 1.03,
        "hr_factor": 1.07,
        "type": "neutral",
        "roof": "retractable",
        "altitude_ft": 42,
        "lat": 29.7573,
        "lon": -95.3555,
    },
    "KC": {
        "park_name": "Kauffman Stadium",
        "runs_factor": 0.97,
        "hr_factor": 0.90,
        "type": "neutral",
        "roof": False,
        "altitude_ft": 750,
        "lat": 39.0517,
        "lon": -94.4803,
    },
    "LAA": {
        "park_name": "Angel Stadium",
        "runs_factor": 0.96,
        "hr_factor": 0.97,
        "type": "pitcher",
        "roof": False,
        "altitude_ft": 160,
        "lat": 33.8003,
        "lon": -117.8827,
    },
    "LAD": {
        "park_name": "Dodger Stadium",
        "runs_factor": 0.95,
        "hr_factor": 0.98,
        "type": "pitcher",
        "roof": False,
        "altitude_ft": 515,
        "lat": 34.0739,
        "lon": -118.2400,
    },
    "MIA": {
        "park_name": "LoanDepot Park",
        "runs_factor": 0.90,
        "hr_factor": 0.85,
        "type": "pitcher",
        "roof": "retractable",
        "altitude_ft": 7,
        "lat": 25.7781,
        "lon": -80.2196,
    },
    "MIL": {
        "park_name": "American Family Field",
        "runs_factor": 1.02,
        "hr_factor": 1.06,
        "type": "neutral",
        "roof": "retractable",
        "altitude_ft": 635,
        "lat": 43.0280,
        "lon": -87.9712,
    },
    "MIN": {
        "park_name": "Target Field",
        "runs_factor": 1.00,
        "hr_factor": 0.98,
        "type": "neutral",
        "roof": False,
        "altitude_ft": 815,
        "lat": 44.9818,
        "lon": -93.2775,
    },
    "NYM": {
        "park_name": "Citi Field",
        "runs_factor": 0.93,
        "hr_factor": 0.89,
        "type": "pitcher",
        "roof": False,
        "altitude_ft": 10,
        "lat": 40.7571,
        "lon": -73.8458,
    },
    "NYY": {
        "park_name": "Yankee Stadium",
        "runs_factor": 1.10,
        "hr_factor": 1.20,
        "type": "hitter",
        "roof": False,
        "altitude_ft": 55,
        "lat": 40.8296,
        "lon": -73.9262,
    },
    "OAK": {
        "park_name": "Oakland Coliseum",
        "runs_factor": 0.94,
        "hr_factor": 0.88,
        "type": "pitcher",
        "roof": False,
        "altitude_ft": 15,
        "lat": 37.7516,
        "lon": -122.2005,
    },
    "PHI": {
        "park_name": "Citizens Bank Park",
        "runs_factor": 1.07,
        "hr_factor": 1.14,
        "type": "hitter",
        "roof": False,
        "altitude_ft": 20,
        "lat": 39.9061,
        "lon": -75.1665,
    },
    "PIT": {
        "park_name": "PNC Park",
        "runs_factor": 0.94,
        "hr_factor": 0.91,
        "type": "pitcher",
        "roof": False,
        "altitude_ft": 730,
        "lat": 40.4468,
        "lon": -80.0057,
    },
    "SD": {
        "park_name": "Petco Park",
        "runs_factor": 0.91,
        "hr_factor": 0.85,
        "type": "pitcher",
        "roof": False,
        "altitude_ft": 13,
        "lat": 32.7076,
        "lon": -117.1570,
    },
    "SF": {
        "park_name": "Oracle Park",
        "runs_factor": 0.88,
        "hr_factor": 0.82,
        "type": "pitcher",
        "roof": False,
        "altitude_ft": 2,
        "lat": 37.7786,
        "lon": -122.3893,
    },
    "SEA": {
        "park_name": "T-Mobile Park",
        "runs_factor": 0.93,
        "hr_factor": 0.90,
        "type": "pitcher",
        "roof": "retractable",
        "altitude_ft": 17,
        "lat": 47.5914,
        "lon": -122.3325,
    },
    "STL": {
        "park_name": "Busch Stadium",
        "runs_factor": 0.98,
        "hr_factor": 1.01,
        "type": "neutral",
        "roof": False,
        "altitude_ft": 455,
        "lat": 38.6226,
        "lon": -90.1928,
    },
    "TB": {
        "park_name": "Tropicana Field",
        "runs_factor": 0.96,
        "hr_factor": 0.94,
        "type": "pitcher",
        "roof": True,
        "altitude_ft": 44,
        "lat": 27.7682,
        "lon": -82.6534,
    },
    "TEX": {
        "park_name": "Globe Life Field",
        "runs_factor": 1.02,
        "hr_factor": 1.05,
        "type": "neutral",
        "roof": "retractable",
        "altitude_ft": 525,
        "lat": 32.7473,
        "lon": -97.0845,
    },
    "TOR": {
        "park_name": "Rogers Centre",
        "runs_factor": 1.01,
        "hr_factor": 1.06,
        "type": "neutral",
        "roof": "retractable",
        "altitude_ft": 266,
        "lat": 43.6414,
        "lon": -79.3894,
    },
    "WSH": {
        "park_name": "Nationals Park",
        "runs_factor": 1.00,
        "hr_factor": 1.02,
        "type": "neutral",
        "roof": False,
        "altitude_ft": 25,
        "lat": 38.8730,
        "lon": -77.0074,
    },
}

# Default for unknown teams
_NEUTRAL_PARK = {
    "park_name": "Unknown",
    "runs_factor": 1.00,
    "hr_factor": 1.00,
    "type": "neutral",
    "roof": False,
    "altitude_ft": 0,
    "lat": 0.0,
    "lon": 0.0,
}


def get_park_factor(team_abbr: str) -> dict:
    """Sync lookup for a team's park factors. Used by other engines that need static data.

    Returns full park dict with park_name, runs_factor, hr_factor, type, roof, altitude_ft, lat, lon.
    Returns neutral defaults for unknown teams.
    """
    return PARK_FACTORS.get(team_abbr, _NEUTRAL_PARK).copy()


def _is_dome(park: dict) -> bool:
    """Return True if the park has a fixed dome (weather irrelevant)."""
    return park.get("roof") is True


def _wind_label(deg: float) -> str:
    """Convert wind degrees to compass direction."""
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(deg / 22.5) % 16
    return dirs[idx]


async def fetch_game_weather(lat: float, lon: float) -> dict:
    """Fetch current weather for a given lat/lon from OpenWeatherMap.

    Returns:
        {"temp_f": 72, "wind_speed_mph": 8, "wind_direction": "SW",
         "humidity": 55, "conditions": "Clear", "description": "clear sky"}

    Returns empty dict if no API key, API fails, or lat/lon is 0.
    Cache: 1 hour per lat/lon pair.
    """
    if not OPENWEATHERMAP_KEY or (lat == 0.0 and lon == 0.0):
        return {}

    cache_key = f"weather:{lat:.4f},{lon:.4f}"
    cached = _get_cached(cache_key, ttl=3600)  # 1 hour
    if cached:
        return cached

    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHERMAP_KEY,
        "units": "imperial",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(OPENWEATHERMAP_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[PARK WEATHER] Weather fetch failed for ({lat}, {lon}): {e}")
        return {}

    wind = data.get("wind", {})
    weather_list = data.get("weather", [{}])
    main = data.get("main", {})

    result = {
        "temp_f": round(main.get("temp", 0)),
        "wind_speed_mph": round(wind.get("speed", 0)),
        "wind_direction": _wind_label(wind.get("deg", 0)),
        "humidity": main.get("humidity", 0),
        "conditions": weather_list[0].get("main", "Unknown") if weather_list else "Unknown",
        "description": weather_list[0].get("description", "") if weather_list else "",
    }

    _set_cache(cache_key, result)
    logger.info(f"[PARK WEATHER] Weather for ({lat}, {lon}): {result['temp_f']}F, {result['conditions']}")
    return result


async def build_park_weather_context(games: list) -> str:
    """Build the park factors + weather context block for the AI analysis prompt.

    Args:
        games: List of game dicts with "away" and "home" team abbreviations.
               e.g. [{"away": "NYY", "home": "BOS"}, {"away": "COL", "home": "SD"}]

    Returns formatted text block:
        === PARK FACTORS + WEATHER ===
        RULE: Park factors significantly impact totals...
        COL (Coors Field): HITTER'S PARK (1.28 runs, 1.39 HR) | 72F | Wind: 8mph SW | Clear
    """
    if not games:
        return "=== PARK FACTORS + WEATHER ===\nNo games provided for park/weather context."

    lines = ["=== PARK FACTORS + WEATHER ==="]
    lines.append(
        "RULE: Park factors significantly impact totals. Coors Field adds ~1.5 runs. "
        "Pitcher parks suppress scoring. Wind out = over lean, wind in = under lean."
    )
    lines.append("")

    for game in games:
        home = game.get("home", "")
        away = game.get("away", "")
        park = get_park_factor(home)
        park_name = park["park_name"]
        runs = park["runs_factor"]
        hr = park["hr_factor"]
        park_type = park["type"].upper()

        # Park type label
        if park["type"] == "hitter":
            type_label = "HITTER'S PARK"
        elif park["type"] == "pitcher":
            type_label = "PITCHER'S PARK"
        else:
            type_label = "NEUTRAL"

        # Weather
        if _is_dome(park):
            weather_str = "Indoor — no weather impact"
            lines.append(
                f"{away} @ {home} ({park_name}): {type_label} "
                f"({runs:.2f} runs, {hr:.2f} HR) | {weather_str}"
            )
        else:
            weather = await fetch_game_weather(park["lat"], park["lon"])
            if weather:
                temp = weather["temp_f"]
                wind_spd = weather["wind_speed_mph"]
                wind_dir = weather["wind_direction"]
                humidity = weather["humidity"]
                conditions = weather["conditions"]
                weather_str = (
                    f"{temp}\u00b0F | Wind: {wind_spd}mph {wind_dir} | "
                    f"Humidity: {humidity}% | {conditions}"
                )
            else:
                weather_str = "Weather unavailable"

            lines.append(
                f"{away} @ {home} ({park_name}): {type_label} "
                f"({runs:.2f} runs, {hr:.2f} HR) | {weather_str}"
            )

    return "\n".join(lines)

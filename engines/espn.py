"""ESPN adapter — bridges Casa EV engines to Edge Crew's existing ESPN functions.
Imports from server.py's existing functions so we don't duplicate code."""

# These get wired up at import time by server.py
# The actual implementations live in server.py already
PROP_STAT_TO_ESPN = {
    "Points": "PTS", "Rebounds": "REB", "Assists": "AST",
    "Threes": "3PM", "Three Pointers": "3PM",
    "Goals": "G", "Shots": "SOG", "Strikeouts": "K",
    "Total Bases": "TB", "Hits": "H",
    "Pass Yds": "YDS", "Rush Yds": "YDS", "Reception Yds": "YDS",
    "Pass Tds": "TD",
}

# These will be monkey-patched from server.py on startup
fetch_player_game_logs = None
get_rosters = None
fetch_injuries = None

"""Odds adapter — bridges Casa EV engines to Edge Crew's existing odds functions.
The actual implementations live in server.py already."""

# These will be monkey-patched from server.py on startup
get_player_props = None
get_events = None

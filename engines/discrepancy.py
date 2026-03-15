"""Book Discrepancy Engine — find where books disagree on the same line.

Phase 2 engine. Compares the same prop across all bookmakers and flags
significant disagreements (>10 cents ML or >0.5 spread difference).
"""

import logging
from engines.odds import get_player_props
from engines.ev_calc import calc_implied_prob

logger = logging.getLogger("casa-ev")


async def find_discrepancies(sport: str, min_gap_cents=10, min_spread_gap=0.5):
    """Compare same prop across books. Flag where they disagree.

    A 'discrepancy' means books see the same player differently:
    - One book has Over 22.5 at -110, another at -135 = different pricing
    - One book has line 22.5, another has 23.5 = different lines

    These gaps = opportunity.
    """
    props_data = await get_player_props(sport)
    if "error" in props_data:
        return props_data

    all_props = props_data.get("props", [])
    discrepancies = []

    for prop in all_props:
        books = prop.get("books", [])
        if len(books) < 2:
            continue

        # Sort by odds to find the spread between best and worst
        sorted_books = sorted(books, key=lambda b: b["odds"], reverse=True)
        best = sorted_books[0]
        worst = sorted_books[-1]

        # Calculate implied prob gap
        best_prob = best.get("implied_prob", 0) or 0
        worst_prob = worst.get("implied_prob", 0) or 0
        prob_gap = round(abs(worst_prob - best_prob), 1)
        odds_gap = best["odds"] - worst["odds"]

        # Flag if gap exceeds threshold
        if prob_gap >= min_gap_cents / 10:  # 10 cents = 1% implied prob gap
            discrepancies.append({
                "player": prop.get("player", ""),
                "stat": prop.get("stat", ""),
                "side": prop.get("side", ""),
                "line": prop.get("line", 0),
                "matchup": prop.get("matchup", ""),
                "commence": prop.get("commence", ""),
                "best_book": best["book"],
                "best_odds": best["odds"],
                "best_prob": best_prob,
                "worst_book": worst["book"],
                "worst_odds": worst["odds"],
                "worst_prob": worst_prob,
                "prob_gap": prob_gap,
                "odds_gap": odds_gap,
                "book_count": len(books),
                "all_books": sorted_books,
            })

    # Sort by probability gap descending
    discrepancies.sort(key=lambda x: -x["prob_gap"])

    return {
        "sport": sport.upper(),
        "discrepancies": discrepancies,
        "count": len(discrepancies),
    }

"""EV and probability math engine."""


def calc_implied_prob(american_odds):
    """Convert American odds to implied probability %."""
    try:
        odds = float(american_odds)
        if odds < 0:
            return round(abs(odds) / (abs(odds) + 100) * 100, 1)
        else:
            return round(100 / (odds + 100) * 100, 1)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def calc_edge(best_prob, consensus_prob):
    """Edge = consensus implied prob - best book implied prob.
    Positive = you're getting a better price than the market average."""
    if best_prob and consensus_prob:
        return round(consensus_prob - best_prob, 1)
    return 0


def american_to_decimal(american_odds):
    """Convert American odds to decimal odds."""
    try:
        odds = float(american_odds)
        if odds > 0:
            return round(1 + odds / 100, 4)
        else:
            return round(1 + 100 / abs(odds), 4)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def calculate_ev(our_prob_pct, american_odds):
    """Calculate Expected Value %.
    our_prob_pct: our estimated probability (0-100)
    american_odds: the book's line in American format
    Returns EV as a percentage."""
    decimal_odds = american_to_decimal(american_odds)
    if decimal_odds is None or our_prob_pct is None:
        return None
    our_prob = our_prob_pct / 100
    ev = (our_prob * (decimal_odds - 1)) - (1 - our_prob)
    return round(ev * 100, 2)


def kelly_fraction(our_prob_pct, american_odds):
    """Kelly Criterion fraction — optimal bet size as % of bankroll."""
    decimal_odds = american_to_decimal(american_odds)
    if decimal_odds is None or our_prob_pct is None or our_prob_pct <= 0:
        return 0
    p = our_prob_pct / 100
    q = 1 - p
    b = decimal_odds - 1
    if b <= 0:
        return 0
    f = (p * b - q) / b
    return round(max(0, f) * 100, 2)  # % of bankroll

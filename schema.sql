-- Edge Crew — Postgres Schema (Phase 1: Dual-Write Tables Only)
-- Run via db.init_schema() on startup
-- Phase 2 will add teams, rosters, odds, injuries, lineups, etc.

-- ============================================================
-- PICKS — mirrors picks.json
-- ============================================================

CREATE TABLE IF NOT EXISTS picks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sport TEXT NOT NULL,
    type TEXT,
    matchup TEXT,
    selection TEXT,
    odds TEXT,
    units TEXT DEFAULT '1',
    confidence TEXT,
    notes TEXT,
    date DATE,
    time TEXT,
    placed BOOLEAN DEFAULT TRUE,
    placed_at TIMESTAMPTZ,
    result TEXT,
    graded_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_picks_name_date ON picks(name, date);
CREATE INDEX IF NOT EXISTS idx_picks_result ON picks(result);
CREATE INDEX IF NOT EXISTS idx_picks_sport ON picks(sport);

-- ============================================================
-- CREW PROFILES — mirrors crew_profiles.json
-- ============================================================

CREATE TABLE IF NOT EXISTS crew_profiles (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    pin_hash TEXT NOT NULL,
    color TEXT DEFAULT '#D4A017',
    is_admin BOOLEAN DEFAULT FALSE,
    last_login TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- SCORES — mirrors scores_archive.json
-- ============================================================

CREATE TABLE IF NOT EXISTS scores (
    id TEXT PRIMARY KEY,
    sport TEXT,
    home_team TEXT,
    away_team TEXT,
    home_score INTEGER,
    away_score INTEGER,
    completed BOOLEAN DEFAULT FALSE,
    commence_time TEXT,
    fetched_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_scores_sport ON scores(sport);
CREATE INDEX IF NOT EXISTS idx_scores_completed ON scores(completed);

-- ============================================================
-- BANKROLL — mirrors bankroll.json
-- ============================================================

CREATE TABLE IF NOT EXISTS bankroll_settings (
    name TEXT PRIMARY KEY,
    starting_bankroll REAL DEFAULT 0,
    unit_size REAL DEFAULT 5,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

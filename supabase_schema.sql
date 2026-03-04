-- Edge Crew — Supabase Schema
-- Run this in the Supabase SQL Editor to create all 4 tables.

-- Crew picks (replaces data/picks.json + localStorage edgePicks)
CREATE TABLE picks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sport TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT 'Spread',
    matchup TEXT NOT NULL,
    selection TEXT NOT NULL,
    odds TEXT DEFAULT '-110',
    units TEXT DEFAULT '1',
    confidence TEXT DEFAULT 'Lean',
    notes TEXT DEFAULT '',
    date DATE NOT NULL,
    time TIME NOT NULL,
    placed BOOLEAN DEFAULT FALSE,
    placed_at TIMESTAMPTZ,
    result TEXT,
    graded_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_picks_date ON picks(date);

-- Upset specials (replaces data/upsets.json)
CREATE TABLE upsets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sport TEXT NOT NULL,
    team TEXT NOT NULL,
    odds TEXT NOT NULL,
    thesis TEXT DEFAULT '',
    date DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_upsets_date ON upsets(date);

-- Bankroll settings (single row, shared by crew — replaces localStorage)
CREATE TABLE bankroll_settings (
    id SERIAL PRIMARY KEY,
    starting_balance NUMERIC(10,2) DEFAULT 1000.00,
    unit_size NUMERIC(10,2) DEFAULT 25.00,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_by TEXT DEFAULT ''
);

-- Gotcha notes (one row per sport — replaces localStorage edgeGotchaNBA)
CREATE TABLE gotcha_notes (
    id SERIAL PRIMARY KEY,
    sport TEXT NOT NULL UNIQUE,
    notes TEXT DEFAULT '',
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_by TEXT DEFAULT ''
);

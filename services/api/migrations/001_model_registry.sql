-- Migration: model_registry table for ml/inference/registry
-- Applied on PostgreSQL startup via docker-compose init volume.
CREATE TABLE IF NOT EXISTS model_registry (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);

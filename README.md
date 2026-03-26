# Edge Crew V2 - Deployment Guide

## Overview
Multi-layer sports betting analysis pipeline with AI-powered grading and real-time dashboards.

## Quick Deploy to Render

### Option 1: Blueprint (Recommended)
1. Push this repo to GitHub
2. In Render Dashboard: **New +** → **Blueprint**
3. Connect your GitHub repo
4. Render will read `render.yaml` and auto-configure

### Option 2: Manual Web Service
1. **New +** → **Web Service**
2. Connect your repo
3. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT --workers 2`

## Required Environment Variables

Set these in Render Dashboard (Environment tab):

| Variable | Source | Required For |
|----------|--------|--------------|
| `AZURE_AI_KEY` | Azure AI / Foundry | Grok / DeepSeek inference key |
| `AZURE_AI_ENDPOINT` | Azure Portal | AI Services / Foundry inference endpoint |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI | GPT deployment key |
| `AZURE_OPENAI_ENDPOINT` | Azure Portal | Azure OpenAI endpoint |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI | API version for GPT deployments |
| `AZURE_GROK_DEPLOYMENT` | Azure AI / Foundry | Optional Grok deployment override |
| `AZURE_DEEPSEEK_DEPLOYMENT` | Azure AI / Foundry | Optional DeepSeek deployment override |
| `AZURE_GPT41_DEPLOYMENT` | Azure OpenAI | Optional GPT-4.1 deployment override |
| `RAPIDAPI_KEY` | RapidAPI Dashboard | Tank01 injuries API |
| `SPORTSGAMEODDS_KEY` | SportsGameOdds | Odds data |
| `BALLDONTLIE_API_KEY` | BallDontLie | NBA player props |

## Project Structure

```
edge_engine/
├── server.py              # FastAPI web server
├── run_pipeline.py        # Full data pipeline
├── grade_engine.py        # Math-based grading
├── model_caller.py        # AI model interface
├── roster_profile.py      # Team profiling (4 models)
├── player_profile.py      # Player chains
├── collect_*.py           # Data collectors
├── slate_dashboard/       # Web UI
├── data/                  # Game data (gitignored)
└── grades/                # Output grades (gitignored)
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Main dashboard |
| `GET /api/slate/{sport}` | Games + grades |
| `GET /api/race/{sport}` | H2H swim lane results |
| `GET /api/workbench/{sport}` | Full analysis |
| `GET /api/props/{sport}` | Player chains |
| `POST /api/collect/{sport}` | Trigger data collection |
| `POST /api/grade/{sport}` | Trigger grading |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set env vars (copy from .env.example)
cp .env.example .env
# Edit .env with your keys

# Run server
python server.py
# Or with uvicorn
uvicorn server:app --reload --port 8080

# Run full pipeline
python run_pipeline.py --sport NBA
```

## Pipeline Layers

1. **Layer 0**: Fetch schedule (ESPN)
2. **Layer 1**: Collect odds (SGO)
3. **Layer 1.5**: Team stats (ESPN)
4. **Layer 1.6**: Injuries (Tank01/ESPN)
5. **Layer 1.7**: Props (BDL)
6. **Layer 2**: Roster profiles (4 Azure models)
7. **Layer 2.5**: Player profiles + chains
8. **Layer 3**: Grade engine (15+ variables)
9. **Layer 3.5**: Swim lane H2H race

## Post-Deploy Verification

Check these URLs after deploy:
- `https://your-service.onrender.com/` - Dashboard loads
- `https://your-service.onrender.com/api/sports` - Returns sports list
- `https://your-service.onrender.com/api/slate/nba?date=20260321` - Returns games

## Troubleshooting

**Issue**: Azure model env vars not set
- Set `AZURE_AI_KEY` / `AZURE_AI_ENDPOINT` for Grok and DeepSeek
- Set `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` for GPT deployments

**Issue**: `No games found`
- Run pipeline: `python run_pipeline.py --sport NBA`
- Or hit `/api/collect/nba` endpoint

**Issue**: Dashboard shows no data
- Ensure data/ and grades/ directories exist
- Check that pipeline has been run

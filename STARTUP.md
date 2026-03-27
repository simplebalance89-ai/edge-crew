# Render Startup Checklist

## Pre-Deploy
- [ ] All API keys available
- [ ] GitHub repo created
- [ ] `render.yaml` committed

## Deploy Steps
1. Push to GitHub
2. New Web Service in Render
3. Connect repo
4. Add environment variables
5. Deploy

## Post-Deploy
1. Visit root URL - should show dashboard
2. Visit `/api/sports` - should return empty list initially
3. Trigger collection: `POST /api/collect/NBA`
4. Check `/api/slate/NBA` - should show games
5. Trigger grading: `POST /api/grade/NBA`
6. Dashboard should populate with grades

## Environment Variables Template
```
AZURE_AI_KEY=<your-foundry-or-ai-services-key>
AZURE_AI_ENDPOINT=https://<your-resource>.services.ai.azure.com/openai/v1/
AZURE_OPENAI_API_KEY=<your-azure-openai-key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
RAPIDAPI_KEY=<your-rapidapi-key>
SPORTSGAMEODDS_KEY=<your-sgo-key>
BALLDONTLIE_API_KEY=<your-bdl-key>
```

# Math Chatbot HITL Frontend (React + Vite)

Single-page React app that connects to a FastAPI backend (`http://localhost:8000`) and implements Human-in-the-Loop (HITL) validation: ask → validate/refine (loop) → get final.

## Requirements
- Node.js 16+
- FastAPI backend running on `http://localhost:8000` with endpoints:
  - `POST /ask` `{ session_id, query }` -> `{ intermediate_answer }`
  - `POST /refine` `{ session_id, feedback }` -> `{ refined_intermediate_answer }`
  - `POST /final` `{ session_id, query }` -> `{ final_answer }`

## Run locally
```bash
cd frontend
npm install
npm run dev
# Open the printed URL (default http://localhost:3000)
```

No environment config needed; the base URL is hardcoded to `http://localhost:8000`.

## Notes
- Each browser tab gets a unique `session_id` via uuid v4
- Do not modify backend agent names, model names, or LangGraph workflow
- CORS: ensure FastAPI allows `http://localhost:3000`

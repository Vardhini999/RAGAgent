from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import ingest, query

app = FastAPI(
    title="Routing API",
    description="Interact with your Documents using Adaptive Agent Routing",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/healthz")
async def health():
    return {"status": "ok"}

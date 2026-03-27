from fastapi import FastAPI

from api.routers.health import router as health_router

app = FastAPI(
    title="sa-fraud-shield",
    description="Real-time SA fraud detection API",
    version="0.1.0",
)

app.include_router(health_router)


@app.get("/")
async def root():
    return {"message": "sa-fraud-shield API is running"}

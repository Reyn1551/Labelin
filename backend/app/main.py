import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import capture, annotate, dataset, train

app = FastAPI(
    title="Labelin Web API Suite",
    description="Backend API for Labelin - Traffic Object Detection Suite with Roboflow UI",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(capture.router)
app.include_router(annotate.router)
app.include_router(dataset.router)
app.include_router(train.router)

# Mount Static Directories for Images
for folder in ["dataset_raw", "dataset_manual", "dataset_labeled", "dataset"]:
    os.makedirs(folder, exist_ok=True)
    app.mount(f"/static/{folder}", StaticFiles(directory=folder), name=folder)

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "service": "Labelin Web Backend",
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

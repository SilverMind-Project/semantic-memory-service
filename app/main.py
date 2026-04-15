from fastapi import FastAPI
from app.config.config import settings

app = FastAPI(title=settings.PROJECT_NAME)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8300)

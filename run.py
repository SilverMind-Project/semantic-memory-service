import uvicorn
from app.main import app

if __name__ == "__main__":
    # In production, port would be 8300 as per design
    uvicorn.run(app, host="0.0.0.0", port=8300)

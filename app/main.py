"""
The main module for running the FastAPI application.
"""

import os
from fastapi import FastAPI
from app.api import router as image_router

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()
app.include_router(image_router)


@app.get("/health")
async def health():
    """
    Endpoint for checking server status.
    Returns "ok" status on successful access.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

# Import routes correctly
from app.api.routes import router

from fastapi import FastAPI

# Create FastAPI instance
app = FastAPI()

# Include the routes from `app/api/routes.py`
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)

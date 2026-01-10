import uvicorn
from fastapi import FastAPI
from app.core.config import settings
from app.routers import fundamentals
# Import other routers like: from app.routers import rag, agents
from app.core.exceptions import AIModelError, ai_model_exception_handler, generic_exception_handler

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Professional AI Backend for specialized role-based learning."
)

# Exception Handlers
app.add_exception_handler(AIModelError, ai_model_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# register routers
app.include_router(fundamentals.router, prefix="/fundamentals", tags=["Module 1: Fundamentals"])
# app.include_router(rag.router, prefix="/rag", tags=["Module 2: RAG"])

@app.get("/")
def root():
    return {"message": "Synapse-Core is online. Go to /docs for the API."}

if __name__ == "__main__":
    # This allows you to run python main.py
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
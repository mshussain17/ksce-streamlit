import os

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_function import ask_ai
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import uvicorn

# CORS Middleware
from fastapi.middleware.cors import CORSMiddleware


# FastAPI app
app = FastAPI()

# Allowed Regions [if enabled - only these urls will be allowed to request]:
origins = ["*"]

# Allowed Methods:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai_router = APIRouter(prefix="/genai")

class Input(BaseModel):
    query: str

@app.get("/genai/health")
def health_check():
    return {"status": "healthy"}


@genai_router.get("/")
def healthcheck():
    return {"status": "healthy"}

@genai_router.post("/answer/")
def answer(inp: Input):
    return ask_ai(inp.query)

# Include the router in the main app
app.include_router(genai_router)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)

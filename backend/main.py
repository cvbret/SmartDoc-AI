from fastapi import FastAPI

app = FastAPI()

@app.get("/")

def home():
    return {"message": "SmartDoc AI Backend is running!"}

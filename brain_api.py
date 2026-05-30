from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# Modules
from avatar_router import router as avatar_router
from style_generator import router as style_router
from appointment_router import router as appointment_router
from smart_router import router as smart_router
from email_service import send_waitlist_email

app = FastAPI(title="AI Beauty Concepts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach All Routers
app.include_router(avatar_router)
app.include_router(style_router)
app.include_router(appointment_router)
app.include_router(smart_router)

class WaitlistRequest(BaseModel):
    email: str

@app.get("/")
async def health():
    return {"status": "online", "msg": "All systems active"}

@app.post("/waitlist")
async def handle_waitlist(data: WaitlistRequest, background_tasks: BackgroundTasks):
    email = data.email.strip().lower()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    
    # Save to file
    os.makedirs("/home/user/barber_ai/data", exist_ok=True)
    with open("/home/user/barber_ai/data/subscribers.txt", "a") as f:
        f.write(f"{email}\n")
    
    # Send Email
    background_tasks.add_task(send_waitlist_email, email)
    return {"status": "ok", "message": "Joined waitlist"}

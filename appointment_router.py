from fastapi import APIRouter
from pydantic import BaseModel
from appointment_db import get_providers, add_slot
from appointment_scheduler import find_available_slots, book_appointment

router = APIRouter(prefix="/appointments", tags=["appointments"])

class SlotRequest(BaseModel):
    provider_id: str
    start_utc: str
    end_utc: str

class AvailabilityRequest(BaseModel):
    provider_id: str
    service_name: str

class BookingRequest(BaseModel):
    provider_id: str
    service_name: str
    start_utc: str
    patron_name: str
    patron_email: str
    style_image: str
    session_id: str

@router.get("/providers")
async def list_providers():
    return get_providers()

@router.post("/add_slot")
async def add_provider_slot(payload: SlotRequest):
    add_slot({
        "provider_id": payload.provider_id,
        "start": payload.start_utc,
        "end": payload.end_utc
    })
    return {"status": "slot added"}

@router.post("/availability")
async def check_availability(payload: AvailabilityRequest):
    slots = find_available_slots(payload.provider_id, payload.service_name)
    return {"available_slots": slots}

@router.post("/book")
async def book(payload: BookingRequest):
    booking = book_appointment(
        payload.provider_id,
        payload.service_name,
        payload.start_utc,
        payload.patron_name,
        payload.patron_email,
        payload.style_image,
        payload.session_id
    )
    return booking

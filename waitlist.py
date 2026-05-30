from fastapi import APIRouter, Request
import datetime

router = APIRouter()

@router.post("/waitlist")
async def waitlist(request: Request):
    try:
        email = None

        try:
            data = await request.json()
            email = data.get("email")
        except:
            pass

        if not email:
            form = await request.form()
            email = form.get("email")

        if not email:
            return {"error": "email missing"}

        with open("/var/www/html/subscribers.txt", "a") as f:
            f.write(f"{email} | {datetime.datetime.now()}\n")

        try:
            from email_service import send_admin_notification, send_user_receipt
            send_admin_notification(email)
            send_user_receipt(email)
        except:
            print("⚠️ email not configured")

        return {"status": "saved"}

    except Exception as e:
        return {"error": str(e)}

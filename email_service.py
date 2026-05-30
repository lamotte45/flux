import smtplib
import time
import logging
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr

# --- CONFIG LOGGING ---
LOG_FILE = "/home/user/barber_ai/email_sys.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def send_waitlist_email(user_email):
    sender_email = "discokenny777@gmail.com"
    app_password = "teqmwnwolorhbgoe"
    brand_name = "AI Beauty Concepts"
    
    logging.info(f"📩 Attempting to send emails for lead: {user_email}")
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=15)
        server.starttls()
        server.login(sender_email, app_password)

        # --- 1. ADMIN NOTIFICATION ---
        msg_admin = MIMEMultipart()
        msg_admin["From"] = formataddr((f"{brand_name} System", sender_email))
        msg_admin["To"] = sender_email
        msg_admin["Subject"] = f"🔥 NEW LEAD: {user_email}"
        msg_admin.attach(MIMEText(f"New user joined: {user_email}", "plain"))
        server.sendmail(sender_email, sender_email, msg_admin.as_string())
        logging.info(f"✅ Admin notification sent to {sender_email}")

        time.sleep(1.5)

        # --- 2. CUSTOMER RECEIPT ---
        msg_user = MIMEMultipart("alternative")
        msg_user["From"] = formataddr((brand_name, sender_email))
        msg_user["To"] = user_email
        msg_user["Subject"] = "AI BEAUTY CONCEPTS - Your Official Receipt"
        
        html_content = f"<html><body style='background:#000;color:#fff;text-align:center;padding:40px;border:3px solid #ffd700;'><h1>AI BEAUTY CONCEPTS</h1><p>You are officially on the list.</p><h2>THE FUTURE IS PLATINUM.</h2></body></html>"
        msg_user.attach(MIMEText(html_content, "html"))
        server.sendmail(sender_email, user_email, msg_user.as_string())
        
        logging.info(f"✅ Customer receipt sent to {user_email}")
        server.quit()
        return True

    except Exception as e:
        logging.error(f"❌ CRITICAL ERROR: {str(e)}")
        return False

def send_style_to_barber(to_email, style_name, image_url=None):
    logging.info(f"🔄 Compatibility redirect for {to_email}")
    return send_waitlist_email(to_email)

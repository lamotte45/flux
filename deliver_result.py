import smtplib
from email.message import EmailMessage
import os

# CONFIGURATION
SENDER_EMAIL = "your-email@gmail.com"
SENDER_PASSWORD = "your-app-password" # Use a Google App Password, not your login
BARBER_EMAIL = "barber-inbox@gmail.com"

def send_to_barber(image_path, session_id, style_name, behavior_score):
    msg = EmailMessage()
    msg['Subject'] = f"✨ New Hair Concept: {style_name}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = BARBER_EMAIL
    msg.set_content(f"""
    🎯 NEW PATRON CHOICE
    ----------------------------
    Session ID: {session_id}
    Style Requested: {style_name}
    Sentiment Score: {behavior_score} (Attention + Joy)
    
    The AI-generated preview is attached below.
    """)

    # Attach the image
    with open(image_path, 'rb') as f:
        file_data = f.read()
        msg.add_attachment(file_data, maintype='image', subtype='png', filename='choice.png')

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        print("✅ Email sent to Stylist!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

from fastapi import FastAPI, UploadFile, File
from flask import Flask, request, jsonify
from redis import Redis
from rq import Queue
import os
import uuid
from hair_tasks import generate_style_previews

app = Flask(__name__)
redis_conn = Redis()
q = Queue(connection=redis_conn)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    email = data.get('email')
    prompt = data.get('prompt')
    image_path = data.get('image_path') # Re-using existing if provided
    
    job_id = str(uuid.uuid4())
    
    # Queue the clean, zero-artifact task
    job = q.enqueue(
        generate_style_previews, 
        args=(email, image_path, prompt),
        job_id=job_id
    )
    
    return jsonify({"status": "queued", "job_id": job_id}), 202

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

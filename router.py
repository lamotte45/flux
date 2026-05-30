from fastapi import FastAPI, UploadFile, File, Form
import requests

app = FastAPI()

@app.post("/avatar/request")
async def avatar_request(image: UploadFile = File(...), text: str = Form(...)):
    try:
        files = {
            "image": (image.filename, await image.read(), image.content_type)
        }

        data = {
            "prompt": text
        }

        response = requests.post(
            "http://127.0.0.1:8000/hair",
            files=files,
            data=data
        )

        return response.json()

    except Exception as e:
        return {"error": str(e)}

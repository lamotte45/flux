import os
import base64
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Barber AI Catalog")

CATALOG_DIR = "/home/kenny/barber_ai/catalog"

@app.get("/")
async def root():
    return {"status": "ok", "service": "catalog_api"}

@app.get("/styles")
async def get_all_styles():
    catalog = {}
    for category in os.listdir(CATALOG_DIR):
        cat_path = os.path.join(CATALOG_DIR, category)
        if os.path.isdir(cat_path):
            catalog[category] = []
            for img in sorted(os.listdir(cat_path)):
                if img.endswith('.jpg'):
                    style_name = img.replace('.jpg', '').replace(f'{category}_', '')
                    catalog[category].append({
                        "id": img.replace('.jpg', ''),
                        "name": style_name.replace('_', ' ').title(),
                        "category": category,
                        "url": f"/static/{category}/{img}"
                    })
    return JSONResponse(catalog)

@app.get("/styles/{category}")
async def get_styles_by_category(category: str):
    cat_path = os.path.join(CATALOG_DIR, category)
    if not os.path.exists(cat_path):
        return JSONResponse({"error": "Category not found"}, status_code=404)
    styles = []
    for img in sorted(os.listdir(cat_path)):
        if img.endswith('.jpg'):
            style_name = img.replace('.jpg', '').replace(f'{category}_', '')
            with open(os.path.join(cat_path, img), 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            styles.append({
                "id": img.replace('.jpg', ''),
                "name": style_name.replace('_', ' ').title(),
                "category": category,
                "image": b64
            })
    return JSONResponse(styles)

app.mount("/static", StaticFiles(directory=CATALOG_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

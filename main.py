from fastapi import FastAPI
from routes.style_route import router as style_router
from routes.concierge_route import router as concierge_router

app = FastAPI()

app.include_router(style_router)
app.include_router(concierge_router)

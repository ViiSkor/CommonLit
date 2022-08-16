from fastapi import FastAPI
from joblib import load

from models.schemas import Regressor
from routes.prediction import app_prediction
from routes.main import app_home


app = FastAPI(title="ML API", description="API for text complexity prediction", version="1.0")


@app.on_event('startup')
async def load_model():
    reg = Regressor()
    reg.load_model()


app.include_router(app_home)
app.include_router(app_prediction, prefix='')

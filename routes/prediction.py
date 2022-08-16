from fastapi import APIRouter
from models.schemas import CommonLit, CommonLitPredictionResponse, Regressor

app_prediction = APIRouter()


@app_prediction.post('/prediction',
                     tags=["Predictions"],
                     response_model=CommonLitPredictionResponse,
                     description="Get a complexity score")
async def get_prediction(common_lit: CommonLit):
    data = dict(common_lit)['data']
    model = Regressor().get_model()
    prediction = model.predict(data).tolist()
    return {'prediction': prediction}

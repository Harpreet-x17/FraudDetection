import os
import socket

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from src.Pipeline.Predict_pipeline import CustomData, PredictPipeline

app = FastAPI()

templates = Jinja2Templates(directory="templates")

## Route for a home page


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"request": request},
    )


@app.get("/predictdata")
async def predict_datapoint_get(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"request": request},
    )


@app.post("/predictdata")
async def predict_datapoint_post(
    request: Request,
    type: str = Form(...),
    amount: float = Form(...),
    oldbalanceOrg: float = Form(...),
    newbalanceOrig: float = Form(...),
    oldbalanceDest: float = Form(...),
    newbalanceDest: float = Form(...),
):
    data = CustomData(
        type=type,
        amount=amount,
        oldbalanceOrg=oldbalanceOrg,
        newbalanceOrig=newbalanceOrig,
        oldbalanceDest=oldbalanceDest,
        newbalanceDest=newbalanceDest,
    )
    pred_df = data.get_data_as_data_frame()
    print(pred_df)
    print("Before Prediction")

    try:
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return templates.TemplateResponse(
            request=request,
            name="home.html",
            context={"request": request, "results": int(results[0])},
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="home.html",
            context={"request": request, "error": str(exc)},
            status_code=500,
        )


def get_available_port(preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", preferred_port)) != 0:
            return preferred_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


if __name__ == "__main__":
    import uvicorn

    preferred_port = int(os.getenv("PORT", "8000"))
    port = get_available_port(preferred_port)

    if port != preferred_port:
        print(f"Port {preferred_port} is busy. Starting on port {port} instead.")

    uvicorn.run(app, host="0.0.0.0", port=port)

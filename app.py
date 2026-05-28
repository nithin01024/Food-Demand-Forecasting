from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pathlib

from pipeline import run_pipeline

app = FastAPI()
ROOT = pathlib.Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/output", StaticFiles(directory=str(ROOT / "output")), name="output")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/run", response_class=HTMLResponse)
async def run(request: Request):
    result = run_pipeline()
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "metrics": result["metrics"],
            "forecast_plot": "/output/forecast_results.png",
            "metrics_image": "/output/metrics_summary.png",
        },
    )

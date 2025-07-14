from fastapi import FastAPI
from .routes import health, predict, train

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

def run():
    app = FastAPI(
        title="Equipment Failure Prediction API",
        description="API for predicting industrial equipment failures",
        version="1.0.0"
    )
    
    app.include_router(health.router, prefix="/health")
    app.include_router(predict.router, prefix="/predict")
    app.include_router(train.router, prefix="/train")
    
    return app

app = run()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(run(), host="0.0.0.0", port=8000, log_level="debug")
import os
import uuid
import time
import shutil
import logging
from fastapi import Request
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from src.inference import predict_image   # IMPORTANT: make sure this path is correct

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Cats vs Dogs Classifier API")

# ----------------------------
# Monitoring Variables
# ----------------------------
request_count = 0
total_latency = 0.0

# ----------------------------
# Prometheus Metrics
# ----------------------------
PROM_REQUEST_COUNT = Counter(
    "api_request_count",
    "Total number of API requests",
    ["method", "endpoint"]
)

PROM_REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds"
)


# ----------------------------
# Monitoring Middleware
# ----------------------------
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    global request_count, total_latency

    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    # Update custom counters
    request_count += 1
    total_latency += latency

    # Update Prometheus metrics
    PROM_REQUEST_COUNT.labels(request.method, request.url.path).inc()
    PROM_REQUEST_LATENCY.observe(latency)

    logger.info(
        f"Request #{request_count} | "
        f"Path: {request.url.path} | "
        f"Latency: {latency:.4f} sec"
    )

    return response


# ----------------------------
# Health Endpoint
# ----------------------------
@app.get("/health")
def health():
    avg_latency = total_latency / request_count if request_count > 0 else 0

    return {
        "status": "ok",
        "total_requests": request_count,
        "avg_latency_sec": round(avg_latency, 4)
    }


# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    os.makedirs("uploads", exist_ok=True)
    temp_filename = f"uploads/{uuid.uuid4()}.jpg"

    try:
        # Save uploaded image temporarily
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Perform prediction
        label, probabilities = predict_image(temp_filename)

        label_name = "cat" if label == 0 else "dog"

        return JSONResponse(
            content={
                "prediction": label_name,
                "probabilities": probabilities
            }
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    finally:
        # Clean up temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# ----------------------------
# Prometheus Metrics Endpoint
# ----------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

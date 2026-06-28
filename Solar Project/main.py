"""
main.py — Solar Panel AI Fault Detection & Maintenance Recommendation System
FastAPI Backend

Tech Stack:
- FastAPI          : REST API framework
- InfluxDB         : Time-series sensor storage
- AWS S3 (boto3)   : Thermal image storage
- Kafka            : Streaming ingestion layer
- TensorFlow/Keras : LSTM anomaly model
- TorchVision      : CNN hotspot detection model
- XGBoost          : Fault fusion classifier
- ChromaDB         : Vector store (RAG)
- OpenAI           : Embeddings + GPT-4 LLM
- Pydantic         : Request/response validation
"""

import os
import io
import json
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import boto3
import chromadb
import xgboost as xgb
import tensorflow as tf
import torch
import torchvision.transforms as transforms
from PIL import Image
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from kafka import KafkaProducer
from openai import AzureOpenAI
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("solar_api")

# ─────────────────────────────────────────────
# Config (replace with env vars / Azure Key Vault in prod)
# ─────────────────────────────────────────────
INFLUX_URL        = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN      = os.getenv("INFLUX_TOKEN", "my-token")
INFLUX_ORG        = os.getenv("INFLUX_ORG", "solar-org")
INFLUX_BUCKET     = os.getenv("INFLUX_BUCKET", "solar_metrics")

AWS_BUCKET_NAME   = os.getenv("AWS_BUCKET_NAME", "solar-thermal-images")
AWS_REGION        = os.getenv("AWS_REGION", "us-east-1")

KAFKA_BROKER      = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC       = os.getenv("KAFKA_TOPIC", "solar.sensor.events")

CHROMA_PERSIST    = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION = "solar_manuals"

AZURE_OPENAI_KEY      = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_VERSION  = os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")
EMBED_DEPLOYMENT      = os.getenv("EMBED_DEPLOYMENT", "text-embedding-3-large")
LLM_DEPLOYMENT        = os.getenv("LLM_DEPLOYMENT", "gpt-4o")

LSTM_MODEL_PATH   = os.getenv("LSTM_MODEL_PATH", "models/lstm_anomaly.h5")
CNN_MODEL_PATH    = os.getenv("CNN_MODEL_PATH", "models/cnn_hotspot.pt")
XGB_MODEL_PATH    = os.getenv("XGB_MODEL_PATH", "models/xgboost_classifier.json")

FAULT_LABELS = [
    "hotspot_cell_degradation",
    "partial_shading",
    "inverter_fault",
    "soiling",
    "bypass_diode_failure",
    "normal",
]

# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class SensorReading(BaseModel):
    panel_id: str = Field(..., example="PANEL_A32")
    voltage: float = Field(..., ge=0, le=100, example=36.4)
    current: float = Field(..., ge=0, le=20, example=8.2)
    temperature: float = Field(..., example=45.3)
    irradiance: float = Field(..., ge=0, example=850.0)
    timestamp: Optional[datetime] = None

class FaultDetectionResponse(BaseModel):
    panel_id: str
    anomaly_score: float
    fault_category: str
    confidence: float
    severity: str          # LOW / MEDIUM / HIGH
    timestamp: datetime

class MaintenanceRecommendation(BaseModel):
    panel_id: str
    fault_category: str
    root_cause: str
    repair_steps: list[str]
    tools_required: list[str]
    safety_warnings: list[str]
    estimated_downtime_hours: float
    priority: str
    source_documents: list[str]

class AlertRequest(BaseModel):
    panel_id: str
    fault_category: str
    severity: str
    technician_email: str

class PipelineResponse(BaseModel):
    panel_id: str
    fault: FaultDetectionResponse
    recommendation: MaintenanceRecommendation

# ─────────────────────────────────────────────
# Service Singletons (initialised at startup)
# ─────────────────────────────────────────────

class Services:
    influx_write_api  = None
    influx_query_api  = None
    s3_client         = None
    kafka_producer    = None
    chroma_collection = None
    openai_client     = None
    lstm_model        = None
    cnn_model         = None
    xgb_model         = None

svc = Services()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise all service connections. Shutdown: flush/close."""
    logger.info("🚀 Starting up Solar Fault Detection API...")

    # ── InfluxDB ──────────────────────────────
    influx = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    svc.influx_write_api = influx.write_api(write_options=SYNCHRONOUS)
    svc.influx_query_api = influx.query_api()
    logger.info("✅ InfluxDB connected")

    # ── AWS S3 ────────────────────────────────
    svc.s3_client = boto3.client("s3", region_name=AWS_REGION)
    logger.info("✅ AWS S3 client initialised")

    # ── Kafka Producer ────────────────────────
    svc.kafka_producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",               # wait for all replicas to ack
        retries=3,
    )
    logger.info("✅ Kafka producer connected")

    # ── ChromaDB (Vector Store) ───────────────
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST)
    svc.chroma_collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("✅ ChromaDB collection ready")

    # ── Azure OpenAI ──────────────────────────
    svc.openai_client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_VERSION,
    )
    logger.info("✅ Azure OpenAI client ready")

    # ── ML Models ─────────────────────────────
    svc.lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    logger.info("✅ LSTM model loaded")

    svc.cnn_model = torch.load(CNN_MODEL_PATH, map_location="cpu")
    svc.cnn_model.eval()
    logger.info("✅ CNN model loaded")

    svc.xgb_model = xgb.Booster()
    svc.xgb_model.load_model(XGB_MODEL_PATH)
    logger.info("✅ XGBoost model loaded")

    logger.info("🟢 All services ready — API accepting requests")
    yield

    # ── Shutdown ──────────────────────────────
    logger.info("🛑 Shutting down — flushing Kafka producer...")
    svc.kafka_producer.flush()
    svc.kafka_producer.close()
    logger.info("✅ Shutdown complete")

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Solar Panel Fault Detection API",
    description="AI-powered fault detection and maintenance recommendation for solar farms.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ─────────────────────────────────────────────
# Helper: LSTM Anomaly Detection
# ─────────────────────────────────────────────

def run_lstm_inference(reading: SensorReading) -> tuple[float, bool]:
    """
    Feeds sensor readings into LSTM model.
    Returns (anomaly_score, is_anomaly).
    In production, 'reading' would be a sliding window of the last N timesteps.
    """
    features = np.array([[
        reading.voltage,
        reading.current,
        reading.temperature,
        reading.irradiance,
    ]], dtype=np.float32)

    # LSTM expects shape: (batch, timesteps, features) — here we simulate 1 timestep
    features = features.reshape(1, 1, 4)
    prediction = svc.lstm_model.predict(features, verbose=0)
    anomaly_score = float(prediction[0][0])
    is_anomaly = anomaly_score > 0.5
    return anomaly_score, is_anomaly

# ─────────────────────────────────────────────
# Helper: CNN Hotspot Detection
# ─────────────────────────────────────────────

def run_cnn_inference(image_bytes: bytes) -> tuple[float, bool]:
    """
    Runs thermal image through CNN model.
    Returns (hotspot_confidence, hotspot_detected).
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # shape: (1, 3, 224, 224)

    with torch.no_grad():
        output = svc.cnn_model(tensor)
        hotspot_conf = float(torch.sigmoid(output)[0][1])  # binary: hotspot vs normal

    return hotspot_conf, hotspot_conf > 0.6

# ─────────────────────────────────────────────
# Helper: XGBoost Fault Classifier
# ─────────────────────────────────────────────

def run_xgboost_classification(
    anomaly_score: float,
    hotspot_conf: float,
    voltage: float,
    current: float,
    temperature: float,
    irradiance: float,
) -> tuple[str, float]:
    """
    Fuses LSTM + CNN outputs with raw sensor features.
    Returns (fault_label, confidence).
    """
    features = np.array([[
        anomaly_score,
        hotspot_conf,
        voltage,
        current,
        temperature,
        irradiance,
    ]], dtype=np.float32)

    dmatrix = xgb.DMatrix(features)
    probs = svc.xgb_model.predict(dmatrix)[0]     # shape: (num_classes,)
    pred_idx = int(np.argmax(probs))
    return FAULT_LABELS[pred_idx], float(probs[pred_idx])

def severity_from_confidence(fault: str, confidence: float) -> str:
    if fault == "normal":
        return "LOW"
    if confidence >= 0.85:
        return "HIGH"
    if confidence >= 0.6:
        return "MEDIUM"
    return "LOW"

# ─────────────────────────────────────────────
# Helper: RAG — Semantic Retrieval
# ─────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Converts text to vector using Azure OpenAI embeddings."""
    response = svc.openai_client.embeddings.create(
        input=text,
        model=EMBED_DEPLOYMENT,
    )
    return response.data[0].embedding

def retrieve_relevant_docs(fault_category: str, panel_context: str, top_k: int = 4) -> list[dict]:
    """
    Builds a semantic query from fault + context,
    retrieves top-k relevant manual chunks from ChromaDB.
    """
    query_text = f"Solar panel fault: {fault_category}. Context: {panel_context}"
    query_vector = get_embedding(query_text)

    results = svc.chroma_collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where={"fault_category": {"$in": [fault_category, "general"]}},  # metadata filter
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({
            "content": doc,
            "source": meta.get("source", "Unknown"),
            "page": meta.get("page", "N/A"),
            "similarity": round(1 - dist, 4),   # cosine distance → similarity
        })
    return docs

# ─────────────────────────────────────────────
# Helper: LLM — Maintenance Recommendation
# ─────────────────────────────────────────────

def generate_recommendation(
    panel_id: str,
    fault_category: str,
    sensor_reading: SensorReading,
    retrieved_docs: list[dict],
) -> MaintenanceRecommendation:
    """
    Injects retrieved manual chunks + fault context into GPT-4 prompt.
    Returns structured JSON parsed into MaintenanceRecommendation.
    """
    context_block = "\n\n".join(
        f"[Source: {d['source']}, Page {d['page']}]\n{d['content']}"
        for d in retrieved_docs
    )

    prompt = f"""
You are an expert solar panel maintenance engineer.

## Detected Fault
Panel ID     : {panel_id}
Fault Type   : {fault_category}
Voltage      : {sensor_reading.voltage} V
Current      : {sensor_reading.current} A
Temperature  : {sensor_reading.temperature} °C
Irradiance   : {sensor_reading.irradiance} W/m²
Timestamp    : {sensor_reading.timestamp}

## Relevant Manual Sections Retrieved
{context_block}

## Instructions
Based ONLY on the retrieved manual sections above, generate a structured maintenance plan.
Respond ONLY with valid JSON matching this exact schema — no extra text, no markdown:

{{
  "root_cause": "string",
  "repair_steps": ["step1", "step2", "step3"],
  "tools_required": ["tool1", "tool2"],
  "safety_warnings": ["warning1", "warning2"],
  "estimated_downtime_hours": 2.0,
  "priority": "HIGH | MEDIUM | LOW"
}}
"""

    response = svc.openai_client.chat.completions.create(
        model=LLM_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a solar panel maintenance expert. Always respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,        # low temp → consistent, factual output
        max_tokens=1000,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)

    return MaintenanceRecommendation(
        panel_id=panel_id,
        fault_category=fault_category,
        root_cause=data["root_cause"],
        repair_steps=data["repair_steps"],
        tools_required=data["tools_required"],
        safety_warnings=data["safety_warnings"],
        estimated_downtime_hours=data["estimated_downtime_hours"],
        priority=data["priority"],
        source_documents=[d["source"] for d in retrieved_docs],
    )

# ─────────────────────────────────────────────
# Helper: InfluxDB Write
# ─────────────────────────────────────────────

def write_sensor_to_influx(reading: SensorReading):
    point = (
        Point("solar_metrics")
        .tag("panel_id", reading.panel_id)
        .field("voltage", reading.voltage)
        .field("current", reading.current)
        .field("temperature", reading.temperature)
        .field("irradiance", reading.irradiance)
        .time(reading.timestamp or datetime.utcnow())
    )
    svc.influx_write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

# ─────────────────────────────────────────────
# Helper: S3 Image Upload
# ─────────────────────────────────────────────

def upload_image_to_s3(panel_id: str, image_bytes: bytes, filename: str) -> str:
    key = f"thermal/{panel_id}/{datetime.utcnow().date()}/{filename}"
    svc.s3_client.put_object(
        Bucket=AWS_BUCKET_NAME,
        Key=key,
        Body=image_bytes,
        ContentType="image/jpeg",
    )
    return f"s3://{AWS_BUCKET_NAME}/{key}"

# ─────────────────────────────────────────────
# Helper: Kafka Publish
# ─────────────────────────────────────────────

def publish_to_kafka(topic: str, payload: dict):
    svc.kafka_producer.send(topic, value=payload)
    logger.info(f"📤 Published to Kafka topic '{topic}': panel_id={payload.get('panel_id')}")

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe — returns 200 if the service is running."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/ingest/sensor", tags=["Ingestion"], response_model=dict)
async def ingest_sensor_data(
    reading: SensorReading,
    background_tasks: BackgroundTasks,
    token: str = Depends(oauth2_scheme),
):
    """
    Accepts real-time sensor readings from IoT panels.
    Persists to InfluxDB and publishes to Kafka for downstream consumers.
    """
    reading.timestamp = reading.timestamp or datetime.utcnow()

    # Write to time-series DB (synchronous — fast write)
    write_sensor_to_influx(reading)

    # Publish to Kafka stream (non-blocking)
    background_tasks.add_task(
        publish_to_kafka,
        KAFKA_TOPIC,
        reading.model_dump(mode="json"),
    )

    logger.info(f"📥 Sensor ingested: panel={reading.panel_id}")
    return {"status": "ingested", "panel_id": reading.panel_id}


@app.post("/predict/fault", tags=["ML Inference"], response_model=FaultDetectionResponse)
async def predict_fault(
    reading: SensorReading,
    thermal_image: UploadFile = File(...),
    token: str = Depends(oauth2_scheme),
):
    """
    Runs the full ML pipeline:
    1. LSTM → anomaly score from sensor time-series
    2. CNN  → hotspot confidence from thermal image
    3. XGBoost → fused fault classification
    """
    # 1. LSTM inference on sensor reading
    anomaly_score, _ = run_lstm_inference(reading)

    # 2. S3 upload + CNN inference on thermal image
    image_bytes = await thermal_image.read()
    s3_key = upload_image_to_s3(reading.panel_id, image_bytes, thermal_image.filename)
    hotspot_conf, _ = run_cnn_inference(image_bytes)
    logger.info(f"🖼️  Thermal image stored at {s3_key}")

    # 3. XGBoost fusion classification
    fault_label, confidence = run_xgboost_classification(
        anomaly_score, hotspot_conf,
        reading.voltage, reading.current,
        reading.temperature, reading.irradiance,
    )

    result = FaultDetectionResponse(
        panel_id=reading.panel_id,
        anomaly_score=round(anomaly_score, 4),
        fault_category=fault_label,
        confidence=round(confidence, 4),
        severity=severity_from_confidence(fault_label, confidence),
        timestamp=datetime.utcnow(),
    )

    logger.info(f"🔍 Fault detected: {fault_label} ({confidence:.0%}) on {reading.panel_id}")
    return result


@app.post("/recommend", tags=["GenAI / RAG"], response_model=MaintenanceRecommendation)
async def get_maintenance_recommendation(
    panel_id: str,
    fault_category: str,
    reading: SensorReading,
    token: str = Depends(oauth2_scheme),
):
    """
    RAG + LLM pipeline:
    1. Embed fault context → semantic vector
    2. Query ChromaDB for relevant manual sections
    3. GPT-4 generates structured repair guidance (JSON)
    """
    panel_context = (
        f"Voltage={reading.voltage}V, Current={reading.current}A, "
        f"Temp={reading.temperature}°C, Irradiance={reading.irradiance}W/m²"
    )

    # RAG retrieval
    docs = retrieve_relevant_docs(fault_category, panel_context)
    if not docs:
        raise HTTPException(status_code=404, detail="No relevant manual sections found in knowledge base.")

    # LLM generation
    recommendation = generate_recommendation(panel_id, fault_category, reading, docs)
    logger.info(f"🤖 Recommendation generated for {panel_id}: priority={recommendation.priority}")
    return recommendation


@app.post("/pipeline/full", tags=["Orchestration"], response_model=PipelineResponse)
async def full_pipeline(
    reading: SensorReading,
    thermal_image: UploadFile = File(...),
    token: str = Depends(oauth2_scheme),
):
    """
    End-to-end orchestration endpoint:
    Sensor + Image → Fault Detection → RAG Recommendation in one call.
    Ideal for dashboard polling or webhook triggers.
    """
    # Step 1: Ingest + store
    reading.timestamp = reading.timestamp or datetime.utcnow()
    write_sensor_to_influx(reading)

    # Step 2: ML fault detection
    anomaly_score, _ = run_lstm_inference(reading)
    image_bytes = await thermal_image.read()
    upload_image_to_s3(reading.panel_id, image_bytes, thermal_image.filename)
    hotspot_conf, _ = run_cnn_inference(image_bytes)

    fault_label, confidence = run_xgboost_classification(
        anomaly_score, hotspot_conf,
        reading.voltage, reading.current,
        reading.temperature, reading.irradiance,
    )

    fault = FaultDetectionResponse(
        panel_id=reading.panel_id,
        anomaly_score=round(anomaly_score, 4),
        fault_category=fault_label,
        confidence=round(confidence, 4),
        severity=severity_from_confidence(fault_label, confidence),
        timestamp=datetime.utcnow(),
    )

    # Step 3: RAG + LLM recommendation
    panel_context = f"Voltage={reading.voltage}V, Current={reading.current}A, Temp={reading.temperature}°C"
    docs = retrieve_relevant_docs(fault_label, panel_context)
    recommendation = generate_recommendation(reading.panel_id, fault_label, reading, docs)

    # Step 4: Publish fault event to Kafka
    publish_to_kafka(
        KAFKA_TOPIC,
        {
            "panel_id": reading.panel_id,
            "fault_category": fault_label,
            "confidence": confidence,
            "severity": fault.severity,
            "timestamp": reading.timestamp.isoformat(),
        },
    )

    return PipelineResponse(panel_id=reading.panel_id, fault=fault, recommendation=recommendation)


@app.get("/dashboard/panels", tags=["Dashboard"])
async def get_panel_overview(token: str = Depends(oauth2_scheme)):
    """
    Queries InfluxDB for the latest reading of each panel.
    Returns a list of panel status summaries for the dashboard.
    """
    flux_query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -1h)
      |> filter(fn: (r) => r._measurement == "solar_metrics")
      |> last()
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    tables = svc.influx_query_api.query(flux_query, org=INFLUX_ORG)
    panels = []
    for table in tables:
        for record in table.records:
            panels.append({
                "panel_id": record.values.get("panel_id"),
                "voltage": record.values.get("voltage"),
                "current": record.values.get("current"),
                "temperature": record.values.get("temperature"),
                "irradiance": record.values.get("irradiance"),
                "last_seen": record.get_time().isoformat(),
            })
    return {"panels": panels, "count": len(panels)}


@app.post("/alerts/dispatch", tags=["Alerting"])
async def dispatch_alert(
    alert: AlertRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(oauth2_scheme),
):
    """
    Dispatches a high-priority fault alert.
    In production: triggers Azure Notification Hub / SNS / email service.
    Here we publish to a dedicated Kafka alerts topic.
    """
    if alert.severity != "HIGH":
        return {"status": "skipped", "reason": "Only HIGH severity alerts are dispatched"}

    alert_payload = {
        "panel_id": alert.panel_id,
        "fault_category": alert.fault_category,
        "severity": alert.severity,
        "technician_email": alert.technician_email,
        "dispatched_at": datetime.utcnow().isoformat(),
    }
    background_tasks.add_task(publish_to_kafka, "solar.alerts.high_priority", alert_payload)
    logger.warning(f"🚨 HIGH priority alert dispatched for panel {alert.panel_id}")
    return {"status": "dispatched", "panel_id": alert.panel_id}

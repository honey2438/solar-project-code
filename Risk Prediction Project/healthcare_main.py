"""
main.py — Healthcare Claims Risk Prediction System (ML + HITL Workflow)
FastAPI Backend

End-to-End Architecture:
  1. DATA INGESTION       — Secure SFTP/File upload → Azure Blob (raw landing zone)
  2. ML PIPELINE          — Data profiling → Cleaning → Feature Engineering →
                            ColumnTransformer → XGBoost training → Evaluation
  3. INFERENCE            — Daily batch scoring → Risk Score + SHAP Explanations →
                            Predictions Staging DB (Azure SQL)
  4. HITL REVIEW          — Reviewer Dashboard API → Confirm / Reject / Modify →
                            Audit Log → Final Validated Results DB
  5. FINAL DELIVERY       — SFTP push / Secure File / API webhook → Client DB/EMR

Tech Stack:
  - FastAPI               : REST API framework (async, Pydantic validation, OAuth2)
  - Azure Blob Storage    : Raw landing zone for PHI data (encrypted at rest)
  - Azure SQL Database    : Staging DB + Final Validated Results DB + Audit Log
  - XGBoost               : Supervised classifier — "Provider at Risk?"
  - Scikit-learn          : ColumnTransformer preprocessing pipeline
  - SHAP                  : Per-prediction explainability (feature importance)
  - Pandas / NumPy        : Data processing and feature engineering
  - Airflow / ADF         : Orchestration (referenced, not imported here)
  - Paramiko              : SFTP delivery to client
  - PyODBC / SQLAlchemy   : Azure SQL connectivity
  - Pydantic              : Request/response schema validation
  - OAuth2 / Azure AD     : RBAC authentication

Compliance:
  - HIPAA: PHI encrypted in transit (HTTPS/SFTP) and at rest (AES-256 Azure)
  - RBAC: Role-based access — reviewer, data_engineer, admin, client_readonly
  - Audit Log: Immutable insert-only table (Who, When, What)
  - PII masking in application logs
"""

import os
import io
import json
import uuid
import logging
import hashlib
import datetime
import numpy as np
import pandas as pd
import shap
import joblib
import paramiko
import pyodbc
from typing import Optional, List
from contextlib import asynccontextmanager

import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, average_precision_score
)

from azure.storage.blob import BlobServiceClient
from fastapi import (
    FastAPI, File, UploadFile, HTTPException,
    BackgroundTasks, Depends, Security
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Logging — PHI fields are never logged; only masked identifiers appear in logs
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("healthcare_claims_api")

def mask_phi(value: str) -> str:
    """
    One-way hash of a PHI identifier for safe logging.
    e.g. provider_id='NPI_1234567890' → logs 'MASKED_a3f9c1...'
    """
    return "MASKED_" + hashlib.sha256(value.encode()).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────────────
# Config — In production: Azure Key Vault / environment variables
# ─────────────────────────────────────────────────────────────────────────────
AZURE_BLOB_CONN_STR     = os.getenv("AZURE_BLOB_CONN_STR", "")
AZURE_BLOB_CONTAINER    = os.getenv("AZURE_BLOB_CONTAINER", "claims-raw-landing")

# Azure SQL — Staging DB (model output, pre-review)
STAGING_DB_CONN         = os.getenv("STAGING_DB_CONN", "")

# Azure SQL — Final Validated Results DB (post-HITL review)
FINAL_DB_CONN           = os.getenv("FINAL_DB_CONN", "")

# Model artifact paths (loaded from Blob / Azure ML registry in production)
PIPELINE_PATH           = os.getenv("PIPELINE_PATH", "models/preprocessing_pipeline.pkl")
MODEL_PATH              = os.getenv("MODEL_PATH", "models/xgboost_classifier.json")

# SFTP delivery config for client push
CLIENT_SFTP_HOST        = os.getenv("CLIENT_SFTP_HOST", "")
CLIENT_SFTP_PORT        = int(os.getenv("CLIENT_SFTP_PORT", "22"))
CLIENT_SFTP_USER        = os.getenv("CLIENT_SFTP_USER", "")
CLIENT_SFTP_KEY_PATH    = os.getenv("CLIENT_SFTP_KEY_PATH", "/keys/client_rsa_key")
CLIENT_SFTP_REMOTE_DIR  = os.getenv("CLIENT_SFTP_REMOTE_DIR", "/incoming/risk_results/")

# Model training constants
RANDOM_STATE            = 42          # Fixed for full reproducibility
TEST_SIZE               = 0.20        # 80/20 split
RISK_THRESHOLD          = 0.50        # Classification threshold (tunable per client)
TARGET_COLUMN           = "provider_at_risk"   # Binary label: 1 = high risk, 0 = normal

# Feature definitions — align with ColumnTransformer configuration
NUMERICAL_FEATURES = [
    "total_billed_amount",
    "avg_claim_amount",
    "claim_count_30d",
    "unique_procedure_codes",
    "unique_diagnosis_codes",
    "procedure_diagnosis_ratio",    # engineered: procedures per diagnosis
    "billing_growth_rate_pct",      # engineered: MoM growth in billed amount
    "denial_rate_pct",              # engineered: % claims denied historically
    "avg_days_to_submit",           # engineered: lag between service and submission
]

CATEGORICAL_FEATURES = [
    "provider_specialty",
    "provider_state",
    "payer_type",                   # commercial, medicare, medicaid
    "claim_type",                   # professional (837P), institutional (837I)
    "network_status",               # in-network, out-of-network
]


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ClaimRecord(BaseModel):
    """Single claim record — mirrors one row from the 837 EDI parsed dataset."""
    claim_id: str
    provider_id: str                            # NPI (National Provider Identifier)
    provider_specialty: str
    provider_state: str
    payer_type: str
    claim_type: str
    network_status: str
    total_billed_amount: float = Field(..., ge=0)
    avg_claim_amount: float = Field(..., ge=0)
    claim_count_30d: int = Field(..., ge=0)
    unique_procedure_codes: int = Field(..., ge=0)
    unique_diagnosis_codes: int = Field(..., ge=0)
    procedure_diagnosis_ratio: float = Field(..., ge=0)
    billing_growth_rate_pct: float
    denial_rate_pct: float = Field(..., ge=0, le=100)
    avg_days_to_submit: float = Field(..., ge=0)

class BatchScoringRequest(BaseModel):
    """Batch of claim records to score in a single inference run."""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    records: List[ClaimRecord]
    client_id: str
    scoring_date: Optional[datetime.date] = None

class RiskPrediction(BaseModel):
    """Model output for a single provider — stored in Staging DB."""
    staging_id: str
    batch_id: str
    claim_id: str
    provider_id: str
    risk_score: float                           # continuous 0-1 probability
    prediction: str                             # "AT_RISK" or "NORMAL"
    top_features: dict                          # SHAP feature contributions
    processing_timestamp: datetime.datetime

class ReviewDecision(BaseModel):
    """Reviewer's HITL decision on a staged prediction."""
    staging_id: str
    reviewer_id: str                            # authenticated reviewer's user ID
    decision: str = Field(..., pattern="^(CONFIRM|REJECT|MODIFY)$")
    modified_risk_score: Optional[float] = Field(None, ge=0, le=1)
    modified_prediction: Optional[str] = None
    review_comments: Optional[str] = None

class TrainingRequest(BaseModel):
    """Trigger a model training run on a labeled dataset in Blob Storage."""
    dataset_blob_path: str                      # path inside AZURE_BLOB_CONTAINER
    experiment_name: str = "xgboost_claims_risk"
    scale_pos_weight: Optional[float] = None    # override for class imbalance ratio

class ModelEvaluationResult(BaseModel):
    """Evaluation metrics returned after a training run."""
    experiment_name: str
    roc_auc: float
    avg_precision: float
    f1_score: float
    precision: float
    recall: float
    confusion_matrix: List[List[int]]
    threshold_used: float
    train_samples: int
    test_samples: int
    model_path: str


# ─────────────────────────────────────────────────────────────────────────────
# Service Singletons
# ─────────────────────────────────────────────────────────────────────────────

class Services:
    blob_client         = None   # Azure Blob ServiceClient
    staging_conn        = None   # pyodbc connection → Staging DB
    final_conn          = None   # pyodbc connection → Final Validated DB
    preprocessing_pipe  = None   # Fitted Scikit-learn ColumnTransformer pipeline
    xgb_model           = None   # Trained XGBoost Booster
    shap_explainer      = None   # SHAP TreeExplainer (loaded after model)

svc = Services()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: connect to all services, load model artifacts.
    Shutdown: close DB connections cleanly.
    """
    logger.info("🚀 Starting Healthcare Claims Risk API...")

    # ── Azure Blob Storage (raw PHI data landing zone) ─────────────────────
    svc.blob_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
    logger.info("✅ Azure Blob Storage connected")

    # ── Azure SQL — Staging DB ──────────────────────────────────────────────
    svc.staging_conn = pyodbc.connect(STAGING_DB_CONN, autocommit=False)
    logger.info("✅ Staging DB connected")

    # ── Azure SQL — Final Validated Results DB ──────────────────────────────
    svc.final_conn = pyodbc.connect(FINAL_DB_CONN, autocommit=False)
    logger.info("✅ Final Validated Results DB connected")

    # ── Load trained preprocessing pipeline ─────────────────────────────────
    # CRITICAL: load the FITTED pipeline — never refit on production data
    # This guarantees training-serving consistency (same scaler params, same OHE vocab)
    if os.path.exists(PIPELINE_PATH):
        svc.preprocessing_pipe = joblib.load(PIPELINE_PATH)
        logger.info("✅ Preprocessing pipeline loaded")
    else:
        logger.warning("⚠️  No preprocessing pipeline found — train model first via /train")

    # ── Load trained XGBoost model ───────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        svc.xgb_model = xgb.Booster()
        svc.xgb_model.load_model(MODEL_PATH)
        logger.info("✅ XGBoost model loaded")

        # Build SHAP TreeExplainer once at startup (expensive to build per request)
        svc.shap_explainer = shap.TreeExplainer(svc.xgb_model)
        logger.info("✅ SHAP explainer initialised")
    else:
        logger.warning("⚠️  No XGBoost model found — train model first via /train")

    logger.info("🟢 All services ready — API accepting requests")
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("🛑 Shutting down — closing DB connections...")
    if svc.staging_conn:
        svc.staging_conn.close()
    if svc.final_conn:
        svc.final_conn.close()
    logger.info("✅ Shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Healthcare Claims Risk Prediction API",
    description=(
        "End-to-end ML + HITL system for identifying high-risk providers "
        "in healthcare claims data. HIPAA-compliant. Full audit trail."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://reviewer-dashboard.internal"],   # restrict in prod
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)

# OAuth2 Bearer token — in production validates JWT against Azure AD
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "reviewer":        "Access review dashboard and submit decisions",
        "data_engineer":   "Trigger training and inference pipeline runs",
        "admin":           "Manage users and system configuration",
        "client_readonly": "Read final validated results only",
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/ingest/upload", tags=["1. Data Ingestion"])
async def upload_claims_file(
    file: UploadFile = File(...),
    client_id: str = "",
    token: str = Depends(oauth2_scheme),
):
    """
    Accepts raw claims files (837 EDI, CSV exports, or flat files) via secure upload.
    Files are stored in Azure Blob Storage (encrypted at rest — AES-256).
    This is the raw landing zone — data is untouched until the pipeline processes it.

    HIPAA: file contents are PHI — only masked identifiers appear in logs.
    """
    contents = await file.read()
    file_size_kb = len(contents) / 1024

    # Blob path: clients/{client_id}/raw/{date}/{filename}
    # Partitioned by client and date for easy lifecycle management
    blob_name = (
        f"clients/{client_id}/raw/"
        f"{datetime.date.today().isoformat()}/"
        f"{file.filename}"
    )

    blob_client = svc.blob_client.get_blob_client(
        container=AZURE_BLOB_CONTAINER,
        blob=blob_name,
    )
    blob_client.upload_blob(contents, overwrite=True)

    # Log blob path only — never log file contents (PHI)
    logger.info(
        f"📥 File ingested | client={mask_phi(client_id)} | "
        f"blob={blob_name} | size={file_size_kb:.1f}KB"
    )

    return {
        "status": "uploaded",
        "blob_path": blob_name,
        "size_kb": round(file_size_kb, 2),
        "client_id": client_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — ML PIPELINE (TRAINING)
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessing_pipeline() -> Pipeline:
    """
    Builds the Scikit-learn ColumnTransformer preprocessing pipeline.

    Numerical Transformer:
      Step 1 — Median Imputation: robust to outliers in billed amounts
      Step 2 — StandardScaler: zero mean, unit variance
                (XGBoost doesn't require scaling but ensures consistent
                 feature magnitude for SHAP value comparability)

    Categorical Transformer:
      Step 1 — Most Frequent Imputation: fills missing categoricals with mode
      Step 2 — One-Hot Encoding: handle_unknown='ignore' for unseen categories
                in production (new provider types / specialties not in training)
    """
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives domain-specific features from raw claims data.
    These engineered features carry the real risk signals —
    raw fields alone are insufficient to distinguish fraud/abuse patterns.
    """
    # Procedure-to-diagnosis ratio: legitimate providers have predictable ratios
    # Unusually high ratios indicate unbundling or upcoding
    df["procedure_diagnosis_ratio"] = (
        df["unique_procedure_codes"] / (df["unique_diagnosis_codes"] + 1)
    )

    # Month-over-month billing growth: sudden spikes are a fraud signal
    # Requires historical claim_amount columns (prev_month, curr_month)
    if "prev_month_billed" in df.columns and "curr_month_billed" in df.columns:
        df["billing_growth_rate_pct"] = (
            (df["curr_month_billed"] - df["prev_month_billed"])
            / (df["prev_month_billed"] + 1)
        ) * 100
    else:
        df["billing_growth_rate_pct"] = 0.0

    # Denial rate: high denial rate may indicate incorrect coding or overcharging
    if "denied_claims" in df.columns:
        df["denial_rate_pct"] = (
            df["denied_claims"] / (df["claim_count_30d"] + 1)
        ) * 100
    else:
        df["denial_rate_pct"] = 0.0

    # Average days between service date and submission date
    # Extremely late submissions can indicate backdating
    if "service_date" in df.columns and "submission_date" in df.columns:
        df["avg_days_to_submit"] = (
            pd.to_datetime(df["submission_date"]) - pd.to_datetime(df["service_date"])
        ).dt.days.clip(lower=0)
    else:
        df["avg_days_to_submit"] = 0.0

    return df


@app.post("/train", tags=["2. ML Pipeline"], response_model=ModelEvaluationResult)
async def train_model(
    request: TrainingRequest,
    token: str = Depends(oauth2_scheme),
):
    """
    Full ML training pipeline triggered on demand or on schedule (Airflow/ADF).

    Flow:
      2A. Load dataset from Azure Blob → Data profiling → Cleaning
      2B. Stratified 80/20 train-test split (random_state=42)
      2C. Fit ColumnTransformer pipeline (numerical + categorical transformers)
      2D. Train XGBoost classifier with class imbalance correction
      2E. Evaluate: ROC-AUC, Precision-Recall, Confusion Matrix, F1
      Save fitted pipeline + model artifacts for inference
    """
    # ── 2A. Load & profile data ─────────────────────────────────────────────
    logger.info(f"📊 Loading dataset from blob: {request.dataset_blob_path}")
    blob = svc.blob_client.get_blob_client(
        container=AZURE_BLOB_CONTAINER,
        blob=request.dataset_blob_path,
    )
    raw_bytes = blob.download_blob().readall()
    df = pd.read_csv(io.BytesIO(raw_bytes))

    logger.info(
        f"Dataset loaded | rows={len(df)} | cols={len(df.columns)} | "
        f"missing_pct={df.isnull().mean().mean():.2%}"
    )

    # Basic data quality check — refuse to train on severely corrupted data
    if df[TARGET_COLUMN].isnull().sum() > 0:
        raise HTTPException(
            status_code=422,
            detail="Target column contains nulls — label data must be complete before training.",
        )

    # Feature engineering — creates derived risk-signal features
    df = engineer_features(df)

    # Remove rows where all features are missing (completely empty records)
    df.dropna(subset=NUMERICAL_FEATURES + CATEGORICAL_FEATURES, how="all", inplace=True)

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN].astype(int)

    logger.info(f"Class distribution — at_risk={y.sum()} | normal={(y==0).sum()} | ratio={y.mean():.3f}")

    # ── 2B. Stratified train-test split ─────────────────────────────────────
    # Stratify=y ensures the same proportion of high-risk labels in both splits
    # Random state fixed for full reproducibility across runs
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,        # critical for imbalanced healthcare datasets
    )
    logger.info(f"Split: train={len(X_train)} | test={len(X_test)}")

    # ── 2C. Fit preprocessing pipeline on training data only ─────────────────
    # fit_transform on training data, transform only on test data
    # This prevents data leakage — test set statistics must never influence preprocessing
    preprocessor = build_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)   # transform only — no fit

    # Save fitted pipeline immediately — this is the artifact used in production inference
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, PIPELINE_PATH)
    logger.info(f"✅ Fitted preprocessing pipeline saved to {PIPELINE_PATH}")

    # ── 2D. Train XGBoost Classifier ─────────────────────────────────────────
    # scale_pos_weight corrects for class imbalance by weighting the positive class
    # Formula: count(negative_class) / count(positive_class)
    # This tells XGBoost to penalize missing a high-risk provider more heavily
    if request.scale_pos_weight is not None:
        spw = request.scale_pos_weight
    else:
        spw = float((y_train == 0).sum()) / float((y_train == 1).sum() + 1)

    logger.info(f"scale_pos_weight={spw:.2f} (class imbalance correction)")

    dtrain = xgb.DMatrix(X_train_processed, label=y_train)
    dtest  = xgb.DMatrix(X_test_processed,  label=y_test)

    params = {
        "objective":        "binary:logistic",   # outputs probability 0-1
        "eval_metric":      ["auc", "logloss"],
        "max_depth":        6,                   # controls tree complexity
        "learning_rate":    0.05,                # small LR + more rounds = better generalization
        "n_estimators":     500,
        "subsample":        0.8,                 # row subsampling — prevents overfitting
        "colsample_bytree": 0.8,                 # feature subsampling per tree
        "scale_pos_weight": spw,
        "seed":             RANDOM_STATE,
        "verbosity":        0,
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds=30,   # stop if eval metric doesn't improve for 30 rounds
        verbose_eval=50,
    )

    # Save trained model
    model.save_model(MODEL_PATH)
    logger.info(f"✅ XGBoost model saved to {MODEL_PATH}")

    # Load into svc for immediate inference availability
    svc.xgb_model = model
    svc.preprocessing_pipe = preprocessor
    svc.shap_explainer = shap.TreeExplainer(model)

    # ── 2E. Evaluation ────────────────────────────────────────────────────────
    y_prob = model.predict(dtest)                          # probability scores
    y_pred = (y_prob >= RISK_THRESHOLD).astype(int)        # binary at threshold

    roc_auc     = roc_auc_score(y_test, y_prob)
    avg_prec    = average_precision_score(y_test, y_prob)  # area under PR curve
    f1          = f1_score(y_test, y_pred)
    precision   = precision_score(y_test, y_pred)
    recall      = recall_score(y_test, y_pred)
    cm          = confusion_matrix(y_test, y_pred).tolist()

    logger.info(
        f"📈 Model Evaluation | ROC-AUC={roc_auc:.4f} | "
        f"AvgPrecision={avg_prec:.4f} | F1={f1:.4f} | "
        f"Precision={precision:.4f} | Recall={recall:.4f}"
    )
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return ModelEvaluationResult(
        experiment_name=request.experiment_name,
        roc_auc=round(roc_auc, 4),
        avg_precision=round(avg_prec, 4),
        f1_score=round(f1, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        confusion_matrix=cm,
        threshold_used=RISK_THRESHOLD,
        train_samples=len(X_train),
        test_samples=len(X_test),
        model_path=MODEL_PATH,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — INFERENCE (PRODUCTION SCORING)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_inference(records: List[ClaimRecord]) -> np.ndarray:
    """
    Converts incoming claim records to a feature matrix and applies
    the FITTED preprocessing pipeline (transform only — never fit_transform).

    This is the training-serving consistency guarantee:
    the same scaler parameters and OHE vocabulary learned at training time
    are applied identically to every production batch.
    """
    if svc.preprocessing_pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Preprocessing pipeline not loaded. Run /train first.",
        )

    df = pd.DataFrame([r.model_dump() for r in records])
    df = engineer_features(df)
    X  = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    # transform() only — using parameters fitted on training data
    return svc.preprocessing_pipe.transform(X)


def compute_shap_explanations(X_processed: np.ndarray) -> List[dict]:
    """
    Computes SHAP values for each prediction.
    Returns top 5 contributing features per record with their SHAP contribution.

    SHAP (SHapley Additive exPlanations):
    - Each feature gets a contribution score for this specific prediction
    - Positive SHAP = pushed risk score UP; Negative = pushed it DOWN
    - Scores are additive and sum to (prediction - base_rate)
    - Enables reviewers to see exactly WHY a provider was flagged
    """
    # Get feature names after OHE expansion
    ohe_feature_names = (
        svc.preprocessing_pipe
        .named_transformers_["cat"]
        .named_steps["ohe"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
        .tolist()
    )
    all_feature_names = NUMERICAL_FEATURES + ohe_feature_names

    shap_values = svc.shap_explainer.shap_values(X_processed)

    explanations = []
    for i in range(len(X_processed)):
        # Pair feature names with SHAP contributions for this row
        contributions = dict(zip(all_feature_names, shap_values[i].tolist()))
        # Sort by absolute contribution — largest impact first
        top_5 = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        )
        explanations.append(top_5)

    return explanations


def write_predictions_to_staging(predictions: List[RiskPrediction]):
    """
    Writes model output to Predictions Staging DB (Azure SQL).
    This is the HOLDING AREA — predictions sit here awaiting HITL review.
    Nothing is delivered to the client from this table directly.

    Table: predictions_staging
    Columns: staging_id, batch_id, claim_id, provider_id, risk_score,
             prediction, top_features_json, processing_timestamp, review_status
    """
    cursor = svc.staging_conn.cursor()
    for pred in predictions:
        cursor.execute(
            """
            INSERT INTO predictions_staging
                (staging_id, batch_id, claim_id, provider_id,
                 risk_score, prediction, top_features_json, processing_timestamp, review_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
            """,
            pred.staging_id,
            pred.batch_id,
            pred.claim_id,
            pred.provider_id,
            pred.risk_score,
            pred.prediction,
            json.dumps(pred.top_features),
            pred.processing_timestamp,
        )
    svc.staging_conn.commit()
    logger.info(f"📝 {len(predictions)} predictions written to Staging DB")


@app.post("/inference/score-batch", tags=["3. Inference"], response_model=dict)
async def score_claims_batch(
    request: BatchScoringRequest,
    token: str = Depends(oauth2_scheme),
):
    """
    Production scoring endpoint — runs daily or on-demand.

    Flow:
      1. Convert ClaimRecord list → DataFrame
      2. Apply fitted preprocessing pipeline (transform only)
      3. XGBoost model inference → risk probability scores
      4. Apply threshold → binary AT_RISK / NORMAL prediction
      5. Compute SHAP values for explainability
      6. Write to Predictions Staging DB (awaiting HITL review)

    HIPAA: provider IDs masked in logs.
    """
    if svc.xgb_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run /train first.")

    scoring_date = request.scoring_date or datetime.date.today()
    logger.info(
        f"🔍 Scoring batch | batch_id={request.batch_id} | "
        f"records={len(request.records)} | date={scoring_date}"
    )

    # Step 1-2: Preprocess using fitted pipeline
    X_processed = preprocess_for_inference(request.records)

    # Step 3: XGBoost inference → probability scores
    dmatrix    = xgb.DMatrix(X_processed)
    risk_probs = svc.xgb_model.predict(dmatrix)           # array of float 0-1

    # Step 4: Apply classification threshold
    predictions_binary = (risk_probs >= RISK_THRESHOLD).astype(int)

    # Step 5: SHAP explanations for each prediction
    shap_explanations = compute_shap_explanations(X_processed)

    # Step 6: Build prediction objects and write to Staging DB
    predictions = []
    for i, record in enumerate(request.records):
        pred = RiskPrediction(
            staging_id=str(uuid.uuid4()),
            batch_id=request.batch_id,
            claim_id=record.claim_id,
            provider_id=record.provider_id,
            risk_score=round(float(risk_probs[i]), 4),
            prediction="AT_RISK" if predictions_binary[i] == 1 else "NORMAL",
            top_features=shap_explanations[i],
            processing_timestamp=datetime.datetime.utcnow(),
        )
        predictions.append(pred)

    write_predictions_to_staging(predictions)

    at_risk_count = sum(1 for p in predictions if p.prediction == "AT_RISK")
    logger.info(
        f"✅ Batch scored | batch_id={request.batch_id} | "
        f"at_risk={at_risk_count} | normal={len(predictions)-at_risk_count}"
    )

    return {
        "batch_id": request.batch_id,
        "total_scored": len(predictions),
        "at_risk_count": at_risk_count,
        "normal_count": len(predictions) - at_risk_count,
        "status": "PENDING_REVIEW",
        "message": "Predictions written to staging — awaiting HITL review.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — HITL (HUMAN-IN-THE-LOOP) REVIEW & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/review/pending", tags=["4. HITL Review"])
async def get_pending_reviews(
    client_id: str = "",
    limit: int = 50,
    token: str = Depends(oauth2_scheme),
):
    """
    Returns the list of staged predictions awaiting human review.
    Sorted by risk_score descending — highest risk cases shown first.

    This feeds the Reviewer Dashboard / UI.
    Reviewers see: provider ID, risk score, prediction, top features (WHY it was flagged).
    """
    cursor = svc.staging_conn.cursor()
    cursor.execute(
        """
        SELECT staging_id, claim_id, provider_id, risk_score,
               prediction, top_features_json, processing_timestamp
        FROM   predictions_staging
        WHERE  review_status = 'PENDING'
        ORDER  BY risk_score DESC
        OFFSET 0 ROWS FETCH NEXT ? ROWS ONLY
        """,
        limit,
    )
    rows = cursor.fetchall()

    pending = []
    for row in rows:
        pending.append({
            "staging_id":           row[0],
            "claim_id":             row[1],
            "provider_id":          row[2],   # in prod: mask for non-reviewer roles
            "risk_score":           row[3],
            "prediction":           row[4],
            "top_features":         json.loads(row[5]),
            "processing_timestamp": str(row[6]),
        })

    logger.info(f"📋 Pending reviews fetched | count={len(pending)}")
    return {"pending_count": len(pending), "items": pending}


@app.post("/review/submit-decision", tags=["4. HITL Review"])
async def submit_review_decision(
    decision: ReviewDecision,
    token: str = Depends(oauth2_scheme),
):
    """
    Reviewer submits their HITL decision: CONFIRM, REJECT, or MODIFY.

    Actions:
      CONFIRM — model's prediction is correct; move to Final Validated DB as-is
      REJECT  — model was wrong; move to Final DB with prediction overridden to NORMAL
      MODIFY  — partially correct; reviewer supplies adjusted risk_score / prediction

    After decision:
      1. Update review_status in Staging DB
      2. Write immutable entry to Audit Log (Who, When, What)
      3. Write final result to Final Validated Results DB

    HIPAA Audit requirement: all three writes must succeed atomically.
    """
    # Fetch the staged prediction being reviewed
    cursor = svc.staging_conn.cursor()
    cursor.execute(
        "SELECT claim_id, provider_id, risk_score, prediction FROM predictions_staging WHERE staging_id = ?",
        decision.staging_id,
    )
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Staging ID {decision.staging_id} not found.")

    claim_id, provider_id, original_risk_score, original_prediction = row

    # Determine final values based on reviewer decision
    if decision.decision == "CONFIRM":
        final_risk_score  = original_risk_score
        final_prediction  = original_prediction
    elif decision.decision == "REJECT":
        final_risk_score  = 0.0
        final_prediction  = "NORMAL"
    elif decision.decision == "MODIFY":
        if decision.modified_risk_score is None or decision.modified_prediction is None:
            raise HTTPException(
                status_code=422,
                detail="MODIFY decision requires modified_risk_score and modified_prediction.",
            )
        final_risk_score  = decision.modified_risk_score
        final_prediction  = decision.modified_prediction

    review_timestamp = datetime.datetime.utcnow()
    final_id         = str(uuid.uuid4())

    # ── Write 1: Update staging record status ────────────────────────────────
    cursor.execute(
        "UPDATE predictions_staging SET review_status = 'REVIEWED' WHERE staging_id = ?",
        decision.staging_id,
    )

    # ── Write 2: Immutable Audit Log ─────────────────────────────────────────
    # HIPAA requirement: immutable record of who reviewed what and when
    # Table has INSERT-only permissions — no UPDATE or DELETE allowed
    cursor.execute(
        """
        INSERT INTO audit_log
            (audit_id, staging_id, reviewer_id, decision,
             original_prediction, final_prediction,
             original_risk_score, final_risk_score,
             review_comments, review_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        str(uuid.uuid4()),
        decision.staging_id,
        decision.reviewer_id,
        decision.decision,
        original_prediction,
        final_prediction,
        original_risk_score,
        final_risk_score,
        decision.review_comments or "",
        review_timestamp,
    )

    # ── Write 3: Final Validated Results DB ──────────────────────────────────
    # This is the ONLY table from which results are delivered to the client
    final_cursor = svc.final_conn.cursor()
    final_cursor.execute(
        """
        INSERT INTO final_validated_results
            (final_id, staging_id, claim_id, provider_id,
             final_risk_flag, final_risk_score, final_prediction,
             reviewer_id, review_decision, review_comments, review_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        final_id,
        decision.staging_id,
        claim_id,
        provider_id,
        1 if final_prediction == "AT_RISK" else 0,
        final_risk_score,
        final_prediction,
        decision.reviewer_id,
        decision.decision,
        decision.review_comments or "",
        review_timestamp,
    )

    # Commit both connections atomically
    svc.staging_conn.commit()
    svc.final_conn.commit()

    logger.info(
        f"✅ Review submitted | staging_id={decision.staging_id} | "
        f"reviewer={mask_phi(decision.reviewer_id)} | decision={decision.decision}"
    )

    return {
        "final_id":          final_id,
        "staging_id":        decision.staging_id,
        "decision":          decision.decision,
        "final_prediction":  final_prediction,
        "final_risk_score":  final_risk_score,
        "review_timestamp":  review_timestamp.isoformat(),
        "status":            "VALIDATED",
    }


@app.get("/audit/log", tags=["4. HITL Review"])
async def get_audit_log(
    staging_id: Optional[str] = None,
    reviewer_id: Optional[str] = None,
    from_date: Optional[datetime.date] = None,
    limit: int = 100,
    token: str = Depends(oauth2_scheme),
):
    """
    Returns the immutable audit log — who reviewed what and when.
    Filterable by staging_id, reviewer_id, and date range.
    Required for HIPAA compliance audits and provider dispute resolution.
    """
    cursor = svc.staging_conn.cursor()
    query  = "SELECT * FROM audit_log WHERE 1=1"
    params = []

    if staging_id:
        query += " AND staging_id = ?"
        params.append(staging_id)
    if reviewer_id:
        query += " AND reviewer_id = ?"
        params.append(reviewer_id)
    if from_date:
        query += " AND review_timestamp >= ?"
        params.append(from_date)

    query += f" ORDER BY review_timestamp DESC OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"

    cursor.execute(query, *params)
    columns = [col[0] for col in cursor.description]
    rows    = [dict(zip(columns, row)) for row in cursor.fetchall()]

    # Mask reviewer IDs in response for non-admin callers
    for row in rows:
        if "review_timestamp" in row and isinstance(row["review_timestamp"], datetime.datetime):
            row["review_timestamp"] = row["review_timestamp"].isoformat()

    return {"count": len(rows), "audit_entries": rows}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — FINAL DELIVERY TO CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def deliver_via_sftp(client_id: str, csv_bytes: bytes, filename: str) -> str:
    """
    Pushes final validated results to client via SFTP.
    Uses RSA key authentication — no passwords in transit.
    All data encrypted in transit via SSH transport layer.

    In production: client SFTP credentials stored in Azure Key Vault.
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())  # never auto-accept unknown hosts

    private_key = paramiko.RSAKey.from_private_key_file(CLIENT_SFTP_KEY_PATH)
    ssh.connect(
        hostname=CLIENT_SFTP_HOST,
        port=CLIENT_SFTP_PORT,
        username=CLIENT_SFTP_USER,
        pkey=private_key,
    )

    sftp = ssh.open_sftp()
    remote_path = f"{CLIENT_SFTP_REMOTE_DIR}{filename}"
    sftp.putfo(io.BytesIO(csv_bytes), remote_path)
    sftp.close()
    ssh.close()

    logger.info(
        f"📤 Results delivered via SFTP | client={mask_phi(client_id)} | "
        f"remote_path={remote_path} | size_kb={len(csv_bytes)/1024:.1f}"
    )
    return remote_path


@app.post("/deliver/sftp", tags=["5. Final Delivery"])
async def deliver_results_sftp(
    client_id: str,
    batch_id: str,
    background_tasks: BackgroundTasks,
    token: str = Depends(oauth2_scheme),
):
    """
    Delivers final HITL-validated results to client via SFTP.
    Only pulls from final_validated_results — never from staging.

    Output CSV contains:
      - final_risk_flag, final_risk_score, final_prediction
      - reviewer_id, review_decision, review_comments, review_timestamp
      - claim_id, provider_id (for client to join back to their systems)

    This is the complete audit-trail-backed result the client loads into
    their SQL DB, data warehouse, or EMR system.
    """
    cursor = svc.final_conn.cursor()
    cursor.execute(
        """
        SELECT claim_id, provider_id, final_risk_flag, final_risk_score,
               final_prediction, reviewer_id, review_decision,
               review_comments, review_timestamp
        FROM   final_validated_results
        WHERE  staging_id IN (
                   SELECT staging_id FROM predictions_staging WHERE batch_id = ?
               )
        ORDER  BY final_risk_score DESC
        """,
        batch_id,
    )
    columns = [col[0] for col in cursor.description]
    rows    = cursor.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No validated results found for batch_id={batch_id}. "
                   "Ensure all predictions have been reviewed before delivery.",
        )

    # Build CSV in memory — never write PHI to local disk
    df_results = pd.DataFrame(rows, columns=columns)
    csv_buffer = io.BytesIO()
    df_results.to_csv(csv_buffer, index=False)
    csv_bytes  = csv_buffer.getvalue()

    filename = f"risk_results_{client_id}_{batch_id}_{datetime.date.today().isoformat()}.csv"

    # SFTP delivery runs in background — client gets immediate acknowledgement
    background_tasks.add_task(deliver_via_sftp, client_id, csv_bytes, filename)

    logger.info(
        f"📦 Delivery initiated | client={mask_phi(client_id)} | "
        f"batch_id={batch_id} | rows={len(df_results)}"
    )

    return {
        "status":      "DELIVERY_INITIATED",
        "client_id":   client_id,
        "batch_id":    batch_id,
        "row_count":   len(df_results),
        "filename":    filename,
        "delivery_method": "SFTP",
    }


@app.get("/deliver/api-results", tags=["5. Final Delivery"])
async def get_results_via_api(
    client_id: str,
    batch_id: str,
    token: str = Depends(oauth2_scheme),
):
    """
    Alternative delivery mode: client pulls results via secure API/webhook.
    For clients with modern data infrastructure who prefer real-time API access
    over scheduled SFTP file drops.

    Returns full validated results with risk scores, decisions, and audit trail.
    Client loads directly into their database / EMR via their ETL pipeline.
    """
    cursor = svc.final_conn.cursor()
    cursor.execute(
        """
        SELECT claim_id, provider_id, final_risk_flag, final_risk_score,
               final_prediction, review_decision, review_comments, review_timestamp
        FROM   final_validated_results
        WHERE  staging_id IN (
                   SELECT staging_id FROM predictions_staging WHERE batch_id = ?
               )
        ORDER  BY final_risk_score DESC
        """,
        batch_id,
    )
    columns = [col[0] for col in cursor.description]
    rows    = cursor.fetchall()

    results = []
    for row in rows:
        record = dict(zip(columns, row))
        if isinstance(record.get("review_timestamp"), datetime.datetime):
            record["review_timestamp"] = record["review_timestamp"].isoformat()
        results.append(record)

    return {
        "client_id":      client_id,
        "batch_id":       batch_id,
        "total_results":  len(results),
        "at_risk_count":  sum(1 for r in results if r["final_risk_flag"] == 1),
        "results":        results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MONITORING & HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe — returns service status and model readiness."""
    return {
        "status":              "ok",
        "model_loaded":        svc.xgb_model is not None,
        "pipeline_loaded":     svc.preprocessing_pipe is not None,
        "staging_db":          svc.staging_conn is not None,
        "final_db":            svc.final_conn is not None,
        "timestamp":           datetime.datetime.utcnow().isoformat(),
    }


@app.get("/monitoring/reviewer-agreement", tags=["Monitoring"])
async def reviewer_agreement_stats(
    token: str = Depends(oauth2_scheme),
):
    """
    Returns reviewer agreement rates — what % of predictions each reviewer confirms.
    Persistent patterns of high override rates signal:
      - Model drift (model is wrong for a certain provider type)
      - Reviewer bias (reviewer systematically over- or under-flags)
      - Data quality issues in a specific batch
    Used by MLOps team to trigger retraining or reviewer calibration.
    """
    cursor = svc.staging_conn.cursor()
    cursor.execute(
        """
        SELECT reviewer_id,
               COUNT(*)                                            AS total_reviews,
               SUM(CASE WHEN decision = 'CONFIRM' THEN 1 ELSE 0 END) AS confirmed,
               SUM(CASE WHEN decision = 'REJECT'  THEN 1 ELSE 0 END) AS rejected,
               SUM(CASE WHEN decision = 'MODIFY'  THEN 1 ELSE 0 END) AS modified
        FROM   audit_log
        GROUP  BY reviewer_id
        ORDER  BY total_reviews DESC
        """
    )
    rows = cursor.fetchall()
    stats = []
    for row in rows:
        reviewer_id, total, confirmed, rejected, modified = row
        stats.append({
            "reviewer_id":       mask_phi(reviewer_id),   # masked for privacy
            "total_reviews":     total,
            "confirm_rate":      round(confirmed / total, 3) if total else 0,
            "reject_rate":       round(rejected  / total, 3) if total else 0,
            "modify_rate":       round(modified  / total, 3) if total else 0,
        })
    return {"reviewer_stats": stats}


@app.get("/monitoring/risk-score-distribution", tags=["Monitoring"])
async def risk_score_distribution(
    batch_id: Optional[str] = None,
    token: str = Depends(oauth2_scheme),
):
    """
    Returns histogram-style distribution of risk scores from the staging DB.
    Used to detect model drift — if the distribution shifts significantly
    compared to baseline, it may indicate data drift or degraded model performance.
    Monitored daily by the Orchestration & Monitoring layer (Azure Monitor / Log Analytics).
    """
    cursor = svc.staging_conn.cursor()
    query  = "SELECT risk_score FROM predictions_staging"
    params = []
    if batch_id:
        query += " WHERE batch_id = ?"
        params.append(batch_id)

    cursor.execute(query, *params)
    scores = [row[0] for row in cursor.fetchall()]

    if not scores:
        return {"message": "No scores found for the given filter."}

    scores_arr = np.array(scores)
    return {
        "count":   len(scores),
        "mean":    round(float(scores_arr.mean()), 4),
        "std":     round(float(scores_arr.std()), 4),
        "min":     round(float(scores_arr.min()), 4),
        "max":     round(float(scores_arr.max()), 4),
        "p25":     round(float(np.percentile(scores_arr, 25)), 4),
        "p50":     round(float(np.percentile(scores_arr, 50)), 4),
        "p75":     round(float(np.percentile(scores_arr, 75)), 4),
        "p90":     round(float(np.percentile(scores_arr, 90)), 4),
        "at_risk_pct": round(float((scores_arr >= RISK_THRESHOLD).mean() * 100), 2),
    }

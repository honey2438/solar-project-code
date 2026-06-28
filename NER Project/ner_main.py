"""
ner_main.py — Contract NER Extraction System (AI Inference + HITL Workflow)
FastAPI Backend — Single File

End-to-End Architecture (13-Step Orchestration Flow):
  1.  CLIENT ENVIRONMENT      — Secure TLS file transfer (SFTP / Azure Blob / S3)
  2.  INGESTION JOB           — Poller detects new files, captures metadata + timestamp
  3.  PREPROCESSING POD       — pdfplumber / OCR extracts raw text from PDFs
  4.  TEXT ENGINE             — Normalizes text while PRESERVING CASE for NER accuracy
  5.  RAW INGEST BUCKET       — Stores original files for traceability and reprocessing
  6.  NER ENGINE              — spaCy en_core_web_trf extracts entities from text
  7.  ATTRIBUTE NORMALIZATION — Standardizes values + computes confidence scores
  8.  RECORD ASSEMBLY         — Builds enriched structured contract records
  9.  STAGING DATABASE        — Persists records + applies quality/business rules
  10. AUTO-APPROVE            — High-confidence records sync directly to Client DB
  11. FLAG FOR HITL           — Low-confidence / rule-failing records → human review
  12. HITL REVIEW             — Reviewer approves / rejects → writes back to staging
  13. SYNC TO PRODUCTION      — Approved records → final Client DB Table

Tech Stack:
  - FastAPI          : REST API framework
  - spaCy            : NER engine (en_core_web_trf — transformer-backed)
  - pdfplumber       : Text-based PDF extraction (preserves layout structure)
  - pytesseract      : OCR for scanned/image-based PDFs
  - Azure Blob / S3  : Raw Ingest Bucket (original file storage)
  - SQLAlchemy       : ORM for Staging DB + Client DB (Azure SQL / PostgreSQL)
  - Paramiko         : SFTP file transfer
  - Pydantic         : Request/response validation
  - APScheduler      : Polling scheduler for ingestion job
  - dateparser       : Robust date normalization from free-form text
  - re / hashlib     : Text cleaning and file deduplication
"""

import os
import io
import re
import uuid
import json
import time
import hashlib
import logging
import datetime
import threading
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

# ── PDF Extraction ─────────────────────────────────────────────────────────
import pdfplumber
import pytesseract
from PIL import Image
import fitz  # PyMuPDF — used to detect if PDF is text-based or image-based

# ── NER ────────────────────────────────────────────────────────────────────
import spacy
from spacy.tokens import Doc

# ── Data & Normalization ───────────────────────────────────────────────────
import dateparser
import re
from babel.numbers import parse_decimal

# ── Storage ────────────────────────────────────────────────────────────────
import boto3
import paramiko
from azure.storage.blob import BlobServiceClient

# ── Database ───────────────────────────────────────────────────────────────
from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    Boolean, DateTime, Text, JSON, Enum
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ── Scheduling ─────────────────────────────────────────────────────────────
from apscheduler.schedulers.background import BackgroundScheduler

# ── FastAPI ────────────────────────────────────────────────────────────────
from fastapi import (
    FastAPI, File, UploadFile, HTTPException,
    BackgroundTasks, Depends, Query
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ner_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Config — replace with Azure Key Vault / environment variables in production
# ─────────────────────────────────────────────────────────────────────────────
AZURE_BLOB_CONN_STR     = os.getenv("AZURE_BLOB_CONN_STR", "")
RAW_INGEST_CONTAINER    = os.getenv("RAW_INGEST_CONTAINER", "raw-ingest-bucket")

AWS_BUCKET_NAME         = os.getenv("AWS_BUCKET_NAME", "raw-ingest-bucket")
AWS_REGION              = os.getenv("AWS_REGION", "us-east-1")

STAGING_DB_URL          = os.getenv("STAGING_DB_URL", "sqlite:///./staging.db")
CLIENT_DB_URL           = os.getenv("CLIENT_DB_URL", "sqlite:///./client_production.db")

SFTP_HOST               = os.getenv("SFTP_HOST", "")
SFTP_PORT               = int(os.getenv("SFTP_PORT", "22"))
SFTP_USER               = os.getenv("SFTP_USER", "")
SFTP_KEY_PATH           = os.getenv("SFTP_KEY_PATH", "/keys/sftp_rsa")
SFTP_REMOTE_DIR         = os.getenv("SFTP_REMOTE_DIR", "/incoming/contracts/")

# NER model — en_core_web_trf is the transformer-backed pipeline
# Fallback to en_core_web_sm if trf not installed (for lightweight environments)
SPACY_MODEL             = os.getenv("SPACY_MODEL", "en_core_web_trf")

# Confidence threshold — records above this auto-approve; below go to HITL
AUTO_APPROVE_THRESHOLD  = float(os.getenv("AUTO_APPROVE_THRESHOLD", "0.80"))

# Entity types we extract from contracts
CONTRACT_ENTITY_TYPES = [
    "ORG",          # Organization / party names
    "PERSON",       # Individual signatories
    "DATE",         # Effective date, end date, renewal date
    "MONEY",        # Contract value, payment amounts
    "GPE",          # Jurisdiction / governing law location
    "PERCENT",      # Percentages (interest rates, penalties)
    "CARDINAL",     # Numbers (term length in months, etc.)
    # Custom types (require fine-tuning):
    "CONTRACT_ID",  # Contract / agreement reference numbers
    "DURATION",     # "3 years", "18 months"
    "CLAUSE_TYPE",  # "indemnification", "termination", "renewal"
]

# Business rules — hard rules prevent auto-approval regardless of confidence
HARD_REQUIRED_FIELDS = ["counterparty_name", "effective_date"]


# ─────────────────────────────────────────────────────────────────────────────
# Database Models (SQLAlchemy ORM)
# ─────────────────────────────────────────────────────────────────────────────
Base = declarative_base()


class IngestedFile(Base):
    """
    Tracks every file that enters the system.
    Used by the poller to detect new files and avoid reprocessing.
    """
    __tablename__ = "ingested_files"

    file_id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename            = Column(String, nullable=False)
    source_path         = Column(String)                   # original SFTP / Blob path
    blob_path           = Column(String)                   # path in Raw Ingest Bucket
    file_hash           = Column(String)                   # SHA-256 for deduplication
    file_size_bytes     = Column(Integer)
    client_id           = Column(String)
    ingested_at         = Column(DateTime, default=datetime.datetime.utcnow)
    processing_status   = Column(String, default="PENDING")  # PENDING / PROCESSING / DONE / FAILED
    error_message       = Column(Text, nullable=True)


class StagingRecord(Base):
    """
    Holds extracted contract records awaiting quality checks and HITL review.
    This is the gate — nothing reaches the Client DB without going through here.
    """
    __tablename__ = "staging_records"

    record_id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    file_id             = Column(String)
    client_id           = Column(String)

    # Extracted fields (normalized)
    counterparty_name   = Column(String, nullable=True)
    party_role          = Column(String, nullable=True)    # vendor / client / guarantor
    effective_date      = Column(String, nullable=True)    # ISO 8601
    end_date            = Column(String, nullable=True)
    renewal_date        = Column(String, nullable=True)
    contract_value      = Column(Float, nullable=True)
    currency            = Column(String, nullable=True)
    jurisdiction        = Column(String, nullable=True)
    contract_duration   = Column(String, nullable=True)    # derived
    days_until_renewal  = Column(Integer, nullable=True)   # derived
    signatory_name      = Column(String, nullable=True)
    contract_id_ref     = Column(String, nullable=True)

    # Confidence scores per field (JSON)
    confidence_scores   = Column(JSON, default=dict)       # {field: score}
    overall_confidence  = Column(Float, default=0.0)

    # Raw NER output for traceability
    raw_entities_json   = Column(JSON, default=list)       # [{text, label, start, end, score}]

    # Quality rules
    rule_flags          = Column(JSON, default=list)       # [{rule, severity, message}]
    has_hard_failure    = Column(Boolean, default=False)

    # Review state
    review_status       = Column(
        String, default="PENDING"
    )  # PENDING / AUTO_APPROVED / FLAGGED / APPROVED / REJECTED
    reviewer_id         = Column(String, nullable=True)
    reviewer_decision   = Column(String, nullable=True)
    reviewer_comments   = Column(Text, nullable=True)
    reviewed_at         = Column(DateTime, nullable=True)

    created_at          = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at          = Column(DateTime, default=datetime.datetime.utcnow,
                                 onupdate=datetime.datetime.utcnow)


class AuditLog(Base):
    """
    Immutable audit trail — INSERT only, never UPDATE or DELETE.
    Records every review decision with who, when, and what.
    """
    __tablename__ = "audit_log"

    audit_id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    record_id           = Column(String, nullable=False)
    reviewer_id         = Column(String, nullable=False)
    action              = Column(String, nullable=False)   # APPROVE / REJECT / MODIFY
    before_state        = Column(JSON)                     # snapshot before decision
    after_state         = Column(JSON)                     # snapshot after decision
    comments            = Column(Text, nullable=True)
    created_at          = Column(DateTime, default=datetime.datetime.utcnow)


class ClientContractRecord(Base):
    """
    Final production table — validated contracts delivered to the client.
    Only auto-approved or human-approved records reach here.
    """
    __tablename__ = "client_contracts"

    contract_record_id  = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    staging_record_id   = Column(String)
    file_id             = Column(String)
    client_id           = Column(String)

    counterparty_name   = Column(String)
    party_role          = Column(String)
    effective_date      = Column(String)
    end_date            = Column(String)
    renewal_date        = Column(String)
    contract_value      = Column(Float)
    currency            = Column(String)
    jurisdiction        = Column(String)
    contract_duration   = Column(String)
    days_until_renewal  = Column(Integer)
    signatory_name      = Column(String)
    contract_id_ref     = Column(String)

    approval_type       = Column(String)                   # AUTO / HUMAN
    reviewer_id         = Column(String, nullable=True)
    synced_at           = Column(DateTime, default=datetime.datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ReviewDecision(BaseModel):
    record_id: str
    reviewer_id: str
    decision: str = Field(..., pattern="^(APPROVE|REJECT)$")
    # Optional field corrections the reviewer makes
    corrected_counterparty_name: Optional[str] = None
    corrected_effective_date: Optional[str] = None
    corrected_end_date: Optional[str] = None
    corrected_contract_value: Optional[float] = None
    corrected_jurisdiction: Optional[str] = None
    comments: Optional[str] = None


class PipelineRunRequest(BaseModel):
    """Manually trigger a pipeline run on a specific file already in the Raw Ingest Bucket."""
    file_id: str
    client_id: str


class StagingRecordOut(BaseModel):
    record_id: str
    file_id: str
    client_id: str
    counterparty_name: Optional[str]
    effective_date: Optional[str]
    end_date: Optional[str]
    renewal_date: Optional[str]
    contract_value: Optional[float]
    currency: Optional[str]
    jurisdiction: Optional[str]
    contract_duration: Optional[str]
    days_until_renewal: Optional[int]
    signatory_name: Optional[str]
    overall_confidence: float
    confidence_scores: dict
    rule_flags: list
    has_hard_failure: bool
    review_status: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────────────────────
# Service Singletons
# ─────────────────────────────────────────────────────────────────────────────

class Services:
    nlp                 = None   # spaCy NER pipeline
    blob_client         = None   # Azure Blob ServiceClient
    s3_client           = None   # AWS S3 client
    staging_engine      = None   # SQLAlchemy engine — Staging DB
    client_engine       = None   # SQLAlchemy engine — Client DB
    StagingSession      = None   # Session factory — Staging DB
    ClientSession       = None   # Session factory — Client DB
    scheduler           = None   # APScheduler for polling

svc = Services()


# ─────────────────────────────────────────────────────────────────────────────
# App Lifespan — startup and shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load spaCy model, connect to storage and DBs, start ingestion poller.
    Shutdown: stop scheduler, close connections.
    """
    logger.info("🚀 Starting NER Contract Extraction API...")

    # ── spaCy NER Model ──────────────────────────────────────────────────────
    # en_core_web_trf: transformer-backed, highest accuracy
    # Preserving case is critical — do NOT lowercase before passing to spaCy
    try:
        svc.nlp = spacy.load(SPACY_MODEL)
        logger.info(f"✅ spaCy model loaded: {SPACY_MODEL}")
    except OSError:
        logger.warning(f"⚠️  {SPACY_MODEL} not found — falling back to en_core_web_sm")
        svc.nlp = spacy.load("en_core_web_sm")

    # ── Azure Blob Storage (Raw Ingest Bucket) ───────────────────────────────
    if AZURE_BLOB_CONN_STR:
        svc.blob_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
        logger.info("✅ Azure Blob Storage connected")

    # ── AWS S3 (alternative raw ingest) ─────────────────────────────────────
    svc.s3_client = boto3.client("s3", region_name=AWS_REGION)
    logger.info("✅ AWS S3 client ready")

    # ── Staging DB ────────────────────────────────────────────────────────────
    svc.staging_engine = create_engine(STAGING_DB_URL, echo=False)
    Base.metadata.create_all(svc.staging_engine)
    svc.StagingSession = sessionmaker(bind=svc.staging_engine)
    logger.info("✅ Staging DB ready")

    # ── Client DB ─────────────────────────────────────────────────────────────
    svc.client_engine = create_engine(CLIENT_DB_URL, echo=False)
    Base.metadata.create_all(svc.client_engine)
    svc.ClientSession = sessionmaker(bind=svc.client_engine)
    logger.info("✅ Client DB ready")

    # ── Ingestion Poller Scheduler ────────────────────────────────────────────
    # Polls SFTP / Blob every 5 minutes for new client documents
    svc.scheduler = BackgroundScheduler()
    svc.scheduler.add_job(
        poll_sftp_for_new_files,
        trigger="interval",
        minutes=5,
        id="sftp_poller",
    )
    svc.scheduler.start()
    logger.info("✅ Ingestion poller started (every 5 minutes)")

    logger.info("🟢 NER API ready — all services online")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("🛑 Shutting down...")
    if svc.scheduler and svc.scheduler.running:
        svc.scheduler.shutdown(wait=False)
    logger.info("✅ Shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Contract NER Extraction API",
    description=(
        "End-to-end NER pipeline for extracting structured data from contract PDFs. "
        "Includes HITL review workflow and auto-approval with audit trail."
    ),
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


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — SECURE FILE TRANSFER & RAW INGEST BUCKET
# ─────────────────────────────────────────────────────────────────────────────

def compute_file_hash(content: bytes) -> str:
    """SHA-256 hash for deduplication — same file uploaded twice is detected."""
    return hashlib.sha256(content).hexdigest()


def store_in_raw_ingest_bucket(
    content: bytes,
    client_id: str,
    filename: str,
) -> str:
    """
    Stores the ORIGINAL, UNMODIFIED file in the Raw Ingest Bucket.
    This is the traceability anchor — every extracted value can be traced
    back to this original document.

    Key structure: clients/{client_id}/raw/{date}/{filename}
    Partitioned by client and date for easy lifecycle management.
    """
    blob_path = (
        f"clients/{client_id}/raw/"
        f"{datetime.date.today().isoformat()}/"
        f"{filename}"
    )

    if svc.blob_client:
        # Azure Blob Storage path
        blob = svc.blob_client.get_blob_client(
            container=RAW_INGEST_CONTAINER,
            blob=blob_path,
        )
        blob.upload_blob(content, overwrite=True)
    else:
        # Fallback: AWS S3
        svc.s3_client.put_object(
            Bucket=AWS_BUCKET_NAME,
            Key=blob_path,
            Body=content,
            ContentType="application/pdf",
        )

    logger.info(f"📦 Raw file stored: {blob_path}")
    return blob_path


@app.post("/ingest/upload", tags=["1. Ingestion"])
async def upload_document(
    file: UploadFile = File(...),
    client_id: str = "",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    token: str = Depends(oauth2_scheme),
):
    """
    Step 1: Client uploads a contract PDF via secure API (TLS encrypted).
    - Stores original file in Raw Ingest Bucket (Azure Blob / S3)
    - Registers file in ingested_files table with hash for deduplication
    - Triggers the full processing pipeline as a background task
    """
    content  = await file.read()
    file_hash = compute_file_hash(content)

    # ── Deduplication check ───────────────────────────────────────────────────
    with svc.StagingSession() as db:
        existing = db.query(IngestedFile).filter_by(file_hash=file_hash).first()
        if existing:
            logger.info(f"♻️  Duplicate file detected: {file.filename} — skipping")
            return {
                "status": "duplicate",
                "message": "This file has already been processed.",
                "file_id": existing.file_id,
            }

    # ── Store in Raw Ingest Bucket ────────────────────────────────────────────
    blob_path = store_in_raw_ingest_bucket(content, client_id, file.filename)

    # ── Register in DB ────────────────────────────────────────────────────────
    file_id = str(uuid.uuid4())
    with svc.StagingSession() as db:
        ingested = IngestedFile(
            file_id=file_id,
            filename=file.filename,
            blob_path=blob_path,
            file_hash=file_hash,
            file_size_bytes=len(content),
            client_id=client_id,
            processing_status="PENDING",
        )
        db.add(ingested)
        db.commit()

    # ── Trigger pipeline in background ───────────────────────────────────────
    background_tasks.add_task(run_full_pipeline, file_id, content, client_id)

    logger.info(f"📥 File ingested | file_id={file_id} | client={client_id}")
    return {
        "status":    "ingested",
        "file_id":   file_id,
        "blob_path": blob_path,
        "message":   "Processing pipeline triggered.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — INGESTION JOB POLLER (SFTP)
# ─────────────────────────────────────────────────────────────────────────────

def poll_sftp_for_new_files():
    """
    Step 2: Runs on a schedule (every 5 minutes via APScheduler).
    Connects to client SFTP server, detects new files by comparing against
    the ingested_files state table, downloads new files, and triggers pipeline.

    Uses timestamp capture at ingestion time (not file modification time)
    as the official 'received' timestamp for audit purposes.
    """
    if not SFTP_HOST:
        return  # SFTP not configured — skip

    logger.info("🔍 Polling SFTP for new files...")

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
        private_key = paramiko.RSAKey.from_private_key_file(SFTP_KEY_PATH)
        ssh.connect(SFTP_HOST, port=SFTP_PORT, username=SFTP_USER, pkey=private_key)
        sftp = ssh.open_sftp()

        remote_files = sftp.listdir_attr(SFTP_REMOTE_DIR)

        with svc.StagingSession() as db:
            for remote_file in remote_files:
                if not remote_file.filename.lower().endswith(".pdf"):
                    continue

                # Check if already processed by filename
                existing = db.query(IngestedFile).filter_by(
                    filename=remote_file.filename
                ).first()
                if existing:
                    continue  # Already ingested

                # Download file
                remote_path = f"{SFTP_REMOTE_DIR}{remote_file.filename}"
                buffer = io.BytesIO()
                sftp.getfo(remote_path, buffer)
                content = buffer.getvalue()

                file_hash = compute_file_hash(content)
                file_id   = str(uuid.uuid4())
                blob_path = store_in_raw_ingest_bucket(
                    content, "sftp_client", remote_file.filename
                )

                ingested = IngestedFile(
                    file_id=file_id,
                    filename=remote_file.filename,
                    source_path=remote_path,
                    blob_path=blob_path,
                    file_hash=file_hash,
                    file_size_bytes=len(content),
                    client_id="sftp_client",
                    processing_status="PENDING",
                )
                db.add(ingested)
                db.commit()

                # Run pipeline in a new thread to not block the scheduler
                threading.Thread(
                    target=run_full_pipeline,
                    args=(file_id, content, "sftp_client"),
                    daemon=True,
                ).start()

                logger.info(f"📥 SFTP file picked up: {remote_file.filename}")

        sftp.close()
        ssh.close()

    except Exception as e:
        logger.error(f"❌ SFTP polling error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — PREPROCESSING POD (PDF TEXT EXTRACTION)
# ─────────────────────────────────────────────────────────────────────────────

def is_text_based_pdf(pdf_bytes: bytes) -> bool:
    """
    Detects whether a PDF has an embedded text layer (text-based)
    or is purely an image/scan (image-based requiring OCR).

    Uses PyMuPDF to check if any text blocks exist on the first page.
    """
    try:
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        text = page.get_text("text")
        doc.close()
        # If at least 20 characters of text found, it's text-based
        return len(text.strip()) > 20
    except Exception:
        return False


def extract_text_pdfplumber(pdf_bytes: bytes) -> str:
    """
    Step 3A: Text-based PDF extraction using pdfplumber.
    pdfplumber preserves layout including tables, columns, and paragraph breaks.
    Tables are extracted in row/column order which preserves structured data.
    """
    full_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract tables first — they contain structured contract data
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if row:
                        row_text = " | ".join(
                            cell.strip() if cell else "" for cell in row
                        )
                        full_text.append(f"[TABLE] {row_text}")

            # Extract remaining text (non-table)
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                full_text.append(f"[PAGE {page_num}]\n{text}")

    return "\n\n".join(full_text)


def extract_text_ocr(pdf_bytes: bytes) -> str:
    """
    Step 3B: Image-based PDF extraction using OCR (Tesseract).
    Converts each PDF page to an image, then applies OCR.
    Used for scanned documents, faxes, and photographed contracts.
    """
    full_text = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at 2x resolution for better OCR accuracy
        mat  = fitz.Matrix(2.0, 2.0)
        pix  = page.get_pixmap(matrix=mat)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Tesseract OCR — preserve layout with psm 6 (uniform block of text)
        ocr_text = pytesseract.image_to_string(img, config="--psm 6")
        if ocr_text.strip():
            full_text.append(f"[PAGE {page_num + 1} — OCR]\n{ocr_text}")

    doc.close()
    return "\n\n".join(full_text)


def preprocess_pdf(pdf_bytes: bytes) -> tuple[str, str]:
    """
    Step 3: Routes to text extraction or OCR based on PDF type.
    Returns (extracted_text, extraction_method).
    """
    if is_text_based_pdf(pdf_bytes):
        text   = extract_text_pdfplumber(pdf_bytes)
        method = "pdfplumber"
    else:
        text   = extract_text_ocr(pdf_bytes)
        method = "tesseract_ocr"

    logger.info(f"📄 Text extracted via {method} | chars={len(text)}")
    return text, method


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — TEXT ENGINE (NORMALIZATION — PRESERVE CASE)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(raw_text: str) -> str:
    """
    Step 4: Cleans and normalizes text for NER processing.

    CRITICAL: DO NOT LOWERCASE.
    Reasons:
      1. en_core_web_trf was trained on cased text — lowercasing degrades NER accuracy
         for proper nouns (exactly what we are extracting).
      2. In legal documents, capitalization signals defined terms —
         'the Agreement', 'the Services', 'the Company' vs their generic meanings.
      3. Organization names rely on case for correct recognition.

    What we DO normalize:
      - Encoding artifacts (smart quotes → straight quotes)
      - Excessive whitespace (multiple spaces / blank lines)
      - Form feed characters and other non-printable characters
      - Hyphenated line breaks from PDF extraction (word- \nbreak → wordbreak)
    """
    # Replace smart quotes with straight quotes
    text = raw_text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Fix hyphenated line breaks from PDF column layouts
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize whitespace — collapse multiple spaces but preserve paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove non-printable characters (form feeds, null bytes, etc.)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Remove page markers added by pdfplumber (not needed for NER)
    text = re.sub(r"\[PAGE \d+\]\n?", "", text)
    text = re.sub(r"\[TABLE\] ", "", text)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — NER ENGINE (spaCy en_core_web_trf)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 5000, overlap: int = 500) -> List[str]:
    """
    Splits long documents into overlapping chunks for NER processing.
    Overlap ensures entities at chunk boundaries are captured in at least one chunk.

    chunk_size=5000 chars is safe for en_core_web_trf's context window.
    overlap=500 chars ensures boundary entities appear in two adjacent chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        # Don't split mid-sentence — find the last sentence boundary before end
        if end < len(text):
            last_period = text.rfind(".", start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1
        chunks.append(text[start:end])
        start = end - overlap  # overlap for boundary entity capture

    return chunks


def run_ner(text: str) -> List[Dict]:
    """
    Step 6: Runs spaCy NER pipeline on normalized text.
    Returns list of entity dicts with text, label, character positions,
    and the sentence context for provenance.

    Handles long documents by chunking with overlap, then deduplicating
    entities that appear in the overlap region of adjacent chunks.
    """
    if svc.nlp is None:
        raise RuntimeError("spaCy model not loaded")

    chunks  = chunk_text(text)
    all_entities = []
    char_offset  = 0

    for chunk_idx, chunk in enumerate(chunks):
        doc = svc.nlp(chunk)

        for ent in doc.ents:
            # Only extract entity types we care about for contracts
            if ent.label_ not in CONTRACT_ENTITY_TYPES:
                continue

            # Get the full sentence containing this entity for provenance
            try:
                sent_text = ent.sent.text.strip()
            except Exception:
                sent_text = ""

            entity = {
                "text":       ent.text.strip(),
                "label":      ent.label_,
                "start_char": ent.start_char + char_offset,
                "end_char":   ent.end_char + char_offset,
                "sentence":   sent_text,
                "chunk_idx":  chunk_idx,
            }
            all_entities.append(entity)

        # Advance offset — subtract overlap to avoid double-counting positions
        if chunk_idx < len(chunks) - 1:
            char_offset += len(chunk) - 500

    # Deduplicate: remove entities with the same text+label that appear
    # in the overlap region of adjacent chunks
    seen     = set()
    deduped  = []
    for ent in all_entities:
        key = (ent["text"].lower(), ent["label"])
        if key not in seen:
            seen.add(key)
            deduped.append(ent)

    logger.info(f"🔍 NER complete | entities_found={len(deduped)}")
    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — ATTRIBUTE NORMALIZATION & CONFIDENCE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def normalize_date(date_str: str) -> Optional[str]:
    """
    Converts any date expression to ISO 8601 (YYYY-MM-DD).
    Handles: '15th January 2024', 'January 15, 2024', '01/15/24',
             'the fifteenth day of January, two thousand and twenty-four', etc.
    Returns None if unparseable.
    """
    try:
        parsed = dateparser.parse(
            date_str,
            settings={"PREFER_DAY_OF_MONTH": "first", "RETURN_AS_TIMEZONE_AWARE": False},
        )
        return parsed.strftime("%Y-%m-%d") if parsed else None
    except Exception:
        return None


def normalize_money(money_str: str) -> tuple[Optional[float], Optional[str]]:
    """
    Extracts numeric value and currency from money strings.
    Handles: '$1,500,000', 'USD 1.5M', '£250,000', 'EUR 500,000.00'
    Returns (amount_float, currency_code).
    """
    # Extract currency symbol / code
    currency_map = {
        "$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY",
        "USD": "USD", "GBP": "GBP", "EUR": "EUR", "INR": "INR",
    }
    currency = None
    for symbol, code in currency_map.items():
        if symbol in money_str:
            currency = code
            money_str = money_str.replace(symbol, "")
            break

    # Handle shorthand: 1.5M → 1500000, 500K → 500000
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    for suffix, mult in multipliers.items():
        if money_str.strip().upper().endswith(suffix):
            try:
                return float(money_str.strip()[:-1]) * mult, currency
            except ValueError:
                pass

    # Remove commas and parse
    cleaned = re.sub(r"[^\d.]", "", money_str)
    try:
        return float(cleaned), currency
    except ValueError:
        return None, currency


def compute_confidence_score(
    entity_text: str,
    entity_label: str,
    all_entities: List[Dict],
    normalized_value: Any,
) -> float:
    """
    Computes confidence score for a single extracted entity.
    Score is a weighted combination of:
      1. Corroboration bonus — same value found in multiple places
      2. Validation bonus   — value passes type validation (e.g., date parses correctly)
      3. Position signal    — entity found in high-signal locations (definitions, headers)
      4. Conflict penalty   — same entity type has conflicting values in the document

    Returns float 0.0 to 1.0.
    """
    base_score = 0.60  # Base confidence for any entity the model identifies

    # 1. Corroboration: count how many times the same value appears
    same_value_count = sum(
        1 for e in all_entities
        if e["label"] == entity_label and e["text"].lower() == entity_text.lower()
    )
    corroboration_bonus = min(0.15, (same_value_count - 1) * 0.075)

    # 2. Validation bonus: was normalization successful?
    validation_bonus = 0.15 if normalized_value is not None else 0.0

    # 3. Position signal: high-signal legal location keywords
    high_signal_keywords = [
        "definitions", "effective date", "term", "party", "parties",
        "governing law", "jurisdiction", "consideration", "whereas",
        "signed", "executed", "agreement dated",
    ]
    position_bonus = 0.0
    sentence_lower = ""
    for e in all_entities:
        if e["text"].lower() == entity_text.lower() and e["label"] == entity_label:
            sentence_lower = e.get("sentence", "").lower()
            break
    if any(kw in sentence_lower for kw in high_signal_keywords):
        position_bonus = 0.10

    # 4. Conflict penalty: multiple different values for the same entity type
    same_type_values = set(
        e["text"].lower() for e in all_entities if e["label"] == entity_label
    )
    conflict_penalty = -0.10 if len(same_type_values) > 2 else 0.0

    score = base_score + corroboration_bonus + validation_bonus + position_bonus + conflict_penalty
    return round(max(0.0, min(1.0, score)), 4)


def normalize_and_score_entities(entities: List[Dict]) -> Dict[str, Any]:
    """
    Step 7: Normalizes all extracted entities and computes confidence scores.
    Groups entities by type and selects the best candidate for each field.
    Returns a dict of normalized field values and their confidence scores.
    """
    # Group entities by label
    by_label: Dict[str, List[Dict]] = {}
    for ent in entities:
        by_label.setdefault(ent["label"], []).append(ent)

    fields    = {}
    scores    = {}

    # ── Organization (counterparty) ──────────────────────────────────────────
    if "ORG" in by_label:
        # Pick the ORG that appears most frequently — likely the counterparty
        org_counts = {}
        for e in by_label["ORG"]:
            key = e["text"].strip()
            org_counts[key] = org_counts.get(key, 0) + 1
        best_org = max(org_counts, key=org_counts.get)
        fields["counterparty_name"] = best_org
        scores["counterparty_name"] = compute_confidence_score(
            best_org, "ORG", entities, best_org
        )

    # ── Person (signatory) ───────────────────────────────────────────────────
    if "PERSON" in by_label:
        # Prefer PERSON entities found near signature-related keywords
        signatory = by_label["PERSON"][0]["text"]
        for e in by_label["PERSON"]:
            if any(kw in e.get("sentence", "").lower()
                   for kw in ["signed", "signatory", "authorized", "executed by"]):
                signatory = e["text"]
                break
        fields["signatory_name"] = signatory
        scores["signatory_name"] = compute_confidence_score(
            signatory, "PERSON", entities, signatory
        )

    # ── Dates ─────────────────────────────────────────────────────────────────
    date_field_hints = {
        "effective_date": ["effective", "commencement", "start date", "begins on", "dated"],
        "end_date":       ["end date", "expiration", "expires", "termination date"],
        "renewal_date":   ["renewal", "renew", "auto-renew", "extended"],
    }
    if "DATE" in by_label:
        for field, hints in date_field_hints.items():
            best_date = None
            best_score = 0.0
            for e in by_label["DATE"]:
                normalized = normalize_date(e["text"])
                if normalized is None:
                    continue
                conf = compute_confidence_score(e["text"], "DATE", entities, normalized)
                # Boost score if hint keyword in surrounding sentence
                if any(h in e.get("sentence", "").lower() for h in hints):
                    conf = min(1.0, conf + 0.12)
                if conf > best_score:
                    best_score = conf
                    best_date  = normalized
            if best_date:
                fields[field] = best_date
                scores[field] = best_score

    # ── Money (contract value) ────────────────────────────────────────────────
    if "MONEY" in by_label:
        # Find the highest-value money entity — most likely total contract value
        best_amount = None
        best_currency = None
        best_score  = 0.0
        for e in by_label["MONEY"]:
            amount, currency = normalize_money(e["text"])
            if amount is not None:
                conf = compute_confidence_score(e["text"], "MONEY", entities, amount)
                if amount > (best_amount or 0):
                    best_amount   = amount
                    best_currency = currency
                    best_score    = conf
        if best_amount is not None:
            fields["contract_value"] = best_amount
            fields["currency"]       = best_currency or "USD"
            scores["contract_value"] = best_score

    # ── Jurisdiction / Governing Law ──────────────────────────────────────────
    if "GPE" in by_label:
        for e in by_label["GPE"]:
            sent = e.get("sentence", "").lower()
            if any(kw in sent for kw in ["governing law", "jurisdiction", "governed by"]):
                fields["jurisdiction"] = e["text"]
                scores["jurisdiction"] = compute_confidence_score(
                    e["text"], "GPE", entities, e["text"]
                )
                break
        if "jurisdiction" not in fields and by_label["GPE"]:
            # Fallback: use any GPE with moderate confidence
            fields["jurisdiction"] = by_label["GPE"][0]["text"]
            scores["jurisdiction"] = 0.55

    return {"fields": fields, "confidence_scores": scores}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8 — RECORD ASSEMBLY & ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def assemble_and_enrich(
    fields: Dict[str, Any],
    scores: Dict[str, float],
) -> Dict[str, Any]:
    """
    Step 8: Assembles extracted fields into a complete record and
    computes derived attributes that are not explicitly stated in the document.

    Derived attributes:
      - contract_duration: calculated from effective_date and end_date
      - days_until_renewal: calculated from today and renewal_date
      - overall_confidence: weighted average of field confidence scores
    """
    record = dict(fields)

    # ── Derived: Contract Duration ────────────────────────────────────────────
    if "effective_date" in record and "end_date" in record:
        try:
            eff = datetime.date.fromisoformat(record["effective_date"])
            end = datetime.date.fromisoformat(record["end_date"])
            delta_days   = (end - eff).days
            delta_months = round(delta_days / 30.44)
            if delta_months >= 12:
                record["contract_duration"] = f"{delta_months // 12} year(s)"
            else:
                record["contract_duration"] = f"{delta_months} month(s)"
        except (ValueError, TypeError):
            pass

    # ── Derived: Days Until Renewal ───────────────────────────────────────────
    if "renewal_date" in record and record["renewal_date"]:
        try:
            renewal = datetime.date.fromisoformat(record["renewal_date"])
            days_until = (renewal - datetime.date.today()).days
            record["days_until_renewal"] = days_until
        except (ValueError, TypeError):
            pass

    # ── Overall Confidence Score (weighted average) ───────────────────────────
    # Weight critical fields more heavily
    field_weights = {
        "counterparty_name": 3.0,
        "effective_date":    2.5,
        "end_date":          2.0,
        "contract_value":    2.0,
        "jurisdiction":      1.5,
        "signatory_name":    1.0,
        "renewal_date":      1.0,
    }
    weighted_sum = 0.0
    total_weight = 0.0
    for field, weight in field_weights.items():
        if field in scores:
            weighted_sum += scores[field] * weight
            total_weight += weight

    overall = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0
    record["overall_confidence"] = overall

    return record


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 9 — STAGING DATABASE + QUALITY / BUSINESS RULES
# ─────────────────────────────────────────────────────────────────────────────

def apply_business_rules(record: Dict[str, Any]) -> List[Dict]:
    """
    Step 9: Applies quality and business rules to each extracted record.
    Returns a list of rule flags with severity and message.

    Hard rules: record CANNOT auto-approve — must go to HITL regardless of confidence.
    Soft rules: advisory flags shown to reviewer but can be overridden.
    """
    flags = []

    # ── HARD RULES ────────────────────────────────────────────────────────────

    # Every contract must have a counterparty name
    if not record.get("counterparty_name"):
        flags.append({
            "rule":     "MISSING_COUNTERPARTY",
            "severity": "HARD",
            "message":  "No counterparty organization name could be extracted.",
        })

    # Every contract must have an effective date
    if not record.get("effective_date"):
        flags.append({
            "rule":     "MISSING_EFFECTIVE_DATE",
            "severity": "HARD",
            "message":  "No effective date could be extracted.",
        })

    # Effective date must precede end date (logical consistency)
    if record.get("effective_date") and record.get("end_date"):
        try:
            eff = datetime.date.fromisoformat(record["effective_date"])
            end = datetime.date.fromisoformat(record["end_date"])
            if eff >= end:
                flags.append({
                    "rule":     "DATE_LOGIC_ERROR",
                    "severity": "HARD",
                    "message":  f"Effective date ({record['effective_date']}) is not before end date ({record['end_date']}).",
                })
        except ValueError:
            pass

    # ── SOFT RULES ────────────────────────────────────────────────────────────

    # Contract value should be positive if present
    if record.get("contract_value") is not None and record["contract_value"] <= 0:
        flags.append({
            "rule":     "INVALID_CONTRACT_VALUE",
            "severity": "SOFT",
            "message":  f"Contract value is zero or negative: {record['contract_value']}.",
        })

    # Effective date should not be more than 10 years in the past
    if record.get("effective_date"):
        try:
            eff = datetime.date.fromisoformat(record["effective_date"])
            years_ago = (datetime.date.today() - eff).days / 365
            if years_ago > 10:
                flags.append({
                    "rule":     "VERY_OLD_CONTRACT",
                    "severity": "SOFT",
                    "message":  f"Effective date is more than 10 years ago: {record['effective_date']}.",
                })
        except ValueError:
            pass

    # Renewal date should be after effective date if both present
    if record.get("renewal_date") and record.get("effective_date"):
        try:
            eff     = datetime.date.fromisoformat(record["effective_date"])
            renewal = datetime.date.fromisoformat(record["renewal_date"])
            if renewal <= eff:
                flags.append({
                    "rule":     "RENEWAL_BEFORE_EFFECTIVE",
                    "severity": "SOFT",
                    "message":  "Renewal date is before or equal to effective date.",
                })
        except ValueError:
            pass

    # Low overall confidence warning
    if record.get("overall_confidence", 0) < 0.5:
        flags.append({
            "rule":     "LOW_CONFIDENCE",
            "severity": "SOFT",
            "message":  f"Overall extraction confidence is low: {record['overall_confidence']:.2%}.",
        })

    return flags


def persist_to_staging(
    file_id: str,
    client_id: str,
    record: Dict,
    confidence_scores: Dict,
    raw_entities: List[Dict],
    rule_flags: List[Dict],
) -> StagingRecord:
    """
    Step 9: Writes the assembled record to the Staging Database.
    Determines initial review_status based on confidence + hard rules.
    """
    has_hard_failure = any(f["severity"] == "HARD" for f in rule_flags)

    # Auto-approve if: above threshold AND no hard rule failures
    if record["overall_confidence"] >= AUTO_APPROVE_THRESHOLD and not has_hard_failure:
        review_status = "AUTO_APPROVED"
    else:
        review_status = "FLAGGED"

    staging_record = StagingRecord(
        record_id           = str(uuid.uuid4()),
        file_id             = file_id,
        client_id           = client_id,
        counterparty_name   = record.get("counterparty_name"),
        party_role          = record.get("party_role"),
        effective_date      = record.get("effective_date"),
        end_date            = record.get("end_date"),
        renewal_date        = record.get("renewal_date"),
        contract_value      = record.get("contract_value"),
        currency            = record.get("currency"),
        jurisdiction        = record.get("jurisdiction"),
        contract_duration   = record.get("contract_duration"),
        days_until_renewal  = record.get("days_until_renewal"),
        signatory_name      = record.get("signatory_name"),
        contract_id_ref     = record.get("contract_id_ref"),
        confidence_scores   = confidence_scores,
        overall_confidence  = record.get("overall_confidence", 0.0),
        raw_entities_json   = raw_entities,
        rule_flags          = rule_flags,
        has_hard_failure    = has_hard_failure,
        review_status       = review_status,
    )

    with svc.StagingSession() as db:
        db.add(staging_record)
        db.commit()
        db.refresh(staging_record)

    logger.info(
        f"💾 Staged | record_id={staging_record.record_id} | "
        f"status={review_status} | confidence={record['overall_confidence']:.2%}"
    )
    return staging_record


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(file_id: str, pdf_bytes: bytes, client_id: str):
    """
    Orchestrates all pipeline stages for a single document.
    Steps 3 → 4 → 6 → 7 → 8 → 9 → 10/11

    Called as a background task after file ingestion.
    Updates IngestedFile processing_status throughout.
    """
    def update_status(status: str, error: str = None):
        with svc.StagingSession() as db:
            f = db.query(IngestedFile).filter_by(file_id=file_id).first()
            if f:
                f.processing_status = status
                if error:
                    f.error_message = error
                db.commit()

    update_status("PROCESSING")
    logger.info(f"⚙️  Pipeline started | file_id={file_id}")

    try:
        # Step 3: PDF text extraction
        raw_text, method = preprocess_pdf(pdf_bytes)
        if not raw_text.strip():
            raise ValueError("No text extracted — document may be blank or corrupted.")

        # Step 4: Text normalization (PRESERVE CASE)
        normalized_text = normalize_text(raw_text)

        # Step 6: NER — extract entities
        entities = run_ner(normalized_text)
        if not entities:
            logger.warning(f"⚠️  No entities found in file_id={file_id}")

        # Step 7: Attribute normalization + confidence scoring
        result  = normalize_and_score_entities(entities)
        fields  = result["fields"]
        scores  = result["confidence_scores"]

        # Step 8: Record assembly + enrichment (derived fields)
        record  = assemble_and_enrich(fields, scores)

        # Step 9: Business rules + staging persistence
        flags   = apply_business_rules(record)
        staging = persist_to_staging(
            file_id, client_id, record, scores, entities, flags
        )

        # Step 10: Auto-approve → sync to production immediately
        if staging.review_status == "AUTO_APPROVED":
            sync_to_production(staging.record_id, approval_type="AUTO")
            logger.info(f"✅ Auto-approved and synced | record_id={staging.record_id}")
        else:
            logger.info(f"🚩 Flagged for HITL review | record_id={staging.record_id}")

        update_status("DONE")

    except Exception as e:
        logger.error(f"❌ Pipeline failed | file_id={file_id} | error={e}", exc_info=True)
        update_status("FAILED", error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 10 — SYNC TO PRODUCTION (Auto-approved path)
# ─────────────────────────────────────────────────────────────────────────────

def sync_to_production(record_id: str, approval_type: str, reviewer_id: str = None):
    """
    Steps 10 / 13: Syncs an approved staging record to the Client DB Table.
    Called after AUTO_APPROVED (step 10) or human APPROVED (step 13).

    ONLY records with review_status in ('AUTO_APPROVED', 'APPROVED') reach here.
    This is the gate — no staging record bypasses this check.
    """
    with svc.StagingSession() as staging_db:
        rec = staging_db.query(StagingRecord).filter_by(record_id=record_id).first()
        if not rec:
            logger.error(f"❌ Staging record not found: {record_id}")
            return
        if rec.review_status not in ("AUTO_APPROVED", "APPROVED"):
            logger.error(f"❌ Cannot sync non-approved record: {record_id} | status={rec.review_status}")
            return

        client_record = ClientContractRecord(
            contract_record_id = str(uuid.uuid4()),
            staging_record_id  = rec.record_id,
            file_id            = rec.file_id,
            client_id          = rec.client_id,
            counterparty_name  = rec.counterparty_name,
            party_role         = rec.party_role,
            effective_date     = rec.effective_date,
            end_date           = rec.end_date,
            renewal_date       = rec.renewal_date,
            contract_value     = rec.contract_value,
            currency           = rec.currency,
            jurisdiction       = rec.jurisdiction,
            contract_duration  = rec.contract_duration,
            days_until_renewal = rec.days_until_renewal,
            signatory_name     = rec.signatory_name,
            contract_id_ref    = rec.contract_id_ref,
            approval_type      = approval_type,
            reviewer_id        = reviewer_id,
        )

    with svc.ClientSession() as client_db:
        client_db.add(client_record)
        client_db.commit()

    logger.info(f"🎯 Synced to Client DB | record_id={record_id} | approval={approval_type}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 (HITL) — HUMAN-IN-THE-LOOP REVIEW WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/review/flagged", tags=["6. HITL Review"])
async def get_flagged_records(
    client_id: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    token: str = Depends(oauth2_scheme),
):
    """
    Step 11: Returns records flagged for human review, sorted by file ingestion time.
    Feeds the UI Review App (Gradio / internal dashboard).

    Each record shows:
      - Extracted field values with confidence scores
      - Rule flags explaining why it was flagged
      - Raw entities for provenance (which sentence each value came from)
    """
    with svc.StagingSession() as db:
        query = db.query(StagingRecord).filter(
            StagingRecord.review_status == "FLAGGED"
        )
        if client_id:
            query = query.filter(StagingRecord.client_id == client_id)
        records = query.order_by(StagingRecord.created_at.asc()).limit(limit).all()

    result = []
    for rec in records:
        result.append({
            "record_id":          rec.record_id,
            "file_id":            rec.file_id,
            "client_id":          rec.client_id,
            "counterparty_name":  rec.counterparty_name,
            "effective_date":     rec.effective_date,
            "end_date":           rec.end_date,
            "renewal_date":       rec.renewal_date,
            "contract_value":     rec.contract_value,
            "currency":           rec.currency,
            "jurisdiction":       rec.jurisdiction,
            "contract_duration":  rec.contract_duration,
            "days_until_renewal": rec.days_until_renewal,
            "signatory_name":     rec.signatory_name,
            "overall_confidence": rec.overall_confidence,
            "confidence_scores":  rec.confidence_scores,
            "rule_flags":         rec.rule_flags,
            "has_hard_failure":   rec.has_hard_failure,
            # Provenance: show reviewer the sentence each entity came from
            "entity_provenance": [
                {"field": e["label"], "text": e["text"], "sentence": e["sentence"]}
                for e in (rec.raw_entities_json or [])
            ],
        })

    return {"flagged_count": len(result), "records": result}


@app.post("/review/decide", tags=["6. HITL Review"])
async def submit_review_decision(
    decision: ReviewDecision,
    token: str = Depends(oauth2_scheme),
):
    """
    Step 12: Reviewer submits APPROVE or REJECT decision.
    - Corrections to extracted fields are applied before approval
    - Writes immutable audit log entry (Who, When, What)
    - If APPROVED: updates staging and syncs to Client DB (Step 13)
    - If REJECTED: marks record as REJECTED with reason logged
    """
    with svc.StagingSession() as db:
        rec = db.query(StagingRecord).filter_by(record_id=decision.record_id).first()
        if not rec:
            raise HTTPException(status_code=404, detail=f"Record {decision.record_id} not found.")
        if rec.review_status not in ("FLAGGED", "PENDING"):
            raise HTTPException(
                status_code=409,
                detail=f"Record already reviewed: {rec.review_status}",
            )

        # Snapshot before state for audit
        before_state = {
            "counterparty_name": rec.counterparty_name,
            "effective_date":    rec.effective_date,
            "contract_value":    rec.contract_value,
            "review_status":     rec.review_status,
        }

        # Apply reviewer corrections if provided
        if decision.corrected_counterparty_name:
            rec.counterparty_name = decision.corrected_counterparty_name
        if decision.corrected_effective_date:
            rec.effective_date = decision.corrected_effective_date
        if decision.corrected_end_date:
            rec.end_date = decision.corrected_end_date
        if decision.corrected_contract_value:
            rec.contract_value = decision.corrected_contract_value
        if decision.corrected_jurisdiction:
            rec.jurisdiction = decision.corrected_jurisdiction

        # Set review outcome
        rec.review_status     = "APPROVED" if decision.decision == "APPROVE" else "REJECTED"
        rec.reviewer_id       = decision.reviewer_id
        rec.reviewer_decision = decision.decision
        rec.reviewer_comments = decision.comments
        rec.reviewed_at       = datetime.datetime.utcnow()

        after_state = {
            "counterparty_name": rec.counterparty_name,
            "effective_date":    rec.effective_date,
            "contract_value":    rec.contract_value,
            "review_status":     rec.review_status,
        }

        # Write immutable audit log — INSERT only, never updated
        audit = AuditLog(
            record_id    = decision.record_id,
            reviewer_id  = decision.reviewer_id,
            action       = decision.decision,
            before_state = before_state,
            after_state  = after_state,
            comments     = decision.comments,
        )
        db.add(audit)
        db.commit()

    # Step 13: Sync approved record to Client DB
    if decision.decision == "APPROVE":
        sync_to_production(
            decision.record_id,
            approval_type="HUMAN",
            reviewer_id=decision.reviewer_id,
        )
        message = "Record approved and synced to Client DB."
    else:
        message = "Record rejected. Logged with reason."

    logger.info(
        f"⚖️  Review submitted | record_id={decision.record_id} | "
        f"decision={decision.decision} | reviewer={decision.reviewer_id}"
    )

    return {
        "record_id":  decision.record_id,
        "decision":   decision.decision,
        "status":     rec.review_status,
        "message":    message,
        "reviewed_at": rec.reviewed_at.isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DELIVERY — CLIENT DB QUERY API
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/client/contracts", tags=["5. Delivery"])
async def get_client_contracts(
    client_id: str,
    renewal_within_days: Optional[int] = None,
    min_contract_value: Optional[float] = None,
    token: str = Depends(oauth2_scheme),
):
    """
    Queries the final Client DB Table — the production-ready structured contracts.
    Supports filtering by upcoming renewal and minimum contract value.

    This is the business value endpoint — the filing cabinet turned into a database.
    'Give me all contracts expiring in the next 60 days' → one API call, instant answer.
    """
    with svc.ClientSession() as db:
        query = db.query(ClientContractRecord).filter_by(client_id=client_id)

        if renewal_within_days is not None:
            cutoff_date = (
                datetime.date.today() + datetime.timedelta(days=renewal_within_days)
            ).isoformat()
            query = query.filter(
                ClientContractRecord.renewal_date <= cutoff_date,
                ClientContractRecord.renewal_date >= datetime.date.today().isoformat(),
            )

        if min_contract_value is not None:
            query = query.filter(
                ClientContractRecord.contract_value >= min_contract_value
            )

        records = query.order_by(ClientContractRecord.synced_at.desc()).all()

    result = [
        {
            "contract_record_id": r.contract_record_id,
            "counterparty_name":  r.counterparty_name,
            "effective_date":     r.effective_date,
            "end_date":           r.end_date,
            "renewal_date":       r.renewal_date,
            "contract_value":     r.contract_value,
            "currency":           r.currency,
            "jurisdiction":       r.jurisdiction,
            "contract_duration":  r.contract_duration,
            "days_until_renewal": r.days_until_renewal,
            "signatory_name":     r.signatory_name,
            "approval_type":      r.approval_type,
            "synced_at":          r.synced_at.isoformat(),
        }
        for r in records
    ]

    return {
        "client_id":      client_id,
        "total_contracts": len(result),
        "contracts":      result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MONITORING & HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe — confirms API is running and model is loaded."""
    return {
        "status":        "ok",
        "spacy_model":   SPACY_MODEL,
        "model_loaded":  svc.nlp is not None,
        "timestamp":     datetime.datetime.utcnow().isoformat(),
    }


@app.get("/monitoring/pipeline-stats", tags=["Monitoring"])
async def pipeline_stats(token: str = Depends(oauth2_scheme)):
    """
    Returns pipeline throughput stats:
    - Total files ingested by status
    - Total records by review status
    - Auto-approval rate (key metric for confidence threshold tuning)
    - Average confidence scores
    """
    with svc.StagingSession() as db:
        file_stats = {}
        for status in ["PENDING", "PROCESSING", "DONE", "FAILED"]:
            file_stats[status.lower()] = (
                db.query(IngestedFile).filter_by(processing_status=status).count()
            )

        record_stats = {}
        for status in ["PENDING", "FLAGGED", "AUTO_APPROVED", "APPROVED", "REJECTED"]:
            record_stats[status.lower()] = (
                db.query(StagingRecord).filter_by(review_status=status).count()
            )

        total_decided = (
            record_stats["auto_approved"] +
            record_stats["approved"] +
            record_stats["rejected"]
        )
        auto_approve_rate = (
            record_stats["auto_approved"] / total_decided if total_decided > 0 else 0.0
        )

        # Average confidence across all records
        all_records = db.query(StagingRecord.overall_confidence).all()
        avg_confidence = (
            sum(r[0] for r in all_records) / len(all_records) if all_records else 0.0
        )

    return {
        "files":             file_stats,
        "records":           record_stats,
        "auto_approve_rate": round(auto_approve_rate, 4),
        "avg_confidence":    round(avg_confidence, 4),
        "auto_approve_threshold": AUTO_APPROVE_THRESHOLD,
    }


@app.get("/audit/log", tags=["Audit"])
async def get_audit_log(
    record_id: Optional[str] = None,
    reviewer_id: Optional[str] = None,
    limit: int = Query(default=100, le=500),
    token: str = Depends(oauth2_scheme),
):
    """
    Returns the immutable audit log.
    Filterable by record_id and reviewer_id.
    Required for compliance audits and dispute resolution.
    """
    with svc.StagingSession() as db:
        query = db.query(AuditLog)
        if record_id:
            query = query.filter_by(record_id=record_id)
        if reviewer_id:
            query = query.filter_by(reviewer_id=reviewer_id)
        entries = query.order_by(AuditLog.created_at.desc()).limit(limit).all()

    return {
        "count": len(entries),
        "entries": [
            {
                "audit_id":    e.audit_id,
                "record_id":   e.record_id,
                "reviewer_id": e.reviewer_id,
                "action":      e.action,
                "before":      e.before_state,
                "after":       e.after_state,
                "comments":    e.comments,
                "created_at":  e.created_at.isoformat(),
            }
            for e in entries
        ],
    }

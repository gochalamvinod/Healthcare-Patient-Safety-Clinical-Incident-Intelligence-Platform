"""
FastAPI Backend Application for Healthcare Patient Safety & Clinical Incident Intelligence Platform.
Exposes REST APIs for Clinical Triage, Deterministic ML, RAG Precedents, CAPA Mapping, and Gemini GenAI.
"""

import os
import urllib.request
from pathlib import Path
from typing import Optional, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

from backend.ml_engine import get_pipeline

app = FastAPI(
    title="Healthcare Patient Safety & Clinical Incident Intelligence Platform",
    description="Clinical Incident Intelligence, Deterministic ML, RAG Precedents, NABH/WHO CAPA & Gemini GenAI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"
CASE_STUDY_FILE = BASE_DIR / "case_study.html"

class Query(BaseModel):
    text: str = Field(..., min_length=3)
    apply_corrections: Optional[Any] = True
    department: Optional[str] = "Inpatient Ward"
    site_type: Optional[str] = None  # Compatibility alias

class IncidentInput(BaseModel):
    narrative: str = Field(..., min_length=3)
    department: Optional[str] = "Inpatient Ward"
    site_type: Optional[str] = None  # Compatibility alias
    apply_corrections: Optional[Any] = True

def _parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    return True

def _get_html(file_path: Path) -> str:
    frontend_url = os.environ.get("FRONTEND_URL")
    if frontend_url:
        try:
            req = urllib.request.Request(frontend_url, headers={"User-Agent": "FastAPI-Healthcare-Gateway"})
            with urllib.request.urlopen(req, timeout=3) as res:
                return res.read().decode("utf-8")
        except Exception:
            pass
    if file_path.exists():
        return file_path.read_text(encoding="utf-8")
    for alt in [BASE_DIR / "frontend" / "dist" / "index.html", BASE_DIR / "index.html", FRONTEND_FILE]:
        if alt.exists():
            return alt.read_text(encoding="utf-8")
    return "<html><body><h2>Healthcare Patient Safety & Clinical Incident Intelligence Platform</h2></body></html>"

# 1. Web Interface Routes (Vercel & Local Native Support)
@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
@app.get("/api", response_class=HTMLResponse, tags=["Web Interface"])
@app.get("/api/", response_class=HTMLResponse, tags=["Web Interface"])
@app.get("/api/index", response_class=HTMLResponse, tags=["Web Interface"])
@app.get("/api/index.py", response_class=HTMLResponse, tags=["Web Interface"])
@app.get("/index.html", response_class=HTMLResponse, tags=["Web Interface"])
def serve_dashboard():
    return HTMLResponse(content=_get_html(FRONTEND_FILE), status_code=200)

@app.get("/case-study", response_class=HTMLResponse, tags=["Web Interface"])
def serve_case_study():
    return HTMLResponse(content=_get_html(CASE_STUDY_FILE), status_code=200)

# 2. Case Study Assessment API Endpoints
@app.get("/api/demo", tags=["Case Study Starter API"])
def demo():
    pipe = get_pipeline()
    an = pipe.get_dashboard_analytics()
    return {
        "case_study": "Healthcare Patient Safety & Clinical Incident Intelligence Platform",
        "status": "production ready",
        "dataset_records_count": an["total_incidents"],
        "model_performance": an["model_health"]
    }

@app.post("/api/run", tags=["Case Study Starter API"])
def run_case(query: Query):
    pipe = get_pipeline()
    use_corr = _parse_bool(query.apply_corrections)
    dept = query.department or query.site_type or "Inpatient Ward"
    analysis = pipe.classify_and_analyze(
        query.text,
        department=dept,
        apply_corrections=use_corr
    )
    m = pipe.evaluation_results
    return {
        "input": query.text,
        "result": {
            "predicted_category": analysis["prediction"]["category"],
            "predicted_severity": analysis["prediction"]["severity"],
            "confidence": analysis["prediction"]["category_confidence"],
            "composite_risk_score": analysis["prediction"]["composite_risk_score"],
            "summary": analysis["nlp_insights"]["extractive_summary"],
            "preventive_actions": analysis["recommended_preventive_actions"],
            "corrected_narrative": analysis["nlp_insights"].get("corrected_narrative"),
            "typo_corrections": analysis["nlp_insights"].get("typo_corrections", []),
            "corrections_applied": analysis.get("spelling_review", {}).get("corrections_applied", True)
        },
        "predictions": analysis["prediction"],
        "spelling_review": analysis.get("spelling_review"),
        "nlp_insights": analysis["nlp_insights"],
        "retrieval_evidence": analysis["retrieval_evidence"],
        "gemini_insights": analysis.get("gemini_insights"),
        "evaluation_metrics": {
            "category_accuracy": m.get("category_model", {}).get("accuracy"),
            "category_f1": m.get("category_model", {}).get("f1_weighted"),
            "severity_accuracy": m.get("severity_model", {}).get("accuracy"),
            "severity_f1": m.get("severity_model", {}).get("f1_weighted"),
            "baseline_gain": m.get("category_model", {}).get("gain")
        }
    }

@app.post("/api/classify", tags=["Clinical Incident Intelligence"])
def classify_narrative(payload: IncidentInput):
    use_corr = _parse_bool(payload.apply_corrections)
    dept = payload.department or payload.site_type or "Inpatient Ward"
    return get_pipeline().classify_and_analyze(
        payload.narrative,
        department=dept,
        apply_corrections=use_corr
    )

@app.post("/api/spellcheck", tags=["Linguistic Normalization"])
def check_spelling(query: Query):
    return get_pipeline().check_spelling_with_gemini(query.text)

@app.get("/api/incidents", tags=["Patient Safety Incident Register"])
def list_incidents(
    category: Optional[str] = None,
    severity: Optional[str] = None,
    department: Optional[str] = None,
    search: Optional[str] = None
):
    pipe = get_pipeline()
    df = pipe.df.copy()
    if category:
        df = df[df["target_category"].str.lower() == category.lower()]
    if severity:
        df = df[df["target_severity"].str.lower() == severity.lower()]
    if department:
        df = df[df["department"].str.lower() == department.lower()]
    if search:
        df = df[df["narrative"].str.contains(search, case=False, na=False)]
    return {"count": len(df), "incidents": df.to_dict(orient="records")}

@app.get("/api/analytics", tags=["Dashboard Analytics"])
def get_analytics():
    return get_pipeline().get_dashboard_analytics()

@app.get("/api/clusters", tags=["Clinical Risk Clusters"])
def get_clusters():
    return {"clusters": get_pipeline().cluster_info}

@app.get("/api/metrics", tags=["Healthcare Quality Metrics"])
def get_metrics():
    return get_pipeline().evaluation_results

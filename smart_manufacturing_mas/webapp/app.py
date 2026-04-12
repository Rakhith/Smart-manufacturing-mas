from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.run_manager import ROOT_DIR, RunConfig, WEB_UPLOAD_DIR, run_manager


APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

app = FastAPI(title="Smart Manufacturing MAS Local App", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/api/health")
async def healthcheck() -> dict:
    return {"status": "ok"}


def _allowed_file_path(path_str: str) -> Path:
    candidate = (ROOT_DIR / path_str).resolve() if not Path(path_str).is_absolute() else Path(path_str).resolve()
    if ROOT_DIR not in candidate.parents and candidate != ROOT_DIR:
        raise HTTPException(status_code=400, detail="Path must stay inside the project workspace.")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Requested file does not exist.")
    return candidate


def _discover_datasets() -> list[dict]:
    datasets = []
    for path in sorted(DATA_DIR.rglob("*")):
        if path.suffix.lower() not in {".csv", ".npz"}:
            continue
        rel = path.relative_to(ROOT_DIR)
        datasets.append(
            {
                "label": path.name,
                "path": str(rel),
                "directory": str(path.parent.relative_to(ROOT_DIR)),
            }
        )
    return datasets


def _dataset_profile(path: Path) -> dict:
    df = pd.read_csv(path, nrows=20) if path.suffix.lower() == ".csv" else None
    if df is None:
        return {"columns": [], "preview_rows": [], "shape_estimate": None}
    return {
        "columns": df.columns.tolist(),
        "preview_rows": df.head(5).replace({pd.NA: None}).to_dict(orient="records"),
        "shape_estimate": [int(df.shape[0]), int(df.shape[1])],
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "datasets": _discover_datasets(),
        },
    )


@app.get("/api/datasets")
async def list_datasets() -> list[dict]:
    return _discover_datasets()


@app.get("/api/datasets/preview")
async def preview_dataset(path: str) -> dict:
    dataset_path = _allowed_file_path(path)
    return {
        "path": str(dataset_path.relative_to(ROOT_DIR)),
        **_dataset_profile(dataset_path),
    }


@app.get("/api/runs")
async def list_runs() -> list[dict]:
    return run_manager.list_runs()


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> JSONResponse:
    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return JSONResponse(run)


@app.post("/api/runs")
async def create_run(
    dataset_path: Optional[str] = Form(default=None),
    dataset_label: Optional[str] = Form(default=None),
    target_column: Optional[str] = Form(default=None),
    problem_type: Optional[str] = Form(default=None),
    feature_columns: Optional[str] = Form(default=None),
    use_pca: bool = Form(default=False),
    use_cache: bool = Form(default=True),
    train_mode: str = Form(default="pretrained"),
    generate_synthetic: bool = Form(default=False),
    synthetic_rows: int = Form(default=300),
    preferred_model: Optional[str] = Form(default=None),
    upload: Optional[UploadFile] = File(default=None),
) -> dict:
    selected_path: Optional[Path] = None
    selected_label = dataset_label

    if synthetic_rows < 10 or synthetic_rows > 10000:
        raise HTTPException(status_code=400, detail="Synthetic rows must be between 10 and 10000.")
    if train_mode not in {"pretrained", "live"}:
        raise HTTPException(status_code=400, detail="Unsupported train mode.")

    if upload and upload.filename:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in {".csv", ".npz"}:
            raise HTTPException(status_code=400, detail="Only CSV and NPZ uploads are supported.")
        safe_name = f"{Path(upload.filename).stem}_{uuid_safe()}{Path(upload.filename).suffix}"
        selected_path = WEB_UPLOAD_DIR / safe_name
        content = await upload.read()
        selected_path.write_bytes(content)
        selected_label = upload.filename
    elif dataset_path:
        selected_path = _allowed_file_path(dataset_path)
        selected_label = selected_label or selected_path.name

    if selected_path is None:
        raise HTTPException(status_code=400, detail="Choose an existing dataset or upload a CSV file.")

    config = RunConfig(
        dataset_path=str(selected_path),
        dataset_label=selected_label or selected_path.name,
        feature_columns=[col.strip() for col in feature_columns.split(",") if col.strip()] if feature_columns else None,
        target_column=target_column or None,
        problem_type=problem_type or None,
        use_pca=use_pca,
        use_cache=use_cache,
        train_mode=train_mode,
        generate_synthetic=generate_synthetic,
        synthetic_rows=synthetic_rows,
        preferred_model=preferred_model or None,
    )
    run_id = run_manager.create_run(config)
    return {"run_id": run_id}


@app.get("/api/files")
async def fetch_file(path: str) -> FileResponse:
    file_path = _allowed_file_path(path)
    return FileResponse(file_path)


def uuid_safe() -> str:
    import uuid

    return uuid.uuid4().hex[:8]

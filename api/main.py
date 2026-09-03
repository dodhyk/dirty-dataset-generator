from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    PresetsResponse,
)
from api.storage import get_generation_file, save_generation_result
from core.generator import generate
from core.presets import get_available_presets

app = FastAPI(
    title="Dirty Dataset Generator API",
    description="Backend API for generating synthetic clean and dirty tabular datasets with ground truth corruption reports.",
    version="1.0.0",
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root():
    """Root endpoint providing service information and links to documentation and endpoints."""
    return {
        "name": "Dirty Dataset Generator API",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "presets": "/presets",
            "generate": "POST /generate",
            "download_clean": "/download/{generation_id}/clean",
            "download_dirty": "/download/{generation_id}/dirty",
            "download_report": "/download/{generation_id}/report",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint to ensure API service is operational."""
    return HealthResponse(status="ok")


@app.get("/presets", response_model=PresetsResponse, tags=["Presets"])
def list_presets():
    """List all available dataset presets."""
    return PresetsResponse(presets=get_available_presets())


@app.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
    tags=["Generator"],
)
def generate_dataset(payload: GenerateRequest):
    """Generate clean and dirty synthetic datasets according to user configuration."""
    available = get_available_presets()
    if payload.preset not in available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown preset '{payload.preset}'. Available presets: {available}",
        )

    try:
        result = generate(
            preset=payload.preset,
            rows=payload.rows,
            dirtiness=payload.dirtiness,
            seed=payload.seed,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset generation failed: {str(e)}",
        )

    gen_id, paths = save_generation_result(result)

    return GenerateResponse(
        generation_id=gen_id,
        preset=result.preset,
        rows=payload.rows,
        dirtiness=payload.dirtiness,
        dirty_rows=result.report.dirty_rows,
        seed=result.report.seed,
        total_corruptions=result.report.total_corruptions,
        clean_download_url=f"/download/{gen_id}/clean",
        dirty_download_url=f"/download/{gen_id}/dirty",
        report_download_url=f"/download/{gen_id}/report",
    )


@app.get("/download/{generation_id}/clean", tags=["Download"])
def download_clean_dataset(generation_id: str):
    """Download the generated clean dataset as CSV."""
    try:
        file_path = get_generation_file(generation_id, "clean")
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type="text/csv",
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fe))
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))


@app.get("/download/{generation_id}/dirty", tags=["Download"])
def download_dirty_dataset(generation_id: str):
    """Download the generated dirty dataset as CSV."""
    try:
        file_path = get_generation_file(generation_id, "dirty")
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type="text/csv",
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fe))
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))


@app.get("/download/{generation_id}/report", tags=["Download"])
def download_corruption_report(generation_id: str):
    """Download the ground truth corruption report as JSON."""
    try:
        file_path = get_generation_file(generation_id, "report")
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type="application/json",
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fe))
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))

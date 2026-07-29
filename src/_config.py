"""Shared configuration for paper1 (izmir-urban-heat-exposure).

Centralises paths, CRS, naming conventions, and logging so that every analysis
and figure script reads from one source of truth.  No hard-coded user paths,
private asset IDs, or machine-specific strings belong here — those live in
untracked environment variables (see ``.env.example``).

Import from analysis and figure scripts as::

    from _config import (
        REPO_ROOT, MAIN_CRS, GEO_CRS,
        DATA_EXTERNAL, DATA_RAW, DATA_INTERIM, DATA_PROCESSED, DATA_MODELS,
        FIGURES_DIR, TABLES_DIR, LOGS_DIR, FIGURES_PNG, PARAMS_DIR,
        GRID_RESOLUTION, RANDOM_SEED,
        get_logger, ensure_dirs, require_file,
    )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root — resolves from this file's location
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------
MAIN_CRS = "EPSG:32635"          # UTM Zone 35N — analysis and display
GEO_CRS  = "EPSG:4326"           # WGS 84 — GEE export and lat / lon

# ---------------------------------------------------------------------------
# Google Earth Engine (defaults; override via environment)
# ---------------------------------------------------------------------------
GEE_PROJECT        = os.getenv("EE_PROJECT", "ee-geophilo")
GEE_DRIVE_FOLDER   = os.getenv("GEE_DRIVE_FOLDER", "bbtmk_paper1")
GEE_DOWNLOAD_DIR   = os.getenv("GEE_DOWNLOAD_DIR", None)

# ---------------------------------------------------------------------------
# Data staging — all five levels are git-ignored and must be populated locally
# ---------------------------------------------------------------------------
DATA_EXTERNAL   = REPO_ROOT / "data" / "00_external"
DATA_RAW        = REPO_ROOT / "data" / "01_raw"        / "paper1"
DATA_INTERIM    = REPO_ROOT / "data" / "02_interim"    / "paper1"
DATA_PROCESSED  = REPO_ROOT / "data" / "03_processed"  / "paper1"
DATA_MODELS     = REPO_ROOT / "data" / "04_models"     / "paper1"

# External data subdirectories
EXTERNAL_MUNICIPAL  = DATA_EXTERNAL / "akilli_sehir_db"
EXTERNAL_ELEVATION  = DATA_EXTERNAL / "elevation_surface"
EXTERNAL_CENSUS     = DATA_EXTERNAL / "census"

PARAMS_DIR      = REPO_ROOT / "data" / "model_params"

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
OUTPUTS_DIR     = REPO_ROOT / "outputs"
FIGURES_DIR     = REPO_ROOT / "outputs" / "figures"   / "paper1"
TABLES_DIR      = REPO_ROOT / "outputs" / "tables"    / "paper1"
LOGS_DIR        = REPO_ROOT / "outputs" / "logs"

# ---------------------------------------------------------------------------
# Repository figures (author-generated, versioned)
# ---------------------------------------------------------------------------
FIGURES_PNG     = REPO_ROOT / "figures" / "png"

# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------
GRID_RESOLUTION = 100            # analysis grid cell size (metres)
RANDOM_SEED     = 42             # fixed seed for all stochastic components

# ---------------------------------------------------------------------------
# Convenience aliases for interim subdirectories
# ---------------------------------------------------------------------------
INTERIM_ANNUAL_LST   = DATA_INTERIM / "annual_lst_2014_2024"
PROCESSED_ALIGNED    = DATA_PROCESSED / "aligned_covariates"
PROCESSED_DATACUBES  = DATA_PROCESSED / "datacubes"
BOUNDARY_GPKG        = DATA_INTERIM / "izmir_study_boundary.gpkg"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Return a logger writing to console and ``LOGS_DIR / <name>.log``."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        log_file or LOGS_DIR / f"{name}.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def ensure_dirs(*dirs: Path) -> None:
    """Create each directory (and parents) if it does not exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def require_file(path: Path, description: str = "") -> Path:
    """Return *path* if it exists, otherwise raise ``FileNotFoundError``."""
    if not path.exists():
        msg = f"Required file not found: {path}"
        if description:
            msg += f" ({description})"
        raise FileNotFoundError(msg)
    return path

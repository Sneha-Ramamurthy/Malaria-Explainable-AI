"""Shared paths/constants for the Malaria-Risk XAI project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"          # put the extracted dataset/ folder here
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

PRODUCTS = ("NDVI", "NDWI", "SCL")   # SCL = Scene classification map

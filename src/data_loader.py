"""Step 1 - discover and load the Sentinel-2 monthly GeoTIFFs."""
import re
import numpy as np
import rasterio
from .config import DATA_DIR

_PAT = re.compile(r"^(\d{4})-(\d{2})-\d{2}-.*_L2A_(NDVI|NDWI|Scene_classification_map_)\.tiff$")


def index_files(data_dir=DATA_DIR):
    """Return {(product, 'YYYY-MM'): Path}. Handles the July-2025 NDWI file whose
    end-date is the 30th instead of the 31st."""
    out = {}
    for f in sorted(data_dir.glob("*.tiff")):
        m = _PAT.match(f.name)
        if not m:
            continue
        prod = "SCL" if m.group(3).startswith("Scene") else m.group(3)
        out[(prod, f"{m.group(1)}-{m.group(2)}")] = f
    return out


def load_rgb(path):
    """Read a 3-band raster -> float32 array (H, W, 3) in [0, 1] + rasterio profile."""
    with rasterio.open(path) as src:
        arr = src.read().astype("float32").transpose(1, 2, 0)
        prof = src.profile
    return arr, prof


def valid_mask(scl_rgb):
    """Pixels with observations. In the SCL render, pure black = no data / masked."""
    return scl_rgb.max(axis=-1) > 0

"""Step 2 - invert the Sentinel-Hub colour renders back into numeric layers.

The GeoTIFFs are RGB *visualisations*, not raw index rasters.  We invert them by
nearest-palette matching.  NDVI/SCL use discrete palettes (exact match).  NDWI is
a continuous white->blue (water) / white->green (land) ramp.

NOTE: index values are approximate (standard EO-Browser NDVI ramp assumed);
treat them as an ordered, monotonic proxy rather than calibrated reflectance.
"""
import numpy as np

# (RGB, approximate NDVI)
_NDVI = [((13, 13, 13), -0.5), ((191, 191, 191), -0.2), ((219, 219, 219), -0.1),
         ((235, 235, 235), 0.0), ((255, 250, 204), 0.025), ((237, 232, 181), 0.05),
         ((222, 217, 156), 0.075), ((204, 199, 130), 0.1), ((189, 184, 107), 0.125),
         ((176, 194, 97), 0.15), ((163, 204, 89), 0.175), ((145, 191, 82), 0.2),
         ((128, 179, 71), 0.25), ((112, 163, 64), 0.3), ((97, 150, 54), 0.35),
         ((79, 138, 46), 0.4), ((64, 125, 36), 0.45), ((48, 110, 28), 0.5),
         ((33, 97, 18), 0.55), ((15, 84, 10), 0.6), ((0, 69, 0), 0.7)]

# SCL classes: 0 nodata | 1 not-vegetated | 2 water | 3 vegetation | 4 cloud/shadow | 5 other
_SCL = [((0, 0, 0), 0), ((255, 230, 90), 1), ((0, 0, 255), 2), ((0, 160, 0), 3),
        ((128, 128, 128), 4), ((192, 192, 192), 4), ((47, 47, 47), 4),
        ((255, 0, 0), 5), ((100, 50, 0), 5), ((100, 200, 255), 5)]
SCL_NAMES = {0: "no data", 1: "not vegetated", 2: "water", 3: "vegetation", 4: "cloud/shadow", 5: "other"}


def _lut(rgb01, palette):
    cols = np.array([p[0] for p in palette], np.int32)
    vals = np.array([p[1] for p in palette], np.float32)
    a = np.rint(rgb01 * 255).astype(np.int32)
    key = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    uk, inv = np.unique(key.ravel(), return_inverse=True)
    ucol = np.stack([(uk >> 16) & 255, (uk >> 8) & 255, uk & 255], 1)
    d = ((ucol[:, None, :] - cols[None]) ** 2).sum(-1)
    out = vals[d.argmin(1)][inv].reshape(key.shape)
    err = np.sqrt(d.min(1))[inv].reshape(key.shape)
    return out, err


def decode_ndvi(rgb01):
    return _lut(rgb01, _NDVI)


def decode_scl(rgb01):
    v, e = _lut(rgb01, _SCL)
    return v.astype(np.uint8), e


def decode_ndwi(rgb01):
    """Blue tint -> positive NDWI (water), green tint -> negative (land), white -> 0."""
    a = rgb01 * 255.0
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    mag = (255.0 - R) / 255.0
    sign = np.where(B > G + 0.5, 1.0, np.where(G > B + 0.5, -1.0, 0.0))
    return (sign * mag).astype(np.float32)

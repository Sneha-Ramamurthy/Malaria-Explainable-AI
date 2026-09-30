"""STEP 1 - Data audit.  Run:  python -m src.step1_data_audit"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import OUT_DIR
from .data_loader import index_files, load_rgb, valid_mask

files = index_files()
months = sorted({m for _, m in files})
rows = []
for m in months:
    scl = load_rgb(files[("SCL", m)])[0]
    vm = valid_mask(scl)
    row = {"month": m, "valid_pct": round(100 * vm.mean(), 1)}
    for p in ("NDVI", "NDWI", "SCL"):
        row[p] = (p, m) in files
    # SCL render colours: blue = water, green = vegetation, yellow = not-vegetated
    r, g, b = scl[..., 0], scl[..., 1], scl[..., 2]
    row["water_pct_of_valid"] = round(100 * ((b > .9) & (r < .2) & (g < .2)).sum() / max(vm.sum(), 1), 2)
    rows.append(row)
df = pd.DataFrame(rows)
df["usable"] = df.valid_pct > 10          # monsoon months are fully cloud-masked
df.to_csv(OUT_DIR / "step1_data_audit.csv", index=False)
print(df.to_string(index=False))

# overview grid of NDVI per month (coverage)
fig, axes = plt.subplots(3, 8, figsize=(24, 8))
for ax, m in zip(axes.ravel(), months):
    ax.axis("off")
    if ("NDVI", m) in files:
        ax.imshow(load_rgb(files[("NDVI", m)])[0][::8, ::8])
    ax.set_title(f"{m}  valid {df.loc[df.month == m, 'valid_pct'].item()}%", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "step1_monthly_overview.png", dpi=70)
print("saved outputs")

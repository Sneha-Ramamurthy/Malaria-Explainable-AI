"""STEP 1 - Data audit. Run: python -m src.step1_data_audit"""

import math
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import OUT_DIR
from .data_loader import index_files, load_rgb, valid_mask


def main():
    # Ensure the output directory exists before saving files.
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = index_files()

    if not files:
        raise ValueError(
            "No dataset files were found. Check your dataset folder and configuration."
        )

    months = sorted({month for _, month in files})
    rows = []

    for month in months:
        required_key = ("SCL", month)

        # Check whether the required SCL file exists.
        if required_key not in files:
            print(f"WARNING: Missing SCL data for month {month}.")
            rows.append({
                "month": month,
                "valid_pct": 0.0,
                "water_pct_of_valid": 0.0,
                "NDVI": ("NDVI", month) in files,
                "NDWI": ("NDWI", month) in files,
                "SCL": False,
                "usable": False,
                "status": "Missing SCL file",
            })
            continue

        try:
            scl = load_rgb(files[required_key])[0]

            # Check that the loaded data has the expected image dimensions.
            if scl is None or scl.size == 0:
                raise ValueError("SCL image is empty.")

            if scl.ndim != 3 or scl.shape[2] < 3:
                raise ValueError(
                    f"Expected an RGB image, received shape {scl.shape}."
                )

            vm = valid_mask(scl)

            if vm.size == 0:
                raise ValueError("Valid-data mask is empty.")

            valid_pct = round(100 * float(vm.mean()), 1)
            valid_count = int(vm.sum())

            # Calculate water percentage using the existing colour rule.
            r, g, b = scl[..., 0], scl[..., 1], scl[..., 2]
            water_count = int(
                ((b > 0.9) & (r < 0.2) & (g < 0.2)).sum()
            )
            water_pct = round(
                100 * water_count / max(valid_count, 1), 2
            )

            usable = valid_pct > 10

            rows.append({
                "month": month,
                "valid_pct": valid_pct,
                "water_pct_of_valid": water_pct,
                "NDVI": ("NDVI", month) in files,
                "NDWI": ("NDWI", month) in files,
                "SCL": True,
                "usable": usable,
                "status": "OK" if usable else "Low valid-data coverage",
            })

        except Exception as exc:
            print(f"WARNING: Could not audit month {month}: {exc}")
            rows.append({
                "month": month,
                "valid_pct": 0.0,
                "water_pct_of_valid": 0.0,
                "NDVI": ("NDVI", month) in files,
                "NDWI": ("NDWI", month) in files,
                "SCL": True,
                "usable": False,
                "status": f"Audit failed: {exc}",
            })

    df = pd.DataFrame(rows)

    # Save the audit report.
    csv_path = OUT_DIR / "step1_data_audit.csv"
    df.to_csv(csv_path, index=False)

    print("\nDATA AUDIT SUMMARY")
    print("==================")
    print(df.to_string(index=False))
    print(f"\nMonths checked: {len(df)}")
    print(f"Usable months: {int(df['usable'].sum())}")
    print(f"Unusable months: {int((~df['usable']).sum())}")
    print(f"CSV report saved to: {csv_path}")

    # Create a monthly NDVI overview.
    columns = 8
    plot_months = months
    rows_needed = max(1, math.ceil(len(plot_months) / columns))

    fig, axes = plt.subplots(
        rows_needed, columns,
        figsize=(24, 3 * rows_needed),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.axis("off")

    for ax, month in zip(axes.ravel(), plot_months):
        if ("NDVI", month) in files:
            try:
                ndvi = load_rgb(files[("NDVI", month)])[0]
                if ndvi is not None and ndvi.size > 0:
                    ax.imshow(ndvi[::8, ::8])
                else:
                    ax.text(0.5, 0.5, "Empty NDVI", ha="center")
            except Exception as exc:
                ax.text(0.5, 0.5, "NDVI load failed", ha="center")
                print(f"WARNING: Could not display NDVI for {month}: {exc}")
        else:
            ax.text(0.5, 0.5, "Missing NDVI", ha="center")

        month_row = df[df["month"] == month]
        valid_pct = (
            month_row["valid_pct"].iloc[0]
            if not month_row.empty else 0
        )
        ax.set_title(f"{month} | valid {valid_pct}%", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    overview_path = OUT_DIR / "step1_monthly_overview.png"
    plt.savefig(overview_path, dpi=70)
    plt.close(fig)

    print(f"Overview image saved to: {overview_path}")
    print("Data audit completed.")


if _name_ == "_main_":
    main()

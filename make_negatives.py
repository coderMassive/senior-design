"""Make one "no fire" patch for every fire patch already in fire_patches/.

For fire row_N (location A, date D) we build neg_N:
  same date D, but at the location of a different fire B (at least --min-km away).
Same date means the model cannot cheat by learning the season.
Locations come from other fires, so negatives sit in fire-prone areas too.
Run fire_patch.py first to create the row_* folders.
"""
import argparse
import json
import math
import random
from pathlib import Path

import ee
import pandas as pd
import rasterio

from fire_patch import build_layers, make_grid, download_layer


def km_between(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    )
    return 6371 * 2 * math.asin(math.sqrt(a))


def make_negative(lat, lon, date, dest, args, source_row):
    dest.mkdir(parents=True, exist_ok=True)
    layers, provenance = build_layers(lat, lon, date)
    crs, transform, region, _ = make_grid(
        lat, lon, args.half_width_km, args.scale
    )

    arrays, names, profile = [], [], None
    for name, (image, _, _) in layers.items():
        data, current = download_layer(
            image, name, dest / f"{name}.tif", crs, transform, region
        )
        if profile is None:
            profile = current
        elif data.shape != arrays[0].shape:
            raise RuntimeError(f"Raster alignment mismatch: {name}")
        arrays.append(data)
        names.append(name)

    stack_profile = profile.copy()
    stack_profile.update(count=len(arrays))
    with rasterio.open(
        dest / "environment_stack.tif", "w", **stack_profile
    ) as dst:
        for i, (name, data) in enumerate(zip(names, arrays), 1):
            dst.write(data, i)
            dst.set_band_description(i, name)

    for name in names:  # per-layer files are not needed after stacking
        (dest / f"{name}.tif").unlink(missing_ok=True)

    metadata = {
        "label": 0,
        "source_fire_row": source_row,
        "latitude": lat,
        "longitude": lon,
        "discovery_date": date.strftime("%Y-%m-%d"),
        "bands": names,
        "output_crs": crs,
        "output_pixel_size_m": args.scale,
        "requested_half_width_km": args.half_width_km,
        "nodata": -9999,
        **provenance,
    }
    # Written last, so a folder without it counts as unfinished.
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("fire_patches"))
    p.add_argument("--project", required=True)
    p.add_argument("--half-width-km", type=float, default=10)
    p.add_argument("--scale", type=float, default=250)
    p.add_argument("--min-km", type=float, default=30)
    p.add_argument("--limit", type=int, default=0, help="0 means no limit")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ee.Initialize(project=args.project)

    fires = []
    for d in sorted(args.root.glob("row_*")):
        meta_path = d / "metadata.json"
        if meta_path.exists() and (d / "environment_stack.tif").exists():
            fires.append(json.loads(meta_path.read_text()))
    print(f"found {len(fires)} fire patches")

    rng = random.Random(args.seed)
    made = 0
    for m in fires:
        row = m["row_index_zero_based"]
        dest = args.root / f"neg_{row}"
        if (dest / "metadata.json").exists():
            continue
        if args.limit and made >= args.limit:
            break

        date = pd.Timestamp(m["discovery_date"])
        far = [
            o for o in fires
            if km_between(m["latitude"], m["longitude"],
                          o["latitude"], o["longitude"]) >= args.min_km
        ]
        for _ in range(3):
            other = rng.choice(far)
            try:
                make_negative(
                    other["latitude"], other["longitude"], date, dest, args, row
                )
                made += 1
                print(f"neg_{row} done ({made})")
                break
            except Exception as e:
                print(f"neg_{row} failed, retrying elsewhere: {e}")


if __name__ == "__main__":
    main()

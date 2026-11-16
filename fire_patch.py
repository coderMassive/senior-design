import argparse
import json
import math
from datetime import timedelta
from pathlib import Path

import ee
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.transform import rowcol


def parse_date(value):
    """Handle FPA-FOD Julian dates and ordinary date strings."""
    if pd.isna(value):
        raise ValueError("Selected fire has no discovery date.")

    if isinstance(value, (int, float, np.number)):
        return pd.to_datetime(
            float(value), unit="D", origin="julian"
        ).normalize()

    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return pd.to_datetime(text).normalize()

    if 2_000_000 < numeric < 3_000_000:
        return pd.to_datetime(
            numeric, unit="D", origin="julian"
        ).normalize()

    return pd.to_datetime(text).normalize()


def load_fire(path, layer, row_number):
    if row_number < 0:
        raise ValueError("--row must be a nonnegative, zero-based row index.")

    if path.suffix.lower() == ".csv":
        table = pd.read_csv(
            path,
            skiprows=range(1, row_number + 1),
            nrows=1,
        )
    else:
        # Loads just the requested record instead of the full database.
        table = gpd.read_file(
            path,
            layer=layer,
            engine="pyogrio",
            skip_features=row_number,
            max_features=1,
        )

    if table.empty:
        raise IndexError(f"No record found at row {row_number}.")

    fire = table.iloc[0]

    for name in ("LATITUDE", "LONGITUDE", "DISCOVERY_DATE"):
        if name not in table.columns:
            raise ValueError(f"Missing required column: {name}")

    lat = float(fire["LATITUDE"])
    lon = float(fire["LONGITUDE"])
    date = parse_date(fire["DISCOVERY_DATE"])

    if not (24 <= lat <= 50 and -125 <= lon <= -66):
        raise ValueError(
            "This script is configured for the contiguous U.S., "
            "where gridMET is available."
        )

    if date.year < 2002:
        raise ValueError(
            "Use a fire from 2002 onward for this MODIS-based configuration."
        )

    return fire, lat, lon, date


def require_latest(collection, name):
    if collection.size().getInfo() == 0:
        raise RuntimeError(f"No {name} image exists for the requested dates.")

    return ee.Image(
        collection.sort("system:time_start", False).first()
    )


def acquisition_date(image):
    return ee.Date(
        image.get("system:time_start")
    ).format("YYYY-MM-dd").getInfo()


def build_layers(lat, lon, fire_date):
    point = ee.Geometry.Point([lon, lat])

    # Use the previous day's completed daily data, not fire-day data.
    weather_day = fire_date - timedelta(days=1)

    weather = require_latest(
        ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
        .filterDate(
            weather_day.strftime("%Y-%m-%d"),
            fire_date.strftime("%Y-%m-%d"),
        ),
        "gridMET",
    )

    # A MODIS composite covers 16 days. Exclude composites whose
    # observation window could overlap discovery day.
    vegetation_end = fire_date - timedelta(days=16)
    vegetation_start = fire_date - timedelta(days=96)

    vegetation = require_latest(
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterBounds(point)
        .filterDate(
            vegetation_start.strftime("%Y-%m-%d"),
            vegetation_end.strftime("%Y-%m-%d"),
        ),
        "MODIS vegetation",
    )

    # Exclude snow/cloud classes; retain good and marginal QA pixels.
    vegetation = vegetation.updateMask(
        vegetation.select("SummaryQA").lte(1)
    )

    previous_year = fire_date.year - 1
    cover = require_latest(
        ee.ImageCollection("MODIS/061/MCD12Q1")
        .filterBounds(point)
        .filterDate(
            f"{previous_year}-01-01",
            f"{fire_date.year}-01-01",
        ),
        "previous-year land cover",
    )

    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation")
    slope = ee.Terrain.slope(elevation)

    # image, display colormap, units
    layers = {
        "ndvi": (
            vegetation.select("NDVI").multiply(0.0001),
            "YlGn",
            "index",
        ),
        "evi": (
            vegetation.select("EVI").multiply(0.0001),
            "YlGn",
            "index",
        ),
        "land_cover": (
            cover.select("LC_Type1"),
            "tab20",
            "IGBP class",
        ),
        "dead_fuel_moisture_100h": (
            weather.select("fm100"),
            "Blues",
            "%",
        ),
        "dead_fuel_moisture_1000h": (
            weather.select("fm1000"),
            "Blues",
            "%",
        ),
        "temperature_max": (
            weather.select("tmmx").subtract(273.15),
            "inferno",
            "deg C",
        ),
        "precipitation": (
            weather.select("pr"),
            "Blues",
            "mm/day",
        ),
        "wind_speed": (
            weather.select("vs"),
            "viridis",
            "m/s",
        ),
        "relative_humidity_min": (
            weather.select("rmin"),
            "Blues",
            "%",
        ),
        "elevation": (
            elevation,
            "terrain",
            "m",
        ),
        "slope": (
            slope,
            "magma",
            "degrees",
        ),
    }

    provenance = {
        "weather_date": acquisition_date(weather),
        "vegetation_composite_start": acquisition_date(vegetation),
        "land_cover_year": previous_year,
        "vegetation_qa": "SummaryQA <= 1",
        "weather_source": "IDAHO_EPSCOR/GRIDMET",
        "vegetation_source": "MODIS/061/MOD13Q1",
        "land_cover_source": "MODIS/061/MCD12Q1",
        "terrain_source": "USGS/SRTMGL1_003",
    }

    return layers, provenance


def make_grid(lat, lon, half_width_km, scale_m):
    # Equal-area CRS with meter units, suitable for CONUS.
    crs = "EPSG:5070"

    center = (
        ee.Geometry.Point([lon, lat])
        .transform(crs, 1)
        .coordinates()
        .getInfo()
    )
    x, y = center

    pixels = math.ceil(2 * half_width_km * 1000 / scale_m)
    if not 1 <= pixels <= 10000:
        raise ValueError("Requested patch dimensions exceed download limits.")

    # Keep each float32 single-band request safely below 32 MB.
    if pixels * pixels * 4 > 24_000_000:
        raise ValueError(
            "Patch is too large for direct download. "
            "Reduce --half-width-km or increase --scale."
        )

    width = pixels * scale_m
    xmin = x - width / 2
    ymax = y + width / 2
    xmax = xmin + width
    ymin = ymax - width

    transform = [scale_m, 0, xmin, 0, -scale_m, ymax]
    region = ee.Geometry.Rectangle(
        [xmin, ymin, xmax, ymax],
        proj=crs,
        geodesic=False,
    )

    return crs, transform, region, center


def download_layer(image, name, destination, crs, transform, region):
    url = (
        image.rename(name)
        .toFloat()
        .unmask(-9999, sameFootprint=False)
        .getDownloadURL({
            "name": name,
            "crs": crs,
            "crs_transform": transform,
            "region": region,
            "format": "GEO_TIFF",
        })
    )

    temporary = destination.with_suffix(".download")
    try:
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)

        with rasterio.open(temporary) as src:
            data = src.read(1)
            profile = src.profile.copy()
            tags = src.tags()

        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=-9999,
            compress="deflate",
        )

        with rasterio.open(destination, "w", **profile) as dst:
            dst.write(data.astype("float32"), 1)
            dst.set_band_description(1, name)
            dst.update_tags(**tags)

        return data, profile
    finally:
        temporary.unlink(missing_ok=True)


def preview(data, name, cmap, units, profile, center, destination):
    masked = np.ma.masked_where(
        (data == -9999) | ~np.isfinite(data), data
    )
    rr, cc = rowcol(profile["transform"], *center)

    fig, ax = plt.subplots(figsize=(6, 6))
    kwargs = {"cmap": cmap, "interpolation": "nearest"}

    if name == "land_cover":
        kwargs.update(vmin=1, vmax=17)
    elif name in ("ndvi", "evi"):
        kwargs.update(vmin=-0.2, vmax=1)

    image = ax.imshow(masked, **kwargs)
    ax.scatter(
        [cc], [rr], marker="+", s=150, color="red", linewidths=2
    )
    ax.set_title(f"{name}\nred cross = reported fire location")
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, shrink=0.75, label=units)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpa", type=Path, required=True)
    parser.add_argument("--layer", default="Fires")
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--half-width-km", type=float, default=10)
    parser.add_argument("--scale", type=float, default=250)
    parser.add_argument("--out", type=Path, default=Path("fire_patches"))
    args = parser.parse_args()

    if args.half_width_km <= 0 or args.scale <= 0:
        parser.error("Patch width and pixel scale must be positive.")

    ee.Initialize(project=args.project)

    fire, lat, lon, date = load_fire(
        args.fpa, args.layer, args.row
    )

    destination = args.out / f"row_{args.row}"
    destination.mkdir(parents=True, exist_ok=True)

    layers, provenance = build_layers(lat, lon, date)
    crs, transform, region, center = make_grid(
        lat, lon, args.half_width_km, args.scale
    )

    arrays = []
    profile = None
    names = []

    for name, (image, cmap, units) in layers.items():
        print(f"Downloading {name}...")
        data, current_profile = download_layer(
            image, name, destination / f"{name}.tif",
            crs, transform, region,
        )

        if profile is not None:
            if (
                current_profile["transform"] != profile["transform"]
                or data.shape != arrays[0].shape
                or current_profile["crs"] != profile["crs"]
            ):
                raise RuntimeError(f"Raster alignment mismatch: {name}")
        else:
            profile = current_profile

        preview(
            data, name, cmap, units, current_profile, center,
            destination / f"{name}.png",
        )
        arrays.append(data)
        names.append(name)

    stack_profile = profile.copy()
    stack_profile.update(count=len(arrays))

    with rasterio.open(
        destination / "environment_stack.tif",
        "w",
        **stack_profile,
    ) as dst:
        for index, (name, data) in enumerate(zip(names, arrays), 1):
            dst.write(data, index)
            dst.set_band_description(index, name)

    metadata = {
        "row_index_zero_based": args.row,
        "fod_id": str(fire.get("FOD_ID", "")),
        "latitude": lat,
        "longitude": lon,
        "discovery_date": date.strftime("%Y-%m-%d"),
        "bands": names,
        "output_crs": crs,
        "output_pixel_size_m": args.scale,
        "requested_half_width_km": args.half_width_km,
        "resampling": "nearest neighbor",
        "nodata": -9999,
        **provenance,
    }

    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"Saved {len(names)} layers and a multiband stack to {destination}")


if __name__ == "__main__":
    main()

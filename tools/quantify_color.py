"""Per-slide color/stain features over TRIDENT tissue masks."""

import glob
import json
import os
import random

import numpy as np
import openslide
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.stats import circmean, circstd
from shapely.geometry import shape as shapely_shape
from skimage.color import rgb2hed, rgb2hsv


def find_geojson(slide_path, geojson_dir):
    base = os.path.basename(slide_path)
    stem = os.path.splitext(base)[0]
    for cand in (f"{stem}.geojson", f"{base}.geojson"):
        p = os.path.join(geojson_dir, cand)
        if os.path.exists(p):
            return p
    return None


def choose_level(slide, target_long_edge=3000):
    level = 0
    for lvl, dim in enumerate(slide.level_dimensions):
        if max(dim) < target_long_edge:
            break
        level = lvl
    return level


def read_rgb(slide, level):
    """Full level as float RGB in [0, 1]."""
    region = slide.read_region((0, 0), level, slide.level_dimensions[level])
    return np.asarray(region.convert("RGB"), dtype=np.float32) / 255.0


def load_geometries(geojson_path):
    with open(geojson_path) as f:
        gj = json.load(f)

    if gj.get("type") == "FeatureCollection":
        geoms = [shapely_shape(feat["geometry"]) for feat in gj["features"]]
    elif gj.get("type") == "Feature":
        geoms = [shapely_shape(gj["geometry"])]
    else:                                          # bare geometry dict
        geoms = [shapely_shape(gj)]
    return [g for g in geoms if not g.is_empty]


def geojson_to_mask(geojson_path, slide, level):
    W0, H0 = slide.level_dimensions[0]
    Wl, Hl = slide.level_dimensions[level]

    geoms = load_geometries(geojson_path)
    if not geoms:
        return np.zeros((Hl, Wl), dtype=bool)

    # transform maps raster pixel -> level-0 coords, so scale = level-0 / level
    arr = rasterize(
        ((g, 1) for g in geoms),
        out_shape=(Hl, Wl),
        transform=Affine.scale(W0 / Wl, H0 / Hl),
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    return arr.astype(bool)


def tissue_features(rgb, mask):
    """Color (HSV) and stain (HED) statistics over the masked tissue."""
    hsv = rgb2hsv(rgb)
    h = hsv[..., 0][mask]                          # hue in [0, 1] == [0, 360)
    s = hsv[..., 1][mask]

    # Hue is circular: 0 and 1 are the same color.
    hue_mean = circmean(h, high=1.0, low=0.0)
    hue_std = circstd(h, high=1.0, low=0.0)

    hed = rgb2hed(rgb)
    h_mean = float(hed[..., 0][mask].mean())
    e_mean = float(hed[..., 1][mask].mean())

    return {
        "hue_mean_deg":     float(hue_mean * 360.0),
        "hue_std_deg":      float(hue_std * 360.0),
        "sat_mean":         float(s.mean()),
        "sat_std":          float(s.std()),
        "sat_p10":          float(np.percentile(s, 10)),
        "sat_p90":          float(np.percentile(s, 90)),
        "tissue_fraction":  float(mask.mean()),
        "hematoxylin_mean": h_mean,
        "eosin_mean":       e_mean,
        "h_e_ratio":        h_mean / (e_mean + 1e-8),
    }


def analyze_slides(paths, geojson_dir, target_long_edge=3000):
    rows = []
    for i, path in enumerate(paths, 1):
        name = os.path.basename(path)
        tag = f"[{i}/{len(paths)}]"

        gj = find_geojson(path, geojson_dir)
        if gj is None:
            print(f"{tag} skip (no GeoJSON): {name}")
            continue

        with openslide.OpenSlide(path) as slide:
            level = choose_level(slide, target_long_edge)
            mask = geojson_to_mask(gj, slide, level)
            if not mask.any():
                print(f"{tag} skip (empty mask): {name}")
                continue
            rgb = read_rgb(slide, level)

        row = {"slide": path, **tissue_features(rgb, mask)}
        rows.append(row)
        print(f"{tag} done: {name}  tissue {row['tissue_fraction']:.1%}")

    df = pd.DataFrame(rows)
    return df.set_index("slide") if not df.empty else df


if __name__ == "__main__":
    geojson_dir = "/path/containing/geojson/segmentations"
    slide_paths = glob.glob("path/to/svs/files")
    print(f"Found {len(slide_paths)} slides")

    slide_paths = random.sample(slide_paths, k=min(100, len(slide_paths)))
    df = analyze_slides(slide_paths, geojson_dir)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n=== Per-slide color features ===")
    print(df)

    df.to_csv("output.csv")
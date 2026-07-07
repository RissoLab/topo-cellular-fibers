#!/usr/bin/env python
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pixhomology as px
import tifffile
from scipy.ndimage import binary_fill_holes, find_objects, label
from skimage.color import rgb2gray, rgb2hed
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (
    closing,
    convex_hull_image,
    disk,
    opening,
    remove_small_holes,
    remove_small_objects,
)
from tqdm import tqdm


def normalize_channel(channel, mask):
    values = channel[mask]
    p1, p99 = np.percentile(values, (1, 99))
    channel = np.clip(channel, p1, p99)
    channel = (channel - p1) / (p99 - p1)
    channel[~mask] = 0
    return channel


def segmentation(
    img,
    window=128,
    t_lifetime=10,
    t_area=500,
    t_distance=2,
    max_objects=None,
    verbose=True,
):
    def log(message):
        if verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")

    def start_step(name):
        log(f"Start: {name}")
        return perf_counter()

    def end_step(name, start):
        elapsed = perf_counter() - start
        log(f"Done: {name} ({elapsed:.2f}s)")

    step = start_step("tissue mask")
    gray = rgb2gray(img)
    threshold = threshold_otsu(gray)
    tissue_mask = gray < threshold
    tissue_mask &= img.max(axis=2) > 20
    tissue_mask = closing(tissue_mask, disk(3))
    tissue_mask = remove_small_objects(tissue_mask, max_size=100)
    tissue_mask = remove_small_holes(tissue_mask, max_size=10000)
    end_step("tissue mask", step)

    step = start_step("stain channel")
    hed = rgb2hed(img)
    h = normalize_channel(hed[..., 0], tissue_mask)
    e = normalize_channel(hed[..., 1], tissue_mask)

    od = -np.log((img.astype(np.float32) + 1) / 256)
    stain = normalize_channel(od.mean(axis=2), tissue_mask)

    h = np.maximum(h, e)
    h = np.maximum(h, stain)
    h = gaussian(h, sigma=1.2)
    h = normalize_channel(h, tissue_mask)
    h = (h * 255).astype(np.uint8)
    end_step("stain channel", step)

    step = start_step("persistent homology")
    dgms, idxs = px.computePH(h, maxdim=1, return_index=True)
    lifetimes = np.abs(dgms[1][:, 0] - dgms[1][:, 1])

    keep = lifetimes > t_lifetime
    order = np.flip(np.argsort(lifetimes[keep]))

    values = dgms[1][:, 0][keep][order]
    birth_points = idxs[1][:, :2][keep][order]
    death_points = idxs[1][:, 2:][keep][order]

    distances = np.sqrt(np.sum((birth_points - death_points) ** 2, axis=1))
    far = distances > t_distance

    values = values[far]
    birth_points = birth_points[far]
    end_step("persistent homology", step)

    step = start_step("segmentation")
    height, width = h.shape
    segm = np.zeros_like(h, dtype=np.uint32)
    n_objects = 0

    for i in tqdm(range(len(values)), disable=not verbose):
        x, y = birth_points[i]
        value = values[i]

        if segm[y, x] != 0:
            continue

        if max_objects is not None and n_objects >= max_objects:
            break

        y0 = max(y - window, 0)
        y1 = min(y + window, height)
        x0 = max(x - window, 0)
        x1 = min(x + window, width)

        cy = y - y0
        cx = x - x0

        patch = h[y0:y1, x0:x1]
        mask = patch >= value
        filled_mask = binary_fill_holes(mask)
        object_mask = binary_fill_holes(filled_mask ^ mask)

        labels, num_labels = label(object_mask)
        if num_labels == 0:
            continue

        nearest_label = labels[cy, cx]
        if nearest_label == 0:
            yy, xx = np.nonzero(object_mask)
            if len(yy) == 0:
                continue

            nearest = np.argmin((yy - cy) ** 2 + (xx - cx) ** 2)
            nearest_label = labels[yy[nearest], xx[nearest]]

        obj = labels == nearest_label
        free = segm[y0:y1, x0:x1] == 0
        if np.any(obj & ~free):
            continue
        if np.count_nonzero(obj) > t_area:
            n_objects += 1
            segm[y0:y1, x0:x1][obj] = n_objects
    end_step("segmentation", step)

    return segm


def postprocess_segmentation(
    segm, radius=2, min_area=500, hole_area=500, pad=8, use_hull=True, verbose=True
):
    def log(message):
        if verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")

    log("Start: postprocess segmentation")

    out = np.zeros_like(segm, dtype=np.uint32)
    occupied = np.zeros(segm.shape, dtype=bool)
    footprint = disk(radius)
    objects = find_objects(segm)

    labels = np.unique(segm)
    labels = labels[labels > 0]

    areas = np.bincount(segm.ravel())
    labels = labels[np.flip(np.argsort(areas[labels]))]

    height, width = segm.shape

    for value in tqdm(labels, disable=not verbose):
        obj_slice = objects[value - 1]
        if obj_slice is None:
            continue

        y_slice, x_slice = obj_slice
        y0 = max(y_slice.start - pad, 0)
        y1 = min(y_slice.stop + pad, height)
        x0 = max(x_slice.start - pad, 0)
        x1 = min(x_slice.stop + pad, width)

        crop = segm[y0:y1, x0:x1] == value
        crop = binary_fill_holes(crop)
        crop = closing(crop, footprint)
        if use_hull:
            crop = convex_hull_image(crop)
        crop = opening(crop, footprint)
        crop = remove_small_holes(crop, max_size=hole_area)
        crop = remove_small_objects(crop, max_size=min_area)

        crop_labels, num_labels = label(crop)
        if num_labels > 1:
            crop_areas = np.bincount(crop_labels.ravel())
            crop = crop_labels == np.argmax(crop_areas[1:]) + 1

        if np.any(crop & occupied[y0:y1, x0:x1]):
            continue

        if np.count_nonzero(crop) < min_area:
            continue

        out[y0:y1, x0:x1][crop] = value
        occupied[y0:y1, x0:x1][crop] = True

    log("Done: postprocess segmentation")
    return out


def save_segmentation(
    segm, labels_path="segm_labels.tif", preview_path="segm_preview.tif"
):
    max_label = segm.max()
    dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32

    tifffile.imwrite(
        labels_path,
        segm.astype(dtype, copy=False),
        compression="deflate",
    )

    preview = (segm > 0).astype(np.uint8) * 255
    tifffile.imwrite(
        preview_path,
        preview,
        compression="deflate",
    )


def parse_args():
    parser = ArgumentParser(
        description="Run persistent-homology segmentation on one H&E image."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--labels-out", required=True, type=Path)
    parser.add_argument("--preview-out", required=True, type=Path)
    parser.add_argument("--window", default=96, type=int)
    parser.add_argument("--t-lifetime", default=1.0, type=float)
    parser.add_argument("--t-area", default=100, type=int)
    parser.add_argument("--t-distance", default=10.0, type=float)
    parser.add_argument("--post-radius", default=2, type=int)
    parser.add_argument("--min-area", default=100, type=int)
    parser.add_argument("--hole-area", default=500, type=int)
    parser.add_argument("--no-hull", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    args.labels_out.parent.mkdir(parents=True, exist_ok=True)
    args.preview_out.parent.mkdir(parents=True, exist_ok=True)

    img = tifffile.imread(args.image)
    segm = segmentation(
        img,
        window=args.window,
        t_lifetime=args.t_lifetime,
        t_area=args.t_area,
        t_distance=args.t_distance,
        verbose=verbose,
    )
    segm = postprocess_segmentation(
        segm,
        radius=args.post_radius,
        min_area=args.min_area,
        hole_area=args.hole_area,
        use_hull=not args.no_hull,
        verbose=verbose,
    )
    save_segmentation(
        segm,
        labels_path=args.labels_out,
        preview_path=args.preview_out,
    )

    if verbose:
        print(f"Labels: {args.labels_out}")
        print(f"Preview: {args.preview_out}")
        print(f"Objects: {int(segm.max())}")


if __name__ == "__main__":
    main()

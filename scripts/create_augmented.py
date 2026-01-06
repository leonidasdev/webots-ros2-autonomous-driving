#!/usr/bin/env python3

"""Create canonical and pre-scaled template images in `resources/`.

Description:
    Utility script that generates canonical augmented copies and a small
    set of pre-scaled template variants for each original template image
    found in the `resources/` directory. Pre-scaling reduces runtime
    work for the sign detector and keeps template matching deterministic.

Behaviour summary:
    - For each original image (files without ``_aug_`` in the name), the
        script writes up to ``max_per_image`` canonical copies named
        ``<basename>_aug_<N>.<ext>``.
    - For each canonical copy a collection of scaled templates is
        written using the suffix ``_scaleNN`` where ``NN`` is the scale as
        an integer percentage (e.g. ``scale110`` for 1.10).
    - A marker file (``.aug_done``) is created after a successful run to
        avoid re-generating templates on subsequent launches.

Usage:
    python create_augmented.py

Notes:
    This script is also invoked by ``launch/launch.py`` during startup
    when template augmentation is required.
"""

import os
import re
import cv2
import numpy as np
from random import seed


def transform_image(img, scale=1.0):
    """Return an isotropically scaled copy of ``img`` about its center.

    Args:
        img (numpy.ndarray): Input image (H x W x C) in BGR or grayscale.
        scale (float): Isotropic scale factor applied about image center.

    Returns:
        numpy.ndarray: A warped image with the same canvas size as ``img``
            containing the scaled content.

    Notes:
        - The returned image keeps the original canvas size; use
            ``cv2.resize`` when explicit output dimensions are required.
        - This helper is provided for completeness; the script primarily
            uses ``cv2.resize`` when producing explicit scaled templates.
    """
    h, w = img.shape[:2]
    M_scale = cv2.getRotationMatrix2D((w / 2, h / 2), 0, float(scale))
    img_s = cv2.warpAffine(
        img,
        M_scale,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return img_s


def create_augmented_templates(input_dir=None, max_per_image=20, seed_val=42):
    """Generate canonical `_aug_N` copies and corresponding scaled templates.

    Args:
        input_dir (str|None): Path to the `resources/` folder. If ``None``
            the script-local ``resources/`` directory is used.
        max_per_image (int): Maximum number of canonical `_aug_N` copies
            to create per original image.
        seed_val (int): Random seed for deterministic file-name choices.

    Returns:
        None

    Behaviour / Side effects:
        - Writes new files to the `resources/` folder for canonical copies
          and scaled variants.
        - Creates a marker file `.aug_done` in the resources folder on
          successful completion to avoid re-generation in subsequent runs.
        - Prints summary information to stdout about created/skipped files.
    """
    seed(seed_val)
    np.random.seed(seed_val)

    # Resolve input_dir relative to the script location when not provided.
    # The script lives in `scripts/`, while `resources/` is one level
    # up in the package. Use the parent `resources/` path by default.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if input_dir is None:
        input_dir = os.path.normpath(os.path.join(script_dir, '..', 'resources'))

    # Marker file prevents re-running augmentation on every launch. Place
    # the marker inside the `resources/` folder so it travels with templates.
    marker_path = os.path.join(input_dir, '.aug_done')
    if os.path.exists(marker_path):
        print('Augmentations already created (marker found). Skipping generation.')
        return

    if not os.path.isdir(input_dir):
        print(f"ERROR: input directory '{input_dir}' not found")
        return

    # Gather candidate image files (case-insensitive extensions)
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    originals = [f for f in all_files if '_aug_' not in f]

    if not originals:
        print(f"No original template images found in {input_dir}/")
        return

    created = 0
    skipped = 0

    print(f"Found {len(originals)} original templates. Generating up to {max_per_image} variants each...")

    # Scales produced for each canonical copy. Values >1.0 are zoomed-in
    # (larger) templates; values <1.0 are zoomed-out (smaller) templates.
    output_scales = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55]

    for fname in originals:
        path = os.path.join(input_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"  Warning: could not load {fname}")
            continue

        name, ext = os.path.splitext(fname)

        # Identify existing canonical augmented copies exactly matching
        # the pattern '<name>_aug_<N><ext>' (case-insensitive). Scaled
        # templates that include '_scale' are excluded by the regex.
        pattern = re.compile(rf"^{re.escape(name)}_aug_(\d+){re.escape(ext)}$", re.IGNORECASE)
        existing_indices = set()
        for ef in all_files:
            m = pattern.match(ef)
            if m:
                try:
                    existing_indices.add(int(m.group(1)))
                except ValueError:
                    continue

        if len(existing_indices) >= max_per_image:
            print(f"  {fname}: already has {len(existing_indices)} augmented variants — skipping")
            continue

        need = max_per_image - len(existing_indices)

        # Allocate the lowest available integer indices starting at 0
        free_indices = []
        idx = 0
        while len(free_indices) < need:
            if idx not in existing_indices:
                free_indices.append(idx)
            idx += 1

        # Generate required number of variants and assign to free indices
        created_this = 0
        for out_idx in free_indices:
            aug_name = f"{name}_aug_{out_idx}{ext}"
            aug_path = os.path.join(input_dir, aug_name)
            if os.path.exists(aug_path):
                skipped += 1
                continue

            # Save canonical copy
            out = img.copy()
            h, w = img.shape[:2]
            try:
                ok = cv2.imwrite(aug_path, out)
                if not ok:
                    print(f"  Warning: failed to write {aug_name}")
                    skipped += 1
                    continue
            except Exception as e:
                print(f"  Warning: exception writing {aug_name}: {e}")
                skipped += 1
                continue

            created += 1
            created_this += 1

            # Write scaled-template files so the detector can load pre-scaled
            # templates without runtime resizing.
            for out_scale in output_scales:
                scaled_h = max(8, int(h * out_scale))
                scaled_w = max(8, int(w * out_scale))
                try:
                    scaled = cv2.resize(out, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                    scale_name = f"{name}_aug_{out_idx}_scale{int(out_scale*100)}{ext}"
                    scale_path = os.path.join(input_dir, scale_name)
                    # Avoid overwriting if already present
                    if not os.path.exists(scale_path):
                        ok2 = cv2.imwrite(scale_path, scaled)
                        if ok2:
                            created += 1
                        else:
                            print(f"  Warning: failed to write {scale_name}")
                            skipped += 1
                except Exception as e:
                    print(f"  Warning: exception creating scaled template {scale_name}: {e}")
                    skipped += 1
                    continue

        print(f"  {fname}: created {created_this} new variants (existing: {len(existing_indices)})")

    print("Done.")
    print(f"  created: {created}")
    print(f"  skipped (already existed or failed writes): {skipped}")
    print(f"  resources folder: {input_dir}/")
    print("Note: Files use the '_aug_' naming pattern and will be loaded by sign_detector from resources/")

    # Create marker file to skip regeneration on future runs
    try:
        with open(marker_path, 'w') as f:
            f.write('done')
        print(f"Wrote marker: {marker_path}")
    except Exception as e:
        print(f"Warning: could not write marker file: {e}")


if __name__ == '__main__':
    create_augmented_templates()
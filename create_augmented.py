#!/usr/bin/env python3
"""
create_augmented.py - generate simple augmentations into `resources/`.

This script produces scale variants of the
original color templates and writes color copies into
the `resources/` folder using the `_aug_` filename pattern. The detector
expects these pre-scaled templates so no runtime resizing is necessary.

Usage:
    python create_augmented.py
"""

import os
import re
import cv2
import numpy as np
from random import choice, seed

def transform_image(img, scale=1.0):
    """Apply isotropic scaling (about the center).

    For canonical augmented images we write the original-sized image; the
    explicit scaled templates are written separately by resizing the
    canonical image to the desired `output_scales`.
    """
    h, w = img.shape[:2]
    M_scale = cv2.getRotationMatrix2D((w/2, h/2), 0, float(scale))
    img_s = cv2.warpAffine(img, M_scale, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return img_s


def create_augmented_templates(input_dir=None, max_per_image=20, seed_val=42):
    seed(seed_val)
    np.random.seed(seed_val)

    # Resolve input_dir relative to the script location when not provided
    if input_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, 'resources')

    # Marker file prevents re-running augmentation on every launch
    marker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.aug_done')
    if os.path.exists(marker_path):
        print('Augmentations already created (marker found). Skipping generation.')
        return

    if not os.path.isdir(input_dir):
        print(f"ERROR: input directory '{input_dir}' not found")
        return

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    originals = [f for f in files if '_aug_' not in f]

    if not originals:
        print(f"No original template images found in {input_dir}/")
        return

    created = 0
    skipped = 0

    print(f"Found {len(originals)} original templates. Generating up to {max_per_image} variants each...")

    # Parameter pools: only scaling. Keep color.
    output_scales = [1.0, 1.1, 1.2, 1.3, 0.9, 0.8]

    for fname in originals:
        path = os.path.join(input_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"  Warning: could not load {fname}")
            continue

        name, ext = os.path.splitext(fname)
        # Find existing augmented variants for this base filename
        all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        existing_aug = [f for f in all_files if f.startswith(f"{name}_aug_") and f.lower().endswith(ext.lower())]

        existing_indices = set()
        for ef in existing_aug:
            m = re.match(rf"^{re.escape(name)}_aug_(\d+){re.escape(ext)}$", ef)
            if m:
                try:
                    existing_indices.add(int(m.group(1)))
                except ValueError:
                    continue

        if len(existing_indices) >= max_per_image:
            print(f"  {fname}: already has {len(existing_indices)} augmented variants — skipping")
            continue

        need = max_per_image - len(existing_indices)

        # We don't create additional canonical photometric variants here;
        # the scaled templates (below) provide the size diversity used at
        # detection time. We'll create up to `max_per_image` canonical
        # copies (identical to original) named with distinct indices.

        # Find available indices to name new augmented files
        free_indices = []
        idx = 0
        while len(free_indices) < need:
            if idx not in existing_indices:
                free_indices.append(idx)
            idx += 1

        # Generate required number of variants and assign to free indices
        created_this = 0
        # Skip indices known to produce problematic variants
        skip_indices = {13, 18}
        for out_idx in free_indices:
            if out_idx in skip_indices:
                print(f"  Skipping problematic augmentation index {out_idx} for {name}")
                continue
            aug_name = f"{name}_aug_{out_idx}{ext}"
            aug_path = os.path.join(input_dir, aug_name)
            if os.path.exists(aug_path):
                skipped += 1
                continue

            # Save canonical copy (no photometric changes)
            out = img.copy()
            h, w = img.shape[:2]
            cv2.imwrite(aug_path, out)
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
                        cv2.imwrite(scale_path, scaled)
                        created += 1
                except Exception:
                    continue

        print(f"  {fname}: created {created_this} new variants (existing: {len(existing_indices)})")

    print("\nDone.")
    print(f"  created: {created}")
    print(f"  skipped (already existed): {skipped}")
    print(f"  resources folder: {input_dir}/")
    print("Note: Files use the '_aug_' naming pattern and will be loaded by sign_detector from resources/")

    # Create marker file to skip regeneration on future runs
    try:
        with open(marker_path, 'w') as f:
            f.write('done')
        print(f"Wrote marker: {marker_path}")
    except Exception:
        pass


if __name__ == '__main__':
    create_augmented_templates()

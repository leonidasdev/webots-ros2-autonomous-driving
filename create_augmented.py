#!/usr/bin/env python3
"""
create_augmented.py - Generate targeted augmentations directly into `resources/`.

This script creates varied augmented templates (rotation, scale, brightness,
noise, small occlusions) and saves them into the existing `resources/` folder
using the `_aug_` filename pattern so `sign_detector` will keep using them.

Usage:
    python create_augmented.py
"""

import os
import re
import cv2
import numpy as np
from random import choice, uniform, randint, seed


def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    # alpha: contrast (1.0 = no change), beta: brightness (0 = no change)
    img2 = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img2


def add_noise(img, sigma=5):
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    img_i = img.astype(np.int16) + noise
    img_i = np.clip(img_i, 0, 255).astype(np.uint8)
    return img_i


def random_occlusion(img, max_frac=0.12):
    h, w = img.shape[:2]
    occ_w = int(w * uniform(0.03, max_frac))
    occ_h = int(h * uniform(0.03, max_frac))
    x = randint(0, max(0, w - occ_w))
    y = randint(0, max(0, h - occ_h))
    color = tuple([int(uniform(0, 255)) for _ in range(3)])
    img2 = img.copy()
    cv2.rectangle(img2, (x, y), (x + occ_w, y + occ_h), color, -1)
    return img2


def transform_image(img, rotate_deg=0.0, scale=1.0, brightness=1.0, beta=0, noise_sigma=0, occlude=False):
    h, w = img.shape[:2]

    # Scale + rotate around center
    M_scale = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
    img_s = cv2.warpAffine(img, M_scale, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    M = cv2.getRotationMatrix2D((w/2, h/2), rotate_deg, 1.0)
    img_r = cv2.warpAffine(img_s, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Brightness/contrast
    img_bc = adjust_brightness_contrast(img_r, alpha=brightness, beta=beta)

    # Noise
    if noise_sigma > 0:
        img_bc = add_noise(img_bc, sigma=noise_sigma)

    # Occlusion
    if occlude:
        img_bc = random_occlusion(img_bc)

    return img_bc


def create_augmented_templates(input_dir=None, max_per_image=20, seed_val=42):
    seed(seed_val)
    np.random.seed(seed_val)

    # Resolve input_dir relative to the script location when not provided
    if input_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, 'resources')

    # Marker file to skip repeated augmentation
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

    # Parameter pools
    rotations = [-15, -10, -5, 0, 5, 10, 15]
    scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    brightnesss = [0.8, 0.9, 1.0, 1.1, 1.25]
    beta_vals = [-30, -15, 0, 15, 30]
    noise_options = [0, 3, 6, 10]

    for fname in originals:
        path = os.path.join(input_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"  Warning: could not load {fname}")
            continue

        name, ext = os.path.splitext(fname)
        # Determine existing augmented files for this base name
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

        # Create variants pool (deterministic + random) and select only 'need' of them
        variants = []
        for rot in rotations[:3]:
            for sc in scales[:3]:
                variants.append((rot, sc, 1.0, 0, 0, False))
        while len(variants) < max_per_image:
            rot = choice(rotations)
            sc = choice(scales)
            br = choice(brightnesss)
            beta = choice(beta_vals)
            noise = choice(noise_options)
            occ = (randint(0, 4) == 0)
            variants.append((rot, sc, br, beta, noise, occ))

        # Find free indices to assign
        free_indices = []
        idx = 0
        while len(free_indices) < need:
            if idx not in existing_indices:
                free_indices.append(idx)
            idx += 1

        # Generate only the required number of variants and assign to free indices
        created_this = 0
        for out_idx, params in zip(free_indices, variants):
            rot, sc, br, beta, noise, occ = params
            aug_name = f"{name}_aug_{out_idx}{ext}"
            aug_path = os.path.join(input_dir, aug_name)
            if os.path.exists(aug_path):
                skipped += 1
                continue

            out = transform_image(img, rotate_deg=rot, scale=sc, brightness=br, beta=beta, noise_sigma=noise, occlude=occ)
            h, w = img.shape[:2]
            if out.shape[0] != h or out.shape[1] != w:
                out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(aug_path, out)
            created += 1
            created_this += 1

        print(f"  {fname}: created {created_this} new variants (existing: {len(existing_indices)})")

    print("\nDone.")
    print(f"  created: {created}")
    print(f"  skipped (already existed): {skipped}")
    print(f"  resources folder: {input_dir}/")
    print("Note: Files use the '_aug_' naming pattern and will be loaded by sign_detector from resources/")

    # create marker file so subsequent launches skip regeneration
    try:
        with open(marker_path, 'w') as f:
            f.write('done')
        print(f"Wrote marker: {marker_path}")
    except Exception:
        pass


if __name__ == '__main__':
    create_augmented_templates()

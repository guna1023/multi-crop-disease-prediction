"""
prepare_dataset.py
-------------------
Splits your raw downloaded dataset into train / validation / test folders
BEFORE any training happens. Doing the split first (by copying whole image
files into separate folders) is what prevents data leakage -- no image
that the model trains on will ever be seen again during validation or
testing.

HOW TO USE
----------
1. Download the Tomato and Corn leaf disease datasets from Kaggle.
2. Put every class folder (e.g. "Tomato___Early_blight", "Corn___Common_rust",
   "Tomato___healthy", etc.) directly inside:

        dataset_raw/

   So it should look like:

        dataset_raw/
            Tomato___Early_blight/
                img1.jpg
                img2.jpg
                ...
            Tomato___Late_blight/
                ...
            Tomato___healthy/
                ...
            Corn___Common_rust/
                ...
            Corn___Gray_leaf_spot/
                ...
            Corn___Northern_Leaf_Blight/
                ...
            Corn___healthy/
                ...

   The folder NAMES are what become your class labels, so don't rename
   them to something else -- keep the crop name at the start.

3. Run:
        python prepare_dataset.py

   This creates:

        dataset_split/
            train/<class_name>/...
            val/<class_name>/...
            test/<class_name>/...

4. Then run train.py.

You can re-run this script safely -- it deletes and recreates
dataset_split/ each time using the same random seed, so the split is
reproducible.
"""

import os
import shutil
import random

# ---------------------------------------------------------------------------
# CONFIG - change these if you like
# ---------------------------------------------------------------------------
RAW_DATASET_DIR = "dataset_raw"
OUTPUT_DIR = "dataset_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # must sum to 1.0 with the two above

SEED = 42
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def get_class_folders(raw_dir):
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"'{raw_dir}' not found. Create it and put your class folders "
            f"inside, e.g. dataset_raw/Tomato___Early_blight/"
        )
    classes = sorted(
        [
            d
            for d in os.listdir(raw_dir)
            if os.path.isdir(os.path.join(raw_dir, d))
        ]
    )
    if not classes:
        raise ValueError(
            f"No class folders found inside '{raw_dir}'. Each disease/crop "
            f"class must be its own subfolder."
        )
    return classes


def split_list(items, train_ratio, val_ratio):
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    return train_items, val_items, test_items


def main():
    assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-6, (
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0"
    )

    random.seed(SEED)

    classes = get_class_folders(RAW_DATASET_DIR)
    print(f"Found {len(classes)} classes in '{RAW_DATASET_DIR}':")
    for c in classes:
        print(f"  - {c}")

    # Fresh output directory every run (reproducible split)
    if os.path.exists(OUTPUT_DIR):
        print(f"\nRemoving existing '{OUTPUT_DIR}' folder to rebuild it...")
        shutil.rmtree(OUTPUT_DIR)

    for split_name in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split_name), exist_ok=True)

    summary_rows = []

    for class_name in classes:
        class_dir = os.path.join(RAW_DATASET_DIR, class_name)
        images = [
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]

        if len(images) < 10:
            print(
                f"WARNING: class '{class_name}' has only {len(images)} images. "
                f"Consider adding more for reliable training."
            )

        random.shuffle(images)  # shuffle BEFORE splitting -> no leakage
        train_imgs, val_imgs, test_imgs = split_list(
            images, TRAIN_RATIO, VAL_RATIO
        )

        for split_name, split_imgs in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]:
            dest_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for img_name in split_imgs:
                src = os.path.join(class_dir, img_name)
                dst = os.path.join(dest_dir, img_name)
                shutil.copy2(src, dst)

        summary_rows.append(
            (class_name, len(images), len(train_imgs), len(val_imgs), len(test_imgs))
        )

    # Print a clean summary table
    print("\n" + "=" * 70)
    print(f"{'Class':40s} {'Total':>7s} {'Train':>7s} {'Val':>6s} {'Test':>6s}")
    print("-" * 70)
    for row in summary_rows:
        print(f"{row[0]:40s} {row[1]:7d} {row[2]:7d} {row[3]:6d} {row[4]:6d}")
    print("=" * 70)
    print(f"\nDone! Split dataset created at: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

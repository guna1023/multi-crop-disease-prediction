"""
utils_common.py
----------------
Shared helper functions used by train.py, evaluate.py, predict.py and app.py.

Main job of this file:
1. Parse a raw class-folder name like "Tomato___Early_blight" into a
   human-readable Crop + Disease pair.
2. Provide short, beginner-friendly disease information for the UI.

This file has NO TensorFlow import so it loads instantly everywhere.
"""

import json
import os


def parse_class_name(raw_class_name: str):
    """
    Convert a folder / class name such as:
        "Tomato___Early_blight"   -> ("Tomato", "Early Blight")
        "Corn___Common_rust"      -> ("Corn", "Common Rust")
        "Tomato___healthy"        -> ("Tomato", "Healthy")

    Works even if the dataset uses a single underscore or a space instead
    of the triple underscore, so it is safe for slightly different
    Kaggle dataset naming conventions.
    """
    raw_class_name = raw_class_name.strip()

    if "___" in raw_class_name:
        crop_part, disease_part = raw_class_name.split("___", 1)
    elif "__" in raw_class_name:
        crop_part, disease_part = raw_class_name.split("__", 1)
    elif "_" in raw_class_name:
        # Fallback: split on the first underscore only
        crop_part, disease_part = raw_class_name.split("_", 1)
    else:
        crop_part, disease_part = raw_class_name, "Unknown"

    crop = crop_part.replace("_", " ").strip().title()
    disease = disease_part.replace("_", " ").strip().title()

    return crop, disease


def load_class_names(class_names_path: str):
    """Load the list of class names saved by train.py."""
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(
            f"Could not find '{class_names_path}'. "
            f"Did you run train.py first? It creates this file automatically."
        )
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    return class_names


# ---------------------------------------------------------------------------
# Short, general-purpose disease information shown in the Streamlit app.
# This is intentionally simple/educational text, NOT medical/agronomic
# advice for real farming decisions.
# ---------------------------------------------------------------------------
DISEASE_INFO = {
    "early blight": (
        "A fungal disease causing dark concentric-ring spots on older leaves. "
        "Often spreads in warm, humid weather. Remove infected leaves and "
        "avoid overhead watering."
    ),
    "late blight": (
        "A serious fungal/oomycete disease causing large dark, water-soaked "
        "patches on leaves and stems. Spreads fast in cool, wet conditions "
        "and can destroy a crop quickly if untreated."
    ),
    "leaf mold": (
        "Caused by fungus that thrives in humid greenhouses. Yellow spots "
        "appear on top of leaves with olive-green mold underneath."
    ),
    "septoria leaf spot": (
        "Small circular spots with dark borders and grey centers on lower "
        "leaves, caused by a fungus. Can cause heavy leaf drop."
    ),
    "bacterial spot": (
        "Small, dark, greasy-looking spots on leaves and fruit caused by "
        "bacteria. Spreads through water splash and contaminated tools."
    ),
    "target spot": (
        "Fungal disease producing brown lesions with concentric rings, "
        "similar to a target/bullseye pattern."
    ),
    "spider mites two spotted spider mite": (
        "Not a disease but a pest infestation. Tiny mites cause stippled, "
        "yellowing leaves and fine webbing."
    ),
    "tomato yellow leaf curl virus": (
        "A viral disease spread by whiteflies, causing upward curling and "
        "yellowing of leaves and stunted growth."
    ),
    "tomato mosaic virus": (
        "A viral disease causing mottled light/dark green patterns and "
        "leaf distortion. Spreads through contact and contaminated tools."
    ),
    "common rust": (
        "A fungal disease showing small, reddish-brown, powdery pustules "
        "on both sides of corn leaves."
    ),
    "gray leaf spot": (
        "A fungal disease causing rectangular grey-to-tan lesions along "
        "corn leaf veins, common in humid climates."
    ),
    "northern leaf blight": (
        "A fungal disease causing long, cigar-shaped grey-green lesions on "
        "corn leaves, which can merge and kill large leaf areas."
    ),
    "healthy": (
        "No visible disease symptoms detected. The leaf appears healthy."
    ),
}


def get_disease_info(disease_name: str) -> str:
    """
    Return a short description for a disease name (case-insensitive).
    Falls back to a generic message if we don't have a description.
    """
    key = disease_name.strip().lower()
    return DISEASE_INFO.get(
        key,
        "No detailed information available for this class yet. "
        "Please consult an agricultural expert for confirmation and treatment.",
    )

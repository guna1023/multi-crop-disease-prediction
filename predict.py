"""
predict.py
----------
Predicts the crop + disease for a SINGLE local leaf image.

USAGE:
    python predict.py --image path/to/leaf.jpg

EXAMPLE:
    python predict.py --image dataset_split/test/Tomato___Early_blight/img001.jpg
"""

import os
import argparse

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

from utils_common import load_class_names, parse_class_name, get_disease_info

MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

IMG_SIZE = (224, 224)


def load_and_preprocess_image(image_path):
    """
    Load an image file and prepare it for the model.
    Preprocessing (MobileNetV2 preprocess_input) is baked INTO the model
    itself (see train.py), so here we only need to:
      1. Open the image safely.
      2. Convert to RGB (handles grayscale / RGBA / palette images).
      3. Resize to the model's expected input size.
      4. Convert to a float32 numpy array with a batch dimension.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    try:
        img = Image.open(image_path)
        img.verify()  # quick sanity check that this is a real image file
        img = Image.open(image_path)  # re-open after verify()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(
            f"'{image_path}' does not look like a valid image file."
        ) from e

    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)

    array = keras.utils.img_to_array(img)  # shape (224, 224, 3), values 0-255
    array = np.expand_dims(array, axis=0)  # shape (1, 224, 224, 3)
    return array


def predict_image(image_path, model=None, class_names=None):
    """
    Run prediction on a single image path.
    model/class_names can be passed in (e.g. from app.py) to avoid
    reloading them for every prediction.
    """
    if model is None:
        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(
                f"'{BEST_MODEL_PATH}' not found. Run train.py first."
            )
        model = keras.models.load_model(BEST_MODEL_PATH)

    if class_names is None:
        class_names = load_class_names(CLASS_NAMES_PATH)

    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)[0]  # shape (num_classes,)

    predicted_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_idx]) * 100
    predicted_class_name = class_names[predicted_idx]

    crop, disease = parse_class_name(predicted_class_name)

    result = {
        "raw_class_name": predicted_class_name,
        "crop": crop,
        "disease": disease,
        "confidence": confidence,
        "all_probabilities": {
            class_names[i]: float(predictions[i]) * 100
            for i in range(len(class_names))
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict disease from a leaf image")
    parser.add_argument("--image", type=str, required=True, help="Path to the leaf image")
    args = parser.parse_args()

    try:
        result = predict_image(args.image)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return

    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Crop      : {result['crop']}")
    print(f"Disease   : {result['disease']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print("\nInfo:")
    print(f"  {get_disease_info(result['disease'])}")

    print("\nTop probabilities:")
    sorted_probs = sorted(
        result["all_probabilities"].items(), key=lambda x: x[1], reverse=True
    )
    for class_name, prob in sorted_probs[:5]:
        print(f"  {class_name:40s}: {prob:.2f}%")


if __name__ == "__main__":
    main()

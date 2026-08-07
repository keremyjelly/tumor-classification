"""Preprocessing, model loading, and inference shared by the Streamlit dashboard.

Preprocessing here mirrors `final_nb_deliverable3_ml.ipynb` exactly:
- Logistic Regression / Scratch CNN expect 128x128 grayscale, LANCZOS-resized, scaled to [0, 1].
- MobileNetV2 expects 128x128 grayscale stacked into 3 channels, left in [0, 255]
  (its `preprocess_input` rescaling is baked into the saved model graph).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (128, 128)


def resize_grayscale(image):
    """128x128 LANCZOS-resized grayscale array, float32 in [0, 255]."""
    return np.asarray(image.convert('L').resize(IMG_SIZE, Image.Resampling.LANCZOS), dtype=np.float32)


def preprocess_for_logistic_regression(image):
    return (resize_grayscale(image) / 255.0).reshape(1, -1)


def preprocess_for_scratch_cnn(image):
    return (resize_grayscale(image) / 255.0)[None, ..., None]


def preprocess_for_mobilenet(image):
    gray = resize_grayscale(image)
    return np.stack([gray, gray, gray], axis=-1)[None, ...]


def _softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def load_models(models_dir):
    """Load whichever exported artifacts exist. Returns (models, metadata, errors).

    Each artifact is loaded independently: a `.keras` file written by a different Keras
    version raises on deserialization, and without isolation one bad file would take the
    whole dashboard down instead of just dropping that model's column.
    """
    models_dir = Path(models_dir)
    models = {}
    errors = {}

    specs = [
        ('Logistic Regression', 'logistic_regression.joblib', 'sklearn'),
        ('Scratch CNN', 'scratch_cnn.keras', 'scratch_cnn'),
        ('MobileNetV2 Transfer', 'mobilenetv2.keras', 'mobilenetv2'),
        ('Modified CNN', 'modified_cnn.keras', 'scratch_cnn'),
    ]

    for name, filename, kind in specs:
        path = models_dir / filename
        if not path.exists():
            continue
        try:
            if kind == 'sklearn':
                import joblib
                model = joblib.load(path)
            else:
                from tensorflow import keras
                model = keras.models.load_model(path)
        except Exception as exc:  # noqa: BLE001 — one bad export must not kill the app
            errors[name] = f'{type(exc).__name__}: {exc}'
            continue
        models[name] = {'kind': kind, 'model': model}

    metadata_path = models_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None

    return models, metadata, errors


def predict_one(entry, image):
    """Return a probability vector over CLASSES for one loaded model entry."""
    kind, model = entry['kind'], entry['model']
    if kind == 'sklearn':
        return model.predict_proba(preprocess_for_logistic_regression(image))[0]
    if kind == 'scratch_cnn':
        return model.predict(preprocess_for_scratch_cnn(image), verbose=0)[0]
    if kind == 'mobilenetv2':
        logits = model.predict(preprocess_for_mobilenet(image), verbose=0)[0]
        return _softmax(logits)
    raise ValueError(f'Unknown model kind: {kind}')


def predict_all(models, image):
    """Run every loaded model on `image`.

    Returns (per_model_probabilities, consensus_probabilities, errors). A model that
    loads but fails at inference — e.g. an sklearn pickle read by a different sklearn
    version — is reported rather than allowed to abort the whole comparison.
    """
    per_model, errors = {}, {}
    for name, entry in models.items():
        try:
            per_model[name] = predict_one(entry, image)
        except Exception as exc:  # noqa: BLE001
            errors[name] = f'{type(exc).__name__}: {exc}'
    consensus = np.mean(list(per_model.values()), axis=0) if per_model else None
    return per_model, consensus, errors


# --- Grad-CAM support -------------------------------------------------------
# Logistic regression is not explainable this way at all: it has no convolutional
# feature maps (its "attention" is just the per-pixel weight vector).
#
# The scratch CNNs are technically supported by `gradcam_for` — the Modified CNN
# produces a 32x32 heatmap — but only MobileNetV2 is surfaced in the dashboard. Its
# attention falls on brain tissue, while the Modified CNN's falls largely on the skull
# margin, image borders, and in at least one sample a source watermark. See the README
# for that finding; it is a result about the model, not a limitation of this code.
GRADCAM_MODELS = ('MobileNetV2 Transfer',)


def preprocess_for_gradcam(entry, image):
    """Return the preprocessed batch matching a model's expected input."""
    if entry['kind'] == 'mobilenetv2':
        return preprocess_for_mobilenet(image)
    return preprocess_for_scratch_cnn(image)


def gradcam_for(entry, image, class_index=None):
    """Compute (heatmap, class_index) for one loaded CNN entry.

    The scratch/modified CNNs end in softmax; the exported MobileNetV2 emits raw
    logits (softmax is applied in `predict_one`), so gradients are taken directly
    on the logit there.
    """
    from gradcam import compute_gradcam

    kind = entry['kind']
    if kind == 'sklearn':
        raise ValueError('Grad-CAM requires a convolutional model.')

    batch = preprocess_for_gradcam(entry, image)
    return compute_gradcam(
        entry['model'],
        batch,
        class_index=class_index,
        ends_in_softmax=(kind == 'scratch_cnn'),
    )

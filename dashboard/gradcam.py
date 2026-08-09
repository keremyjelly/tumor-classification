"""Grad-CAM (Gradient-weighted Class Activation Mapping) for the CNN models.

Grad-CAM weights the final convolutional feature maps by the gradient of 
the target class score with respect to those maps, then averaging over channels:

    w_k = GAP( d y_c / d A_k )          # importance of feature map k for class c
    L_c = ReLU( sum_k w_k * A_k )       # spatially-resolved class evidence

"""
from __future__ import annotations

import numpy as np

# Layer types whose output is a valid Grad-CAM target (4D spatial feature maps).
# Normalization layers are included so we tap the representation that actually feeds
# global average pooling — for the Modified CNN that is GroupNormalization, not Conv2D.
_CONV_TYPES = (
    'Conv2D', 'DepthwiseConv2D', 'SeparableConv2D', 'Add', 'ReLU', 'Activation',
    'BatchNormalization', 'GroupNormalization', 'LayerNormalization',
)


def _iter_layers(model):
    """Yield (layer, parent_model) for every layer, descending into nested models."""
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # nested Model / Sequential
            yield from _iter_layers(layer)
        yield layer, model


def _find_last_conv_layer(model):
    """Return (layer, owning_model) for the deepest 4D conv-like layer in the graph."""
    candidates = []
    for layer, parent in _iter_layers(model):
        cls = layer.__class__.__name__
        try:
            shape = layer.output.shape
        except (AttributeError, ValueError):
            continue
        if len(shape) == 4 and cls in _CONV_TYPES:
            candidates.append((layer, parent))
    if not candidates:
        raise ValueError('No 4D convolutional layer found for Grad-CAM.')
    return candidates[-1]


def _replay_chain(model, tap_layer=None, swap=None):
    """    
    Args:
        tap_layer: layer whose output should be captured as the Grad-CAM target.
        swap: optional (layer, replacement_model) where `replacement_model` returns
            [tap_output, layer_output] — used to reach inside MobileNetV2's nested base.
    """
    from tensorflow import keras

    inp = keras.Input(shape=model.input_shape[1:])
    x = inp
    tap = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        if swap is not None and layer is swap[0]:
            tap, x = swap[1](x)
        else:
            x = layer(x)
            if layer is tap_layer:
                tap = x
    if tap is None:
        raise ValueError('Grad-CAM tap layer was never reached while replaying the graph.')
    return keras.Model(inp, [tap, x])


def _build_grad_model(model):
    """Return a Keras model mapping model input -> (last_conv_activations, predictions).

    Two cases:
    - Flat graph (Modified CNN / Scratch CNN): the conv layer's output tensor already
      belongs to the top-level graph, so we can tap it directly.
    - Nested graph (MobileNetV2): the tap point lives inside the ImageNet base, whose
      tensors are not part of the outer graph. We build an inner model exposing both the
      tap and the base's own output, then replay the outer chain around it.
    """
    from tensorflow import keras

    conv_layer, owner = _find_last_conv_layer(model)

    if owner is model:
        try:
            return keras.Model(model.inputs, [conv_layer.output, model.output])
        except (AttributeError, ValueError):
            # Sequential models loaded from .keras have no materialized output tensors
            # until called, so replay the chain to create them.
            return _replay_chain(model, tap_layer=conv_layer)

    if owner not in model.layers:
        raise ValueError(
            f'Grad-CAM target lives in {owner.name}, which is not a direct child of the '
            'top-level model. Deeper nesting is not supported.'
        )

    if tuple(owner.output.shape) != tuple(conv_layer.output.shape):
        raise ValueError(
            f"Grad-CAM target '{conv_layer.name}' is nested inside '{owner.name}' but is "
            "not that sub-model's output. Tapping it would require rebuilding the outer "
            'graph, which risks dropping non-layer preprocessing ops.'
        )

    layers = [l for l in model.layers if not isinstance(l, keras.layers.InputLayer)]
    successor = layers[layers.index(owner) + 1]
    tap_tensor = successor.input

    if tuple(tap_tensor.shape) != tuple(conv_layer.output.shape):
        raise ValueError(
            f"Resolved tap tensor {tuple(tap_tensor.shape)} does not match the target "
            f'activation shape {tuple(conv_layer.output.shape)}.'
        )
    return keras.Model(model.inputs, [tap_tensor, model.output])


_GRAD_MODEL_CACHE = {}


def _get_grad_model(model):
    """Cache the rebuilt graph per model instance — rebuilding it on every interaction
    is the dominant cost of rendering Grad-CAM in a Streamlit rerun loop."""
    key = id(model)
    if key not in _GRAD_MODEL_CACHE:
        _GRAD_MODEL_CACHE[key] = _build_grad_model(model)
    return _GRAD_MODEL_CACHE[key]


def compute_gradcam(model, input_array, class_index=None, ends_in_softmax=True):
    """Compute a normalized Grad-CAM heatmap.

    Args:
        model: loaded Keras model.
        input_array: preprocessed batch of shape (1, H, W, C) matching the model's input.
        class_index: target class; defaults to the model's own predicted class.
        ends_in_softmax: True if the model's final activation is softmax (scratch CNNs),
            False if it emits raw logits (MobileNetV2 export).

    Returns:
        (heatmap, class_index) where heatmap is float32 in [0, 1] at feature-map resolution.
    """
    import tensorflow as tf

    # Take the reference prediction 
    reference = model.predict(input_array, verbose=0)

    grad_model = _get_grad_model(model)
    x = tf.convert_to_tensor(input_array, dtype=tf.float32)

    rebuilt = grad_model(x, training=False)[1].numpy()
    if not np.allclose(reference, rebuilt, atol=1e-4):
        raise RuntimeError(
            'Grad-CAM graph rebuild does not reproduce the model output '
            f'(max abs diff {np.abs(reference - rebuilt).max():.4g}). '
            'Refusing to return a heatmap that would explain a different graph.'
        )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        tape.watch(conv_out)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[0, class_index]
        if ends_in_softmax:
            score = tf.math.log(tf.clip_by_value(score, 1e-12, 1.0))

    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise ValueError('Gradient is None — the conv layer is not connected to the output.')

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))          # GAP over spatial dims
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)       # weighted channel sum
    cam = tf.nn.relu(cam)

    cam = cam.numpy().astype(np.float32)
    peak = cam.max()
    if peak > 0:
        cam /= peak
    return cam, int(class_index)


def upsample_heatmap(cam, size):
    """Bilinearly resize a (h, w) heatmap to (H, W) using PIL (no OpenCV dependency)."""
    from PIL import Image

    img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8))
    img = img.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


_JET_STOPS = np.array([0.0, 0.125, 0.375, 0.625, 0.875, 1.0], dtype=np.float32)
_JET_RGB = np.array(
    [[0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0],
     [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
    dtype=np.float32,
)


def _jet(values):
    """Map an array in [0, 1] to jet RGB via piecewise-linear interpolation."""
    v = np.clip(values, 0.0, 1.0)
    return np.stack([np.interp(v, _JET_STOPS, _JET_RGB[:, c]) for c in range(3)], axis=-1)


def overlay_heatmap(gray_image, cam, alpha=0.45, threshold=0.25):
    """Blend a Grad-CAM heatmap over a grayscale MRI. Returns a uint8 RGB array.
    """
    gray = np.asarray(gray_image, dtype=np.float32)
    if gray.max() > 1.0:
        gray = gray / 255.0
    base = np.stack([gray] * 3, axis=-1)

    heat = upsample_heatmap(cam, gray.shape[:2])
    colored = _jet(heat).astype(np.float32)

    # Fade the overlay in proportion to activation, and hide weak activations entirely.
    mask = np.clip((heat - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)[..., None]
    blended = base * (1 - alpha * mask) + colored * (alpha * mask)
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)

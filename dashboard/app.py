"""Educational dashboard for the Brain MRI Tumor Classification project.

Run with: streamlit run dashboard/app.py
Ships with the final, committed model weights in models/ (*.joblib, *.keras, metadata.json) —
exported once by Section 19 of final_nb_deliverable3_ml.ipynb and tracked in git, so the
dashboard does not depend on re-running the notebook.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from gradcam import overlay_heatmap
from model_utils import (
    CLASSES,
    GRADCAM_MODELS,
    gradcam_for,
    load_models,
    predict_all,
    resize_grayscale,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / 'models'
DATA_TEST_DIR = REPO_ROOT / 'data' / 'Testing'
# data/ is gitignored, so the deployed app falls back to this small committed sample set.
SAMPLES_DIR = Path(__file__).resolve().parent / 'samples'
CONTRIBUTORS = "Jeremy Kelly, Marco Basile, Tyler Asmussen, Odon Ineza"

st.set_page_config(page_title="Brain MRI Tumor Classification — Benchmark Explorer", layout="wide")

st.markdown(
    """
    <style>
    .disclaimer-banner {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 999999;
        background-color: #d4a017;
        color: #1c1f26;
        text-align: center;
        font-weight: 700;
        padding: 0.5rem 1rem;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .block-container { padding-top: 4rem; }

    .prob-card {
        background: #1c1f26;
        border: 1px solid #30343c;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        height: 100%;
    }
    .prob-card-title {
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 0.6rem;
        color: #e6e6e6;
    }
    .prob-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.35rem 0;
        font-size: 0.8rem;
        color: #b8bcc4;
    }
    .prob-row.predicted {
        color: #e6e6e6;
        font-weight: 700;
    }
    .prob-label { width: 84px; flex-shrink: 0; }
    .prob-bar {
        flex: 1;
        background: #30343c;
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }
    .prob-fill { background: #5b8def; height: 100%; }
    .prob-row.predicted .prob-fill { background: #2ECC71; }
    .prob-value { width: 36px; text-align: right; flex-shrink: 0; }

    .consensus-card {
        background: #1c1f26;
        border: 1px solid #d4a017;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin-top: 0.5rem;
    }
    .consensus-title {
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.7rem;
        color: #d4a017;
    }
    .consensus-card .prob-row { font-size: 0.9rem; }
    .consensus-card .prob-label { width: 110px; }
    </style>
    <div class="disclaimer-banner">
        ⚠️ Educational prototype only — not for medical or diagnostic use ⚠️
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Brain MRI Tumor Classification — Benchmark Explorer")
st.caption(f"Contributors: {CONTRIBUTORS}")
st.caption(
    "Dataset: Brain Tumor MRI Dataset (Kaggle, NickParvar 2021) — merges Figshare, SARTAJ, "
    "and Br35H sources. Classes: glioma, meningioma, notumor, pituitary."
)

@st.cache_resource(show_spinner="Loading trained models…")
def get_models(models_dir):
    """Load once per server process. Streamlit reruns the whole script on every widget
    interaction, so without this the 24 MB MobileNetV2 export is re-read each time."""
    return load_models(models_dir)


models, metadata, load_errors = get_models(MODELS_DIR)

for _name, _err in load_errors.items():
    st.warning(
        f"**{_name}** could not be loaded and is omitted below — {_err}\n\n"
        "This usually means the deployed Keras version differs from the one that "
        "exported `models/*.keras` (pinned in `requirements.txt`)."
    )


def build_sample_options():
    """Sample images from the local test split when present, else the committed samples."""
    options = {"None": None}
    if DATA_TEST_DIR.exists():
        for c in CLASSES:
            folder = DATA_TEST_DIR / c
            if not folder.exists():
                continue
            for i, f in enumerate(sorted(folder.iterdir())[:2], start=1):
                options[f"Sample {c.title()} {i:02d}"] = f
    elif SAMPLES_DIR.exists():
        for f in sorted(SAMPLES_DIR.glob('*.jpg')):
            cls, _, idx = f.stem.rpartition('_')
            options[f"Sample {cls.title()} {idx}"] = f
    return options


def render_prob_card(title, classes, probs, highlight=True):
    top_idx = int(np.argmax(probs)) if highlight else -1
    rows = []
    for i, c in enumerate(classes):
        row_class = "prob-row predicted" if i == top_idx else "prob-row"
        pct = probs[i] * 100
        rows.append(
            f'<div class="{row_class}"><span class="prob-label">{c.title()}</span>'
            f'<div class="prob-bar"><div class="prob-fill" style="width:{pct:.1f}%"></div></div>'
            f'<span class="prob-value">{pct:.0f}%</span></div>'
        )
    return f'<div class="prob-card"><div class="prob-card-title">{title}</div>{"".join(rows)}</div>'


sample_options = build_sample_options()

col_input, col_preview = st.columns(2)

with col_input:
    with st.container(border=True):
        st.markdown("#### Input selection")
        uploaded = st.file_uploader("Upload custom MRI image (.jpg / .png)", type=["jpg", "jpeg", "png"])
        sample_choice = st.selectbox("Or select a sample validation image", list(sample_options.keys()))
        st.button("Run inference", type="primary", width='stretch')

image = None
if uploaded is not None:
    image = Image.open(uploaded)
elif sample_options.get(sample_choice) is not None:
    image = Image.open(sample_options[sample_choice])

with col_preview:
    with st.container(border=True):
        st.markdown("#### Preprocessed view")
        if image is not None:
            gray = resize_grayscale(image)
            st.image(gray.astype(np.uint8), caption="128 x 128 - Grayscale - Normalized [0, 1]", width='stretch')
        else:
            st.info("No image selected yet. Upload a slice or choose a sample on the left.")

st.markdown("### Multi-model comparative inference probabilities")

MODEL_CARDS = [
    ("Logistic Regression", "2", "Logistic Regression"),
    ("Scratch CNN", "3", "Scratch CNN (3 conv blocks)"),
    ("MobileNetV2 Transfer", "4", "MobileNetV2 (transfer learning)"),
    ("Modified CNN", "5", "Modified CNN"),
]

if not models:
    st.warning(
        "No trained models found in `models/`. Run **Section 19 — Export Models for "
        "Dashboard** in `final_nb_deliverable3_ml.ipynb`, then refresh this page."
    )
elif image is None:
    st.info("Upload an image or pick a sample above to run inference.")
else:
    per_model, consensus, predict_errors = predict_all(models, image)
    for _name, _err in predict_errors.items():
        st.warning(f"**{_name}** failed during inference and is excluded — {_err}")

    cols = st.columns(len(MODEL_CARDS) + 1)
    with cols[0]:
        st.markdown(render_prob_card("1. Majority Class Baseline", CLASSES, np.full(4, 0.25), highlight=False), unsafe_allow_html=True)
    for col, (name, number, title) in zip(cols[1:], MODEL_CARDS):
        with col:
            if name in per_model:
                st.markdown(render_prob_card(f"{number}. {title}", CLASSES, per_model[name]), unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prob-card"><div class="prob-card-title">{number}. {title}</div>'
                             f'<div class="prob-row">Not available — export missing</div></div>', unsafe_allow_html=True)

    if consensus is not None:
        consensus_class = CLASSES[int(np.argmax(consensus))]
        agree = sum(1 for p in per_model.values() if CLASSES[int(np.argmax(p))] == consensus_class)
        consensus_html = render_prob_card(f"Consensus — average across {len(per_model)} models", CLASSES, consensus)
        consensus_html = consensus_html.replace('class="prob-card"', 'class="prob-card consensus-card"').replace('class="prob-card-title"', 'class="consensus-title"')
        st.markdown(consensus_html, unsafe_allow_html=True)
        st.caption(
            f"Consensus class: **{consensus_class.title()}** ({consensus.max():.0%} average confidence) — "
            f"{agree}/{len(per_model)} models agree."
        )

st.markdown("### Model attention — Grad-CAM")
st.caption(
    "Grad-CAM weights MobileNetV2's final convolutional feature map by the gradient of the "
    "class score with respect to that map, so warm regions are the evidence the model "
    "actually used for the selected class. Shown for MobileNetV2, the top-performing model."
)

available_cam = [n for n in GRADCAM_MODELS if n in models]

if not available_cam:
    st.info("Grad-CAM requires the MobileNetV2 export (`models/mobilenetv2.keras`).")
elif image is None:
    st.info("Upload an image or pick a sample above to see where the model is looking.")
else:
    cam_target = st.radio(
        "Explain which class?",
        ["Model's own prediction"] + [c.title() for c in CLASSES],
        horizontal=True,
        help="Pick a specific class to see the evidence for that class even when the "
             "model predicted something else — useful for inspecting confusions.",
    )
    target_idx = None if cam_target.startswith("Model's") else CLASSES.index(cam_target.lower())

    gray = resize_grayscale(image)
    cam_cols = st.columns(len(available_cam) + 1)

    with cam_cols[0]:
        st.image(gray.astype(np.uint8), caption="Input (128 x 128 grayscale)", width='stretch')

    for col, name in zip(cam_cols[1:], available_cam):
        with col:
            try:
                cam, class_idx = gradcam_for(models[name], image, class_index=target_idx)
                overlay = overlay_heatmap(gray, cam)
                res = f"{cam.shape[0]}x{cam.shape[1]}"
                st.image(
                    overlay,
                    caption=f"{name} — evidence for {CLASSES[class_idx].title()} "
                            f"(feature map {res})",
                    width='stretch',
                )
            except Exception as exc:  # noqa: BLE001 — surface, never fake a heatmap
                st.warning(f"{name}: Grad-CAM unavailable — {exc}")

    st.caption(
        "**Reading this honestly.** MobileNetV2's final feature map is only 4x4 at this "
        "input size, so the heatmap is necessarily coarse — it indicates a broad region of "
        "evidence, not a tumor boundary, and should not be read as localization. It shows "
        "which part of the slice drove the decision, not where the tumor is."
    )

st.markdown("### Error analysis & system metadata")
with st.container(border=True):
    st.markdown(
        "- **Image provenance:** Figshare / SARTAJ / Br35H slices distributed via the Kaggle "
        "Brain Tumor MRI Dataset (NickParvar, 2021).\n"
        "- **Known confusion pattern:** across all four trained models, the largest source of "
        "error is glioma vs. meningioma slices (see Section 12 confusion matrices) — visually "
        "similar tumor locations and intensities.\n"
        "- **Generalization:** no patient IDs are available, so results reflect image-level "
        "validation performance only, not verified patient-level generalization (Section 17)."
    )
    if metadata and metadata.get('validation_metrics'):
        st.markdown("**Validation metrics (side-by-side model comparison):**")
        st.dataframe(pd.DataFrame(metadata['validation_metrics']).T)

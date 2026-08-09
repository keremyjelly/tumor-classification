# Tumor Classification

Brain tumor MRI classification project comparing several models — Logistic Regression,
a scratch-built CNN, a modified CNN, and a MobileNetV2 transfer-learning model — on
classifying MRI slices into **glioma**, **meningioma**, **pituitary**, or **no tumor**.

**Live app:** [mle-tumor-classification.streamlit.app](https://mle-tumor-classification.streamlit.app)

**Authors:** Jeremy Kelly, Marco Basile, Tyler Asmussen, Oden Ineza

## Project Status

All four models are trained and exported to `models/`. Validation results (128x128 input):

| Model                  | Accuracy | Val Loss | Misclassified |
|-------------------------|----------|----------|----------------|
| MobileNetV2 Transfer   | 91.4%    | 0.267    | 96 / 1120      |
| Modified CNN            | 90.7%    | 0.249    | 104 / 1120     |
| Scratch CNN             | 87.4%    | 0.356    | 141 / 1120     |
| Logistic Regression     | 84.6%    | 1.176    | 173 / 1120     |

Full analysis, training curves, confusion matrices, and visualizations are in
`final_nb_deliverable3_ml.ipynb` and `outputs/`.

## Features

- **Four-model comparison** — Logistic Regression, a scratch-built CNN, a modified CNN,
  and MobileNetV2 (transfer learning), trained on the same 128x128 MRI slices and
  benchmarked head-to-head (see Project Status above).
- **Interactive Streamlit dashboard** (`dashboard/app.py`, live at
  [mle-tumor-classification.streamlit.app](https://mle-tumor-classification.streamlit.app)) —
  upload an MRI slice, or pick one of the built-in sample images, and see all four
  models' predictions side by side.
- **Grad-CAM explainability** — heatmaps showing where MobileNetV2 draws its evidence,
  with a class selector to inspect evidence for any class, not just the predicted one.
  See [Grad-CAM — where the models look](#grad-cam--where-the-models-look) below.
- **Verified heatmaps** — Grad-CAM output is checked against the original model's
  prediction before display, so a silently broken graph rebuild can't produce a
  plausible-looking but wrong explanation.
- **Sample images included** — eight committed slices (two per class) in
  `dashboard/samples/` so the dashboard works even without the full dataset present,
  which is what the deployed app uses.

## Setup

Requires Python 3.11 or 3.12 (`python3 --version`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt                    # dashboard only
pip install -r requirements-notebook.txt           # add this to run the notebook
```

Deactivate the environment later with `deactivate`.

`requirements.txt` is deliberately limited to what the dashboard needs, because
Streamlit Community Cloud installs that file on every deploy. Notebook-only packages
(`kagglehub`, `seaborn`, `opencv-python`, `matplotlib`, `ipywidgets`) live in
`requirements-notebook.txt`.

The pins in `requirements.txt` matter. The exported `.keras` files record the
serialization format of the Keras that wrote them (3.14.1), and `logistic_regression.joblib`
records its scikit-learn version (1.9.0). Loading either with a different version fails.
If you re-export the models from a newer environment, update the pins to match.

## Running the Dashboard

The dashboard is an **educational prototype** — not for diagnostic use.

The trained models are already committed in `models/`, so you can run the dashboard
directly:

```bash
streamlit run dashboard/app.py
```

If you retrain or change the models, regenerate them by running
`final_nb_deliverable3_ml.ipynb` end-to-end (with `RUN_CNN=True`), including
**Section 18 — Export Models for Dashboard**, which writes
`models/logistic_regression.joblib`, `models/scratch_cnn.keras`,
`models/mobilenetv2.keras`, and `models/metadata.json`.

`data/` is gitignored, so the sample-image dropdown falls back to the eight committed
slices in `dashboard/samples/` (two per class) when the full test split is not present.
That is what the deployed app uses.

## Grad-CAM — where the models look

The dashboard renders Grad-CAM heatmaps for **MobileNetV2**, the top-performing model.
Grad-CAM weights the final convolutional feature map by the gradient of a class score
with respect to that map, so warm regions mark the evidence the model actually used.
Logistic regression is excluded because it has no feature maps at all.

Implementation lives in `dashboard/gradcam.py`. Three details are worth knowing:

MobileNetV2's final feature map is 4x4 at 128x128 input, so its heatmaps mark a broad
region and cannot localize a tumor boundary. They show which part of the slice drove the
decision, not where the tumor is.

`gradcam_for` also supports the scratch CNNs — the Modified CNN yields a much sharper
32x32 map — but they are not surfaced in the dashboard. See the finding below for why.

Every heatmap is verified before it is displayed. Grad-CAM needs a re-wired copy of the
model graph, and a rebuild that silently drops a layer still produces a plausible-looking
heatmap for a model that was never evaluated. `compute_gradcam` therefore checks the
rebuilt graph reproduces the original model's output and raises if it does not. This
caught a real bug during development: the MobileNetV2 export applies `preprocess_input`
as raw graph ops that Keras does not expose in `model.layers`, so rebuilding by replaying
layers dropped the normalization entirely.

You can also select a specific class to explain, which shows the evidence for that class
even when the model predicted something else — useful for inspecting the recurring
glioma/meningioma confusion.

### Finding: the Modified CNN attends to non-anatomical features

Grad-CAM on the Modified CNN placed its evidence largely on image borders, the skull
margin, and background rather than on brain tissue. On a glioma slice with an obvious
central lesion, its evidence sat at the bottom-right edge and it predicted meningioma. On
a `notumor` sample carrying a source watermark, its evidence fell on the watermark itself.

This is the signature of shortcut learning: the model may be separating classes partly by
acquisition and source artifacts that correlate with class in this dataset, rather than by
pathology. It qualifies the Modified CNN's 90.7% validation accuracy — that number is real
on this data, but it does not establish that the model learned tumor morphology, and it
would not be expected to transfer to images from a different source pipeline.

MobileNetV2's attention, by contrast, concentrates on brain parenchyma. That is why it is
the model surfaced in the dashboard. The Modified CNN remains in the quantitative
comparison, where its accuracy is reported as measured.

This was not investigated further because the dataset provides no acquisition metadata or
patient IDs to test the hypothesis directly. A controlled check would be to retrain on
center-cropped, border-stripped images and see whether the accuracy gap persists.

## Contributing

```bash
git pull origin main
git checkout -b feature/your-name-task
# make changes
git add .
git commit -m "Add preprocessing pipeline"
git push origin feature/your-name-task
# open a PR on GitHub
```

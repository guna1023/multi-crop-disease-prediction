# AI-Based Multi-Crop Disease Detection Using Deep Learning

A final-year project that detects **Tomato** and **Corn (Maize)** leaf
diseases from images using **MobileNetV2 transfer learning**, with a local
**Streamlit** web app for predictions. Runs 100% locally after setup — no
cloud APIs, no internet needed for training/prediction.

---

## 1. Project Structure

```
multi_crop_disease/
│
├── dataset_raw/              <- YOU put downloaded Kaggle classes here
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   ├── Tomato___healthy/
│   ├── Corn___Common_rust/
│   ├── Corn___Gray_leaf_spot/
│   ├── Corn___Northern_Leaf_Blight/
│   └── Corn___healthy/
│
├── dataset_split/             <- auto-created by prepare_dataset.py
│   ├── train/<class>/...
│   ├── val/<class>/...
│   └── test/<class>/...
│
├── models/
│   ├── best_model.keras        <- created by train.py
│   ├── last_checkpoint.keras   <- created by train.py (for resuming)
│   └── class_names.json        <- created by train.py
│
├── results/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
├── utils_common.py     <- shared helper functions (class name parsing, disease info)
├── prepare_dataset.py  <- splits raw data into train/val/test (run 1st)
├── train.py             <- trains the model (run 2nd)
├── evaluate.py          <- evaluates on test set (run 3rd)
├── predict.py           <- predicts a single image from the terminal
├── app.py                <- Streamlit web app
├── requirements.txt
└── README.md
```

---

## 2. Where to Download the Datasets

Use Kaggle (search for these, several mirrors exist — pick ones that use
the standard `Crop___Disease` folder naming, based on the PlantVillage
dataset):

- **Tomato leaf disease dataset** — search Kaggle for
  `"Tomato Leaf Disease Dataset"` or `"PlantVillage Tomato"`.
- **Corn / Maize leaf disease dataset** — search Kaggle for
  `"Corn or Maize Leaf Disease Dataset"`.

You need a free Kaggle account to download (`kaggle.com`). Download the
ZIP files manually through the browser — this project does **not**
auto-download anything, exactly as required.

---

## 3. How to Arrange the Datasets

1. Unzip both downloads on your computer.
2. Inside, you'll find folders per class, e.g.:
   - `Tomato___Early_blight`, `Tomato___Late_blight`, `Tomato___healthy`, ...
   - `Corn___Common_rust`, `Corn___Gray_leaf_spot`, `Corn___Northern_Leaf_Blight`, `Corn___healthy`, ...
   (exact names vary slightly by dataset version — that's fine, folder
   names automatically become your class labels.)
3. Copy **every one of those class folders** (both tomato and corn)
   directly inside the project's `dataset_raw/` folder, so they sit
   side-by-side — do **not** nest them inside extra "Tomato" / "Corn"
   parent folders. Example:

```
dataset_raw/
    Tomato___Early_blight/
        img1.jpg
        img2.jpg
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
```

4. Delete any non-image files, README files, or duplicate nested folders
   that came inside the Kaggle zip.

---

## 4. How to Verify the Folder Structure

From inside the project folder, run this quick check (Windows PowerShell
or CMD):

```powershell
dir dataset_raw
```

You should see one folder per class (7 folders total for the example
above), each containing image files directly (not another subfolder).
To check inside one class:

```powershell
dir dataset_raw\Tomato___Early_blight
```

You should see `.jpg` / `.JPG` / `.png` files listed, not more folders.

---

## 5. Install Python and Dependencies (Windows + VS Code)

### Step 5.1 — Install Python
1. Download Python 3.10 or 3.11 from https://www.python.org/downloads/
   (TensorFlow 2.16 supports Python 3.9–3.11 — avoid 3.12+ for now).
2. During install, check **"Add Python to PATH"**.
3. Verify in a terminal:
   ```powershell
   python --version
   ```

### Step 5.2 — Open the project in VS Code
1. Open VS Code → File → Open Folder → select `multi_crop_disease`.
2. Open a terminal inside VS Code: `Terminal > New Terminal`.

### Step 5.3 — Create and activate a virtual environment
```powershell
python -m venv venv
venv\Scripts\activate
```
(You should now see `(venv)` at the start of your terminal prompt.)

### Step 5.4 — Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

This is also the step where MobileNetV2's ImageNet pretrained weights get
cached locally the **first time** you run training (downloaded
automatically by Keras into `~/.keras/models/`). After that first
download, everything works fully offline.

> **Note (GPU users):** Native Windows only supports TensorFlow GPU up to
> version 2.10. If you have an NVIDIA GPU and want GPU acceleration on
> Windows, either (a) use TensorFlow via **WSL2 (Ubuntu)**, or (b) just
> train on CPU — with ~2000–5000 images per class and MobileNetV2 (a
> lightweight model), CPU training is slow but perfectly workable for a
> final-year project.

---

## 6. Step-by-Step: From Empty Folder to Working Web App

```powershell
# 1. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Put your downloaded Kaggle class folders inside dataset_raw/
#    (see Section 3 above)

# 4. Split the dataset into train/val/test (no data leakage)
python prepare_dataset.py

# 5. Train the model (Stage 1 head training + Stage 2 fine-tuning)
python train.py

# 6. Evaluate on the untouched test set
python evaluate.py

# 7. Test a single image from the terminal
python predict.py --image dataset_split/test/Tomato___Early_blight/some_image.jpg

# 8. Launch the local web app
streamlit run app.py
```

Streamlit will open automatically in your browser at
`http://localhost:8501`. Upload a leaf photo, click **Predict**, and it
will show the crop, disease, and confidence.

---

## 7. Training Details

- `python train.py` uses sensible defaults (15 epochs head training + 10
  epochs fine-tuning). Override with:
  ```powershell
  python train.py --epochs 20 --fine_tune_epochs 15
  ```
- `EarlyStopping` will stop training early if validation loss stops
  improving, and automatically restores the best weights.
- `ReduceLROnPlateau` lowers the learning rate automatically if progress
  stalls.
- Only the **best** model (by validation accuracy) is kept at
  `models/best_model.keras`.

### Stopping and Resuming Training Safely
Every epoch, in addition to the best model, the full model state is saved
to `models/last_checkpoint.keras`. If you need to stop training (close
the laptop, Ctrl+C, power cut):

```powershell
python train.py --resume
```

This picks up from `last_checkpoint.keras` instead of restarting from
ImageNet weights. (Note: it resumes the *model weights*, not the exact
epoch counter — for a final-year project this is normally sufficient;
just keep an eye on the printed validation accuracy to judge progress.)

---

## 8. Evaluation Output

After `python evaluate.py`, check:
- Terminal output: accuracy, precision, recall, F1-score.
- `results/classification_report.txt` — full per-class report.
- `results/confusion_matrix.png` — visual confusion matrix.

---

## 9. Common Issues

| Problem | Fix |
|---|---|
| `FileNotFoundError: dataset_raw not found` | Create the folder and add class subfolders (Section 3). |
| Very low validation accuracy | Add more images per class, check for mislabeled folders, train more epochs. |
| `ModuleNotFoundError` | Make sure the venv is activated (`venv\Scripts\activate`) before running any script. |
| Streamlit says "No trained model found" | Run `python train.py` first — it must complete at least Stage 1 to create `best_model.keras`. |
| Training very slow | Expected on CPU-only Windows laptops; reduce epochs, or use a smaller subset per class while testing your pipeline. |

---

## 10. Notes on Offline Behaviour

- After `pip install -r requirements.txt` and the **first** run of
  `train.py` (which downloads MobileNetV2's ImageNet weights once),
  no further internet access is required for training, evaluation,
  prediction, or the Streamlit app.
- All data, models, and results stay on your local machine.
#   m u l t i - c r o p - d i s e a s e - d e t e c t i o n  
 
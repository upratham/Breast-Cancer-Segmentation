# Medical Image Segmentation (U-Net / DeepLabV3+ / MultiResUNet)

This project trains and evaluates deep-learning models for **binary image segmentation** (e.g., separating a target structure from the background) using:
- **U-Net**
- **DeepLabV3+**
- **MultiResUNet**

The workflow is notebook-driven:
- `data_preprocessing.ipynb` – prepare images/masks for training
- `train.ipynb` – train a segmentation model (dice/IoU metrics)
- `eval.ipynb` – evaluate the trained model and export/visualize predictions

---

## Project structure

> (Folders like `src/` are referenced by the notebooks.)

```
.
├─ data_preprocessing.ipynb
├─ train.ipynb
├─ eval.ipynb
├─ requirements.txt
├─ src/
│  ├─ model_unet.py
│  ├─ model2_DeeplabV3.py
│  ├─ model3_MultiResUNET.py
│  ├─ metrics.py
│  └─ ...
└─ (data)/
   ├─ train
   ├─ test
   └─ valid
```

---

## Setup

### 1) Create & activate an environment (Python 3.9 recommended)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contains core scientific + CV dependencies (NumPy/Pandas/OpenCV/Albumentations, etc.). fileciteturn0file0

> **Note:** Install your preferred deep-learning backend:
> - TensorFlow (often easiest for these notebooks): `pip install tensorflow`
> - OR PyTorch if you adapt the code accordingly

---

## How to run

### Step A — Data preprocessing
Open and run:

```bash
jupyter notebook data_preprocessing.ipynb
```

Typical outputs:
- resized/normalized images
- aligned binary masks
- train/val/test splits (if created in the notebook)

### Step B — Training
Open and run:

```bash
jupyter notebook train.ipynb
```

What it does (high-level):
- loads prepared image/mask pairs
- builds a model (U-Net / DeepLabV3+ / MultiResUNet)
- trains using **Dice loss**, tracks **Dice coefficient** and **IoU**
- saves best checkpoints (via callbacks like `ModelCheckpoint`)

### Step C — Evaluation / inference
Open and run:

```bash
jupyter notebook eval.ipynb
```

Typical outputs:
- prediction masks
- side-by-side visualizations (image / ground-truth / prediction)
- summary metrics (Dice, IoU)

---

## Configuration tips

Common things you may want to change inside the notebooks:
- dataset paths (`images/`, `masks/`)
- image size (H, W)
- batch size / epochs / learning rate
- model choice (U-Net vs DeepLabV3+ vs MultiResUNet)
- augmentation settings (Albumentations)

---

## Result sample

An example visualization (input image + segmentation mask/prediction):


![Result sample](result%20sample.png)


---

## Troubleshooting

- **ModuleNotFoundError: src...**  
  Ensure the `src/` folder exists and is on your Python path. If running from notebooks, keeping `src/` in the project root is usually enough.

- **TensorFlow / CUDA issues**  
  Try CPU-only first, then match CUDA/cuDNN versions to your TensorFlow build.



## License
Add your preferred license here (MIT/Apache-2.0/etc.).
---
## Author
  - Maintained by **Prathamesh Uravane**  
  - Email: [upratham2002@gmail.com](mailto:upratham2002@gmail.com)
---
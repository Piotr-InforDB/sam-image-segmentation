# SAM Segmentation + Roof Classification

This repo provides a simple pipeline to:

- Generate segmentation proposals from images using Meta's Segment Anything Model (SAM).
- Train a lightweight CNN to classify cropped segments as `roof` vs `other`.
- Combine predicted roof segments into a single mask per image and (optionally) compute IoU against ground truth.

Key scripts:

- `sam.py` — runs SAM on images to produce per-image PNG cutouts with alpha masks.
- `train.py` — trains a small CNN classifier on `roof` vs `other` crops.
- `run.py` — classifies a folder of segments using a trained model.
- `iou.py` — generates combined roof masks from SAM+classifier and computes IoU.


## Requirements

- Python 3.9+
- PyTorch (with CUDA if available)
- torchvision
- segment-anything (SAM)
- opencv-python
- numpy
- Pillow
- matplotlib
- tqdm

Install, for example:

```
pip install torch torchvision opencv-python pillow numpy matplotlib tqdm
# Install SAM (from the official repo)
# pip install git+https://github.com/facebookresearch/segment-anything.git
```

Note: You also need a SAM checkpoint file. See the next section.


## SAM Weights

- Default config expects the SAM-H checkpoint at `sam_models/sam_vit_h_4b8939.pth`.
- Create the folder and place the weights there, or update the path in:
  - `sam.py` (`CHECKPOINT_PATH`)
  - `iou.py` (`SAM_MODEL`)

```
mkdir -p sam_models
# Place sam_vit_h_4b8939.pth into sam_models/
```


## Data Layout

- For running SAM cutouts:
  - Put input images in `images/` (e.g., `.jpg`, `.png`).
  - Outputs go to `cutouts/<image_stem>/cutout_XXX.png`.

- For training the classifier (`train.py`):
  - Create `datasets/roof/` and `datasets/other/` with your labeled crops.
  - The script builds a temporary resized split under `temp_dataset/`.

- For classification (`run.py`):
  - By default it reads from `segments/`. You can point it to any folder of crops by editing `IMAGE_FOLDER` in the script.

- For IoU evaluation (`iou.py`):
  - Put source images in `iou/images/`.
  - Put ground-truth masks in `iou/masks/` (PNG). Red-tinted GT or single-channel binary is supported.
  - Predicted combined masks are saved to `iou/output/`.


## Usage

1) Generate segment cutouts with SAM

```
python sam.py
```

- Configure model type and sampling density in `sam.py`:
  - `MODEL_TYPE = "vit_h"`
  - `points_per_side` in `SamAutomaticMaskGenerator` (default 8).
- Input images: `images/`
- Outputs: `cutouts/<image_stem>/cutout_XXX.png` (RGBA with alpha from the mask)

2) Train the roof/other classifier

```
python train.py
```

- Prepare `datasets/roof/` and `datasets/other/` with training images.
- The script creates `temp_dataset/` with train/val splits and heavy augmentations.
- Model hyperparameters are defined at the top of `train.py` (e.g., `IMG_SIZE`, `BATCH_SIZE`, `EPOCHS`, `LR`).
- The intended checkpoint output path is `best_model.pth` (`MODEL_PATH`). If saving is not yet implemented in your copy, add a `torch.save(model.state_dict(), MODEL_PATH)` when validation improves.

3) Classify a folder of segments

```
python run.py
```

- Expects a trained checkpoint at `best_model.pth` (edit `MODEL_PATH` if needed).
- Reads images from `segments/` by default (edit `IMAGE_FOLDER` or move files accordingly).
- Shows a tiled visualization and writes `classification_results.txt` with `Filename,Class,Confidence`.

Tip: If you generated cutouts with `sam.py`, either:
- Copy/merge relevant cutouts into `segments/`, or
- Edit `run.py` and set `IMAGE_FOLDER = "cutouts/<your_image_stem>"`.

4) Generate masks and compute IoU

```
python iou.py
```

- Uses SAM to propose segments, classifies each with your model, keeps those labeled `roof`, then combines them.
- Saves combined predicted masks to `iou/output/` and prints per-image IoU and mean IoU (if GT exists in `iou/masks/`).


## Configuration Notes

- Device selection is automatic (`cuda` if available, else CPU).
- SAM is memory-intensive; reduce `points_per_side` if you hit OOM.
- Class names order is `["other", "roof"]` in `run.py`/`iou.py`. Ensure training data labels align (ImageFolder uses alphabetical class dirs).
- Resize size for classifier is `IMG_SIZE = 256` across scripts.


## Repository Contents

- `sam.py` — Cutouts from SAM proposals (RGBA, alpha = mask).
- `train.py` — CNN training for `roof` vs `other` on crops.
- `run.py` — Batch classifier with tiled matplotlib preview and CSV output.
- `iou.py` — Combined roof mask creation + IoU evaluation.
- `test.png` — Example image (for quick smoke tests).


## License

No license file is provided. Use at your own discretion or add a license appropriate to your project.


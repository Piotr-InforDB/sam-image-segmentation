import os
import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np

IMAGES_DIR = "images"
OUTPUT_DIR = "cutouts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "sam_models/sam_vit_h_4b8939.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=8,
)

index = 0
for filename in sorted(os.listdir(IMAGES_DIR)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(IMAGES_DIR, filename)
    print(f"Processing {image_path} ...")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {filename} (unreadable)")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1024, 1024))

    masks = mask_generator.generate(image)

    base_name = os.path.splitext(filename)[0]
    save_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(save_dir, exist_ok=True)

    for i, ann in enumerate(masks):
        mask = ann["segmentation"].astype(np.uint8)
        x, y, w, h = map(int, ann["bbox"])

        region = image[y:y+h, x:x+w]
        mask_crop = mask[y:y+h, x:x+w]

        cutout = cv2.bitwise_and(region, region, mask=mask_crop)
        cutout_rgba = cv2.cvtColor(cutout, cv2.COLOR_RGB2RGBA)
        cutout_rgba[:, :, 3] = mask_crop * 255

        out_path = os.path.join(save_dir, f"cutout_{index:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(cutout_rgba, cv2.COLOR_RGBA2BGRA))
        index += 1

    print(f"Saved {len(masks)} cutouts to {save_dir}")

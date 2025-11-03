import os
import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import shutil
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image



MODEL_TYPE = "vit_h"
SAM_MODEL = "sam_models/sam_vit_h_4b8939.pth"
CLASSIFICATION_MODEL = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGES_DIR = "iou/images"
OUTPUT_DIR = "iou/output"
MASKS_DIR = "iou/masks"
IMG_SIZE = 256
class_names = ["other", "roof"]

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_MODEL).to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,
)

def load_model(model_path, num_classes=2):
    model = RoofClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

class RoofClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
model = load_model(CLASSIFICATION_MODEL, num_classes=len(class_names))



total_iou = 0

for filename in sorted(os.listdir(IMAGES_DIR)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(IMAGES_DIR, filename)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1024, 1024))

    masks = mask_generator.generate(image)

    output_dir = os.path.join(IMAGES_DIR, "temp")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    output_masks = []

    for i, ann in enumerate(masks):
        mask = ann["segmentation"].astype(np.uint8)
        x, y, w, h = map(int, ann["bbox"])

        region = image[y:y + h, x:x + w]
        mask_crop = mask[y:y + h, x:x + w]

        cutout = cv2.bitwise_and(region, region, mask=mask_crop)
        cutout_rgba = cv2.cvtColor(cutout, cv2.COLOR_RGB2RGBA)
        cutout_rgba[:, :, 3] = mask_crop * 255

        out_path = os.path.join(output_dir, f"cutout_{i:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(cutout_rgba, cv2.COLOR_RGBA2BGRA))

        img = Image.open(out_path).convert("RGB")
        x_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(x_tensor)
            probs = torch.softmax(pred, dim=1)
            label = pred.argmax(1).item()
            confidence = probs[0][label].item()

        label_name = class_names[label]

        if label_name == "roof":
            output_masks.append((mask, x, y, w, h))

    # Combine all roof segments
    if output_masks:
        combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for mask, x, y, w, h in output_masks:
            combined_mask[y:y + h, x:x + w] = np.maximum(combined_mask[y:y + h, x:x + w], mask[y:y + h, x:x + w] * 255)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        mask_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.png")
        cv2.imwrite(mask_path, combined_mask)

        #Coompute IoU
        gt_path = os.path.join(MASKS_DIR, f"{os.path.splitext(filename)[0]}.png")
        if os.path.exists(gt_path):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            gt_mask = cv2.resize(gt_mask, (1024, 1024))

            # Convert red-tinted ground truth to binary mask
            if gt_mask.ndim == 3:
                red_channel = gt_mask[:, :, 2]
                gt_binary = (red_channel > 100).astype(np.uint8)
            else:
                gt_binary = (gt_mask > 100).astype(np.uint8)

            pred_binary = (combined_mask > 127).astype(np.uint8)

            intersection = np.logical_and(gt_binary, pred_binary).sum()
            union = np.logical_or(gt_binary, pred_binary).sum()
            iou = intersection / union if union > 0 else 0

            total_iou += iou
            print(f"IoU for {filename}: {iou:.4f}")
        else:
            print(f"No ground truth mask found for {filename}")

num_processed = len([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
if num_processed > 0:
    mean_iou = total_iou / num_processed
    print(f"\nMean IoU: {mean_iou:.4f}")
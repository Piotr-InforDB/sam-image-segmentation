import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import math

MODEL_PATH = "best_model.pth"
IMAGE_FOLDER = "segments"
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_VIS = 12



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



def load_model(model_path, num_classes=2):
    model = RoofClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model



transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_folder(model_path, folder_path, class_names=("other", "roof")):
    model = load_model(model_path, num_classes=len(class_names))
    image_paths = [p for p in Path(folder_path).glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]

    print(f"Found {len(image_paths)} images in {folder_path}")

    results = []
    for i in range(0, len(image_paths), BATCH_SIZE_VIS):
        batch_paths = image_paths[i:i + BATCH_SIZE_VIS]
        fig_cols = 4
        fig_rows = math.ceil(len(batch_paths) / fig_cols)
        plt.figure(figsize=(15, 3.5 * fig_rows))

        for j, img_path in enumerate(batch_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                x = transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    pred = model(x)
                    probs = torch.softmax(pred, dim=1)
                    label = torch.argmax(pred, dim=1).item()
                    confidence = probs[0][label].item()

                label_name = class_names[label]
                results.append((img_path.name, label_name, confidence))

                plt.subplot(fig_rows, fig_cols, j + 1)
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"{label_name} ({confidence:.2%})", fontsize=9)
            except Exception as e:
                print(f"Error on {img_path}: {e}")

        plt.tight_layout()
        plt.show()

    # Print summary
    print("\n" + "=" * 50)
    print("CLASSIFICATION SUMMARY")
    print("=" * 50)
    class_counts = {name: 0 for name in class_names}
    for _, label_name, _ in results:
        class_counts[label_name] += 1

    for class_name, count in class_counts.items():
        percentage = (count / len(results) * 100) if results else 0
        print(f"{class_name}: {count} images ({percentage:.1f}%)")
    print("=" * 50)

    return results



if __name__ == "__main__":
    results = classify_folder(MODEL_PATH, IMAGE_FOLDER)

    # Optionally save results to file
    with open("classification_results.txt", "w") as f:
        f.write("Filename,Class,Confidence\n")
        for filename, label, confidence in results:
            f.write(f"{filename},{label},{confidence:.4f}\n")
    print("\nResults saved to classification_results.txt")
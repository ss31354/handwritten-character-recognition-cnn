import torch
from model import HandwritingDeepCNN
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# Config (must match training)
CSV = "data/english.csv"
NUM_CLASSES = 62
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "handwriting_model.pth"

# Load label map
labels_sorted = sorted(set(pd.read_csv(CSV)['label']))
label_map = labels_sorted

# Define transforms for inference
test_tfms = T.Compose([
    T.Resize((64,64)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

def predict_image(image_path):
    img = Image.open(image_path).convert("L")
    img_tensor = test_tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        pred_label = label_map[pred_idx]

    # Display image
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {pred_label}")
    plt.axis('off')
    plt.show()

    print(f"🖼 {os.path.basename(image_path)} ➜ 🧠 Predicted Character: {pred_label}")

# Load model
model = HandwritingDeepCNN(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py path_to_image")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f"Error: File not found: {img_path}")
        sys.exit(1)

    predict_image(img_path)

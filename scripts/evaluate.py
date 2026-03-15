# scripts/evaluate.py
import torch
from model import HandwritingDeepCNN
from dataset import HandwrittenDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup
transform = T.Compose([
    T.Resize((64,64)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])
dataset = HandwrittenDataset("data/english.csv", "data/img", transform=transform)
loader  = DataLoader(dataset, batch_size=64)

# Load model
model = HandwritingDeepCNN(num_classes=62)
model.load_state_dict(torch.load("handwriting_model.pth", map_location=DEVICE))
model.to(DEVICE).eval()

# Evaluate
correct = total = 0
with torch.no_grad():
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"✅ Accuracy on full dataset: {100 * correct / total:.2f}%")

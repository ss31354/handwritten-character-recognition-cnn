import torch, time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset import HandwrittenDataset
from model   import HandwritingDeepCNN     # deep model with BatchNorm

# --------- configuration ----------
CSV          = "data/english.csv"
IMG_DIR      = "data/img"
NUM_CLASSES  = 62
BATCH_SIZE   = 64
EPOCHS       = 20
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MODEL_PATH   = "handwriting_model.pth"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
# -----------------------------------

# --------- transforms ----------
train_tfms = T.Compose([
    T.RandomRotation(10),
    T.RandomAffine(0, translate=(0.1,0.1)),
    T.RandomHorizontalFlip(),
    T.Resize((64,64)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

test_tfms  = T.Compose([
    T.Resize((64,64)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# --------- dataset ----------
dataset = HandwrittenDataset(CSV, IMG_DIR, transform=train_tfms)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --------- model / loss / optimiser ----------
model      = HandwritingDeepCNN(NUM_CLASSES).to(DEVICE)
criterion   = nn.CrossEntropyLoss()
optimizer   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# --------- training loop ----------
t0 = time.time()
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*y.size(0)
        correct      += (out.argmax(1) == y).sum().item()
        total        += y.size(0)

    scheduler.step()
    epoch_loss = running_loss/total
    epoch_acc  = 100*correct/total
    print(f"Epoch {epoch:2}/{EPOCHS} | Loss {epoch_loss:.3f} | Acc {epoch_acc:.2f}%")

print(f"Training finished in {(time.time()-t0)/60:.1f} min")
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved to", MODEL_PATH)

import io, base64, os, sys
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch, torchvision.transforms as T
import pandas as pd

# ---------- locate project paths ----------
APP_DIR   = os.path.dirname(os.path.abspath(__file__))      # .../app
ROOT_DIR  = os.path.abspath(os.path.join(APP_DIR, ".."))    # project root
SCRIPTS   = os.path.join(ROOT_DIR, "scripts")
sys.path.append(SCRIPTS)

from model import HandwritingDeepCNN

MODEL_PATH = os.path.join(ROOT_DIR, "handwriting_model.pth")
CSV_PATH   = os.path.join(ROOT_DIR, "data", "english.csv")

# ---------- load label map & model ----------
label_map = sorted(set(pd.read_csv(CSV_PATH)["label"]))

model = HandwritingDeepCNN(num_classes=62)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
softmax = torch.nn.Softmax(dim=1)

# ---------- preprocess identical for canvas & uploads ----------
transform = T.Compose([
    T.Resize((64, 64)),
    T.Grayscale(1),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])
def preprocess(pil_img: Image.Image):
    return transform(pil_img.convert("L")).unsqueeze(0)

# ---------- Flask ----------
app = Flask(__name__,
            template_folder=os.path.join(APP_DIR, "templates"),
            static_folder=os.path.join(APP_DIR, "static"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data_uri = (request.json or {}).get("image", "")
    if "," not in data_uri:
        return jsonify({"error": "no image"}), 400

    _, enc = data_uri.split(",", 1)
    pil    = Image.open(io.BytesIO(base64.b64decode(enc)))

    with torch.no_grad():
        out   = model(preprocess(pil))
        probs = softmax(out)
        idx   = probs.argmax().item()
        conf  = probs[0, idx].item()

    return jsonify({"pred": label_map[idx], "conf": round(conf, 3)})

if __name__ == "__main__":
    app.run(debug=True)

import os
import uuid
from typing import List, Tuple, Dict

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

# -----------------------------
# Configuration
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}

# Model (OpenAI CLIP via Hugging Face)
MODEL_NAME = "openai/clip-vit-base-patch32"

# Confidence controls
SOFTMAX_TEMP = 12.0          # increases separation (tweak 8-15)
UNCERTAIN_THRESHOLD = 0.45   # below this => Uncertain
CLOSE_GAP_THRESHOLD = 0.07   # if top1-top2 < gap => "Mixed/Unclear" note

# Prompt labels (use "a photo of ..." style - better for CLIP)
PROMPTS: List[str] = [
    "a photo of a plastic bottle",
    "a photo of a plastic food container",
    "a photo of a plastic cup",
    "a photo of paper waste",
    "a photo of cardboard",
    "a photo of a glass bottle",
    "a photo of a glass jar",
    "a photo of a battery",
    "a photo of a chemical container",
    "a photo of medicine blister packs",
    "a photo of electronic waste",
    "a photo of a metal can",
    "a photo of food waste",
    "a photo of general trash"
]

# Map prompt -> final category
PROMPT_TO_CATEGORY: Dict[str, str] = {
    "a photo of a plastic bottle": "Plastic",
    "a photo of a plastic food container": "Plastic",
    "a photo of a plastic cup": "Plastic",

    "a photo of paper waste": "Paper",
    "a photo of cardboard": "Paper",

    "a photo of a glass bottle": "Glass",
    "a photo of a glass jar": "Glass",

    "a photo of a battery": "Hazardous",
    "a photo of a chemical container": "Hazardous",
    "a photo of medicine blister packs": "Hazardous",

    "a photo of electronic waste": "E-Waste",
    "a photo of a metal can": "Recycling (Metal)",

    "a photo of food waste": "Food Waste",
    "a photo of general trash": "Landfill",
}

# What to display for each final category
GUIDE: Dict[str, Dict[str, List[str] or str]] = {
    "Plastic": {
        "bin": "Recycling (Plastic)",
        "tips": [
            "Rinse if dirty and remove food residue",
            "Remove lids if your city requires it",
            "Soft plastics/film rules vary by city"
        ]
    },
    "Paper": {
        "bin": "Recycling (Paper)",
        "tips": [
            "Keep paper dry",
            "Flatten cardboard",
            "Remove plastic wrap/covers if possible"
        ]
    },
    "Glass": {
        "bin": "Recycling (Glass)",
        "tips": [
            "Rinse containers",
            "Handle broken glass carefully",
            "Remove lids if required locally"
        ]
    },
    "Recycling (Metal)": {
        "bin": "Recycling (Metal)",
        "tips": [
            "Rinse cans",
            "Check local rules for aerosol cans",
            "Remove food residue"
        ]
    },
    "Food Waste": {
        "bin": "Green Bin / Compost",
        "tips": [
            "Remove packaging",
            "Check what your city accepts in green bin",
            "If mixed with plastic/metal, separate first"
        ]
    },
    "E-Waste": {
        "bin": "E-Waste Drop-off / Retail Takeback",
        "tips": [
            "Do NOT put in recycling bins",
            "Use municipal drop-off or retailer takeback",
            "Erase personal data if it’s a device"
        ]
    },
    "Hazardous": {
        "bin": "Hazardous Waste Depot / Special Drop-off",
        "tips": [
            "Do NOT put in recycling bins",
            "Batteries/chemicals/medicine need special disposal",
            "Store safely until you can drop it off"
        ]
    },
    "Landfill": {
        "bin": "Garbage / Landfill",
        "tips": [
            "If the item is mixed-material or heavily soiled, landfill may be required",
            "When unsure, avoid contaminating recycling",
            "Try to separate recyclable parts if possible"
        ]
    },
    "Uncertain": {
        "bin": "Uncertain",
        "tips": [
            "Retake the photo in good lighting with a plain background",
            "If it’s a battery/chemical/medicine, treat as hazardous",
            "Show the whole item clearly (not too close)"
        ]
    }
}

# -----------------------------
# App Setup
# -----------------------------
app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def _softmax(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max()
    return torch.exp(x) / torch.exp(x).sum()

def predict_topk(image: Image.Image, k: int = 3) -> List[Tuple[str, float]]:
    """
    Returns list of (prompt, probability) sorted by probability desc.
    """
    inputs = clip_processor(
        text=PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image[0]  # [num_prompts]

    probs = _softmax(logits * SOFTMAX_TEMP).detach().cpu()
    top_idx = torch.argsort(probs, descending=True)[:k].tolist()

    return [(PROMPTS[i], float(probs[i].item())) for i in top_idx]

def final_category_from_top(top_prompt: str, top_prob: float) -> str:
    if top_prob < UNCERTAIN_THRESHOLD:
        return "Uncertain"
    return PROMPT_TO_CATEGORY.get(top_prompt, "Uncertain")

def mixed_note(top3: List[Tuple[str, float]]) -> str:
    if len(top3) < 2:
        return ""
    gap = top3[0][1] - top3[1][1]
    if gap < CLOSE_GAP_THRESHOLD:
        return "This looks mixed/unclear (top matches are close). Consider retaking the photo."
    return ""

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("result.html", error="No file uploaded.")

    file = request.files["file"]
    if not file or file.filename == "":
        return render_template("result.html", error="No file selected.")

    if not allowed_file(file.filename):
        return render_template("result.html", error="Invalid file type. Upload JPG/PNG/WEBP.")

    # Save upload
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = secure_filename(f"{uuid.uuid4().hex}.{ext}")
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # Read image
    try:
        image = Image.open(save_path).convert("RGB")
    except Exception:
        return render_template("result.html", error="Could not read the image file.")

    # Predict
    top3 = predict_topk(image, k=3)
    top_prompt, top_prob = top3[0]

    category = final_category_from_top(top_prompt, top_prob)
    guide = GUIDE.get(category, GUIDE["Uncertain"])
    note = mixed_note(top3)

    # Format top3 for display
    top3_display = [(p.replace("a photo of ", ""), round(prob * 100, 2)) for p, prob in top3]

    return render_template(
        "result.html",
        image_url="/" + save_path.replace("\\", "/"),
        category=category,
        confidence=round(top_prob * 100, 2),
        recommended_bin=guide["bin"],
        tips=guide["tips"],
        top3=top3_display,
        note=note
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

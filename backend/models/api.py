"""
api.py — FastAPI wrapper around the cranberry prediction pipeline.
Run with: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import pickle
import io
import base64
import cv2

# ── Your existing pipeline imports ────────────────────────────────────────
from pipeline import load_DINO, load_sam3, extract_masks, extract_features_single

app = FastAPI()

# Allow the React dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_origins=["https://cranberry-rot-detector-677icdang-adrianjackson.vercel.app/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup — not on every request ───────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading models...")
DINO = load_DINO(
    BASE_DIR / "training" / "DINO_grid_models"
             / "ft-patch_mean_ru-3_bl-4_tmp-0.07_ep-30"
             / "backbone.pth",
    device=DEVICE
)
SAM3 = load_sam3(device=DEVICE)

clf_path = (BASE_DIR / "training" / "DINO_grid_models"
                     / "ft-patch_mean_ru-3_bl-4_tmp-0.07_ep-30"
                     / "svm_clf.pkl")
with open(clf_path, 'rb') as f:
    CLF = pickle.load(f)

print("Models ready.")


def _overlay_masks(image_pil: Image.Image,
                   predictions: list[dict],
                   masks_tensor: torch.Tensor,
                   ) -> str:
    """
    Renders each mask onto the image as a semi-transparent overlay.
    Returns the result as a base64 PNG string for the frontend.

    Rot   (class 0) → red
    Ripe  (class 1) → green
    """
    img_np = np.array(image_pil).copy()
    H, W   = img_np.shape[:2]

    # Resize masks to match original image
    masks_resized = torch.nn.functional.interpolate(
        masks_tensor.float(), size=(H, W),
        mode='bilinear', align_corners=False
    ).cpu().numpy()  # [N, 1, H, W]

    overlay = img_np.copy()
    for pred in predictions:
        idx   = pred['mask_index']
        color = (255, 80, 80) if pred['predicted_class'] == 0 else (80, 255, 80)
        mask  = masks_resized[idx, 0] > 0.5
        overlay[mask] = color

    blended = cv2.addWeighted(overlay, 0.35, img_np, 0.65, 0)

    # Encode to base64 so it travels cleanly over JSON
    _, buf = cv2.imencode('.png', cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode('utf-8')


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ── Load uploaded image ───────────────────────────────────────────────
    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert('RGB')
    original_size = image.size   # save before SAM resizes it

    # ── Run pipeline ──────────────────────────────────────────────────────
    sam_output, image_resized = extract_masks(SAM3, image=image)

    if sam_output is None or sam_output['masks'].shape[0] == 0:
        return JSONResponse({'error': 'No cranberries detected'}, status_code=422)

    masks  = sam_output['masks']
    boxes  = sam_output['boxes']
    scores = sam_output['scores']

    predictions, _ = extract_features_single(
        DINO, None, CLF, image_resized, masks, boxes, scores, DEVICE
    )

    # ── Build annotated image ─────────────────────────────────────────────
    annotated_b64 = _overlay_masks(image_resized, predictions, masks)

    # ── Scale bounding boxes back to original image size ─────────────────
    # (image was thumbnailed by SAM — boxes are in resized space)
    rw = original_size[0] / image_resized.size[0]
    rh = original_size[1] / image_resized.size[1]

    cranberries = []
    for pred in predictions:
        x1, y1, x2, y2 = pred['bounding_box']
        cranberries.append({
            'mask_index':      pred['mask_index'],
            'predicted_class': pred['predicted_class'],
            'label':           'rot' if pred['predicted_class'] == 0 else 'ripe',
            'confidence':      pred['confidence'],
            'p_rot':           float(pred['all_class_probs'][0]),
            'p_ripe':          float(pred['all_class_probs'][1]),
            'sam_score':       pred['sam_score'],
            # Bounding box in resized image space (matches annotated_image)
            'bbox':            [x1, y1, x2, y2],
        })

    # ── Summary stats ─────────────────────────────────────────────────────
    n_rot  = sum(1 for c in cranberries if c['predicted_class'] == 0)
    n_ripe = len(cranberries) - n_rot

    return {
        'annotated_image': annotated_b64,   # base64 PNG
        'image_size':      list(image_resized.size),  # [W, H] of annotated image
        'cranberries':     cranberries,
        'summary': {
            'total':   len(cranberries),
            'n_rot':   n_rot,
            'n_ripe':  n_ripe,
            'pct_rot': round(n_rot / max(len(cranberries), 1) * 100, 1),
        }
    }
"""
pipeline.py
-----------
Cranberry quality inspection pipeline.

Runs SAM3 (segmentation) followed by DINOv2 + SVM (classification) on a
cranberry image and returns per-berry rot/ripe predictions.

Typical usage
-------------
    processor = load_sam3("path/to/sam3_root")
    dino      = load_dino("path/to/backbone.pth")
    clf       = load_svm("path/to/svm_clf.pkl")

    sam_output, image = run_sam(processor, image_path="photo.jpg")
    predictions, _    = run_dino(dino, clf, image,
                                 sam_output["masks"],
                                 sam_output["boxes"],
                                 sam_output["scores"])

    annotated = draw_predictions(image, predictions, sam_output["masks"])
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path

import cv2
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ── SAM3 imports ──────────────────────────────────────────────────────────
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

# Minimum SAM confidence to keep a detected mask.
# Lower values detect more berries but increase false positives.
SAM_CONF_THRESH = 0.1

# P(rot) threshold for the SVM classifier.
# Predictions with P(rot) >= this value are labelled as rot.
# Lowering this increases rot recall at the cost of precision.
ROT_THRESHOLD = 0.55

# Maximum image dimension fed to SAM — larger images are resized
# proportionally to stay within GPU memory limits.
SAM_MAX_DIM = 1024

# ImageNet normalisation required by DINOv2 (pretrained on ImageNet).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Normalisation transform applied to each crop before DINOv2 inference.
_normalize = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# Label map used by draw_predictions and the API.
CLASS_NAMES = {0: "rot", 1: "ripe"}

# Overlay colours (RGB) for each class.
CLASS_COLORS = {0: (255, 80, 80), 1: (80, 255, 80)}


# ─────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────

def load_sam3(sam3_root: str | Path,
              conf_thresh: float = SAM_CONF_THRESH,
              ckpt_path = None,
              device: str = "cuda") -> Sam3Processor:
    """
    Loads the SAM3 image segmentation model.

    Args:
        sam3_root:   Root directory of the sam3 package, which must contain
                     ``sam3/assets/bpe_simple_vocab_16e6.txt.gz``.
        conf_thresh: Minimum confidence for a detected mask to be kept.
        device:      ``'cuda'`` or ``'cpu'``.

    Returns:
        A ``Sam3Processor`` instance ready for inference.
    """
    sam3_root = Path(sam3_root)
    bpe_path  = sam3_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    if not bpe_path.exists():
        raise FileNotFoundError(f"SAM3 BPE file not found: {bpe_path}")

    model     = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=ckpt_path)
    processor = Sam3Processor(model, confidence_threshold=conf_thresh)
    return processor


def load_dino(model_path: str | Path, device: str = "cuda"):
    """
    Loads a fine-tuned DINOv2 ViT-S/14 backbone.

    The model is loaded in evaluation mode with all parameters frozen.
    No classification head is attached — feature extraction uses the
    patch token mean (see ``run_dino``).

    Args:
        model_path: Path to the ``.pth`` state-dict saved during training.
        device:     ``'cuda'`` or ``'cpu'``.

    Returns:
        A DINOv2 model in eval mode on ``device``.
    """
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_svm(clf_path: str | Path):
    """
    Loads the fitted scikit-learn SVM classifier (StandardScaler + SVC pipeline).

    Args:
        clf_path: Path to the ``.pkl`` file saved during training.

    Returns:
        A fitted scikit-learn ``Pipeline``.
    """
    with open(clf_path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────
# SAM3 — segmentation
# ─────────────────────────────────────────────────────────────────────────

def run_sam(processor: Sam3Processor,
            image: Image.Image | None = None,
            image_path: str | Path | None = None,
            ) -> tuple[dict, Image.Image]:
    """
    Runs SAM3 on an image and returns per-berry masks, boxes, and scores.

    Exactly one of ``image`` or ``image_path`` must be provided.
    The image is resized proportionally to ``SAM_MAX_DIM`` before inference
    to avoid GPU out-of-memory errors on high-resolution inputs.

    Args:
        processor:  SAM3 processor returned by ``load_sam3``.
        image:      PIL image (RGB).
        image_path: Path to an image file on disk.

    Returns:
        A tuple of:
          - ``sam_output`` dict with keys ``masks`` [N,1,H,W],
            ``boxes`` [N,4], ``scores`` [N].
          - ``image`` PIL image at the resolution SAM used for inference.
            All bounding boxes and masks are in this coordinate space.

    Raises:
        ValueError: If neither or both of ``image`` / ``image_path`` are given.
    """
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
    elif image is None:
        raise ValueError("Provide either `image` or `image_path`.")

    # Resize to fit GPU memory while preserving aspect ratio.
    # PIL thumbnail modifies in-place and only shrinks, never enlarges.
    image.thumbnail((SAM_MAX_DIM, SAM_MAX_DIM), Image.Resampling.LANCZOS)

    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    raw = processor.set_text_prompt(state=inference_state, prompt="cranberry")

    return {
        "masks":  raw["masks"],   # [N, 1, H, W] bool tensor
        "boxes":  raw["boxes"],   # [N, 4]        float tensor (x1,y1,x2,y2)
        "scores": raw["scores"],  # [N]            float tensor
    }, image


# ─────────────────────────────────────────────────────────────────────────
# DINOv2 + SVM — classification
# ─────────────────────────────────────────────────────────────────────────

def run_dino(model,
             clf,
             image: Image.Image,
             masks:  torch.Tensor,
             boxes:  torch.Tensor,
             scores: torch.Tensor,
             device: str = "cuda",
             rot_threshold: float = ROT_THRESHOLD,
             ) -> tuple[list[dict], torch.Tensor]:
    """
    Extracts DINOv2 patch-mean features for each SAM mask and classifies
    each berry as rot (0) or ripe (1) using the SVM.

    Feature extraction
    ------------------
    For each berry:
      1. Crop the normalised image tensor to the SAM bounding box.
      2. Zero out pixels outside the SAM mask (background suppression).
      3. Resize the crop to 224×224 (DINOv2 input size).
      4. Extract the mean of all patch tokens from DINOv2's last layer.
         Patch mean captures local texture better than the CLS token,
         which is important for rot detection where rot is a surface feature.

    Classification
    --------------
    All berry features are passed to the SVM in a single batched call.
    A custom threshold (default 0.55) on P(rot) is applied instead of
    the SVM's default 0.5 decision boundary.

    Args:
        model:         DINOv2 backbone from ``load_dino``.
        clf:           SVM pipeline from ``load_svm``.
        image:         PIL image at the resolution SAM used (returned by ``run_sam``).
        masks:         SAM mask tensor [N, 1, H, W].
        boxes:         SAM box tensor  [N, 4].
        scores:        SAM score tensor [N].
        device:        ``'cuda'`` or ``'cpu'``.
        rot_threshold: P(rot) >= this → predicted class is rot (0).

    Returns:
        A tuple of:
          - ``predictions``: list of dicts, one per berry, with keys:
              ``mask_index``     int    — index into the SAM mask tensor
              ``bounding_box``   list   — [x1, y1, x2, y2] in SAM image space
              ``sam_score``      float  — SAM's confidence in the mask
              ``predicted_class`` int   — 0 = rot, 1 = ripe
              ``label``          str    — "rot" or "ripe"
              ``confidence``     float  — SVM confidence in predicted class
              ``p_rot``          float  — P(rot)
              ``p_ripe``         float  — P(ripe)
          - ``image_tensor``: the normalised image tensor [3, H, W] on CPU,
            useful for downstream visualisation or debugging.
    """
    image_tensor = _normalize(image).to(device)  # [3, H, W]

    all_feats = []
    metadata  = []

    # ── Pass 1: extract patch-mean features for each berry ────────────────
    model.eval()
    with torch.no_grad():
        for mask_idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            x1, y1, x2, y2 = box.int()

            # Crop both the image and the mask to the bounding box
            cropped_img  = image_tensor[:, y1:y2, x1:x2]       # [3, h, w]
            cropped_mask = mask[:, y1:y2, x1:x2]                # [1, h, w]

            # Zero out background pixels using the SAM mask
            masked = (cropped_img.unsqueeze(0) * cropped_mask).to(device)

            # Resize to 224×224 — required DINOv2 input resolution
            masked = F.interpolate(masked, size=(224, 224),
                                   mode="bilinear", align_corners=False)

            # Extract patch token mean — better than CLS for local texture
            out          = model.forward_features(masked)
            patch_tokens = out["x_norm_patchtokens"]   # [1, N_patches, 384]
            feat         = patch_tokens.mean(dim=1)    # [1, 384]

            all_feats.append(feat.cpu().numpy())
            metadata.append({
                "mask_index":   mask_idx,
                "sam_score":    float(score.cpu().numpy()),
                "bounding_box": [int(x) for x in [x1, y1, x2, y2]],
            })

    # ── Pass 2: batched SVM classification ───────────────────────────────
    # All features are classified in one call for efficiency.
    all_feats_np = np.concatenate(all_feats, axis=0)   # [N, 384]
    all_probs    = clf.predict_proba(all_feats_np)      # [N, 2] — [p_rot, p_ripe]

    # Apply custom threshold: lower than 0.5 increases rot recall,
    # trading some precision for fewer missed rot berries.
    all_preds = np.where(all_probs[:, 0] >= rot_threshold, 0, 1)

    # ── Assemble structured output ─────────────────────────────────────────
    predictions = [
        {
            **meta,
            "predicted_class": int(pred),
            "label":           CLASS_NAMES[int(pred)],
            "confidence":      float(probs[pred]),
            "p_rot":           float(probs[0]),
            "p_ripe":          float(probs[1]),
        }
        for meta, pred, probs in zip(metadata, all_preds, all_probs)
    ]

    return predictions, image_tensor.cpu()


# ─────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────

def draw_predictions(image: Image.Image,
                     predictions: list[dict],
                     masks: torch.Tensor,
                     alpha: float = 0.3,
                     ) -> np.ndarray:
    """
    Renders SAM masks and prediction labels onto the image.

    Each berry is coloured by its predicted class:
      - Red  (255, 80, 80) → rot
      - Green (80, 255, 80) → ripe

    A semi-transparent colour overlay is blended over the mask region,
    and a labelled bounding box is drawn at the top of each berry.

    Args:
        image:       PIL image at the same resolution SAM used.
        predictions: List of prediction dicts from ``run_dino``.
        masks:       SAM mask tensor [N, 1, H, W].
        alpha:       Opacity of the mask colour overlay (0 = invisible,
                     1 = fully opaque). Default 0.3.

    Returns:
        Annotated image as a uint8 numpy array (RGB).
    """
    img_np = np.array(image).astype(float)
    H, W   = img_np.shape[:2]

    # Resize masks to match the display image if they differ in size
    if masks.shape[-2:] != (H, W):
        masks_resized = F.interpolate(
            masks.float(), size=(H, W),
            mode="bilinear", align_corners=False
        )
    else:
        masks_resized = masks.float()

    for pred in predictions:
        idx        = pred["mask_index"]
        color      = CLASS_COLORS[pred["predicted_class"]]
        confidence = pred["confidence"]
        label      = pred["label"]
        x1, y1, x2, y2 = pred["bounding_box"]

        # Blend mask colour over the berry region
        mask    = masks_resized[idx, 0].cpu().numpy() > 0.5
        overlay = img_np.copy()
        overlay[mask] = color
        img_np  = (1 - alpha) * img_np + alpha * overlay

        # Bounding box
        cv2.rectangle(img_np.astype(np.uint8),
                      (x1, y1), (x2, y2), color, 2)

        # Label with confidence
        text      = f"{label}: {confidence:.0%}"
        font      = cv2.FONT_HERSHEY_SIMPLEX
        scale     = 0.55
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

        # Filled background rectangle behind text for legibility
        cv2.rectangle(img_np.astype(np.uint8),
                      (x1, y1 - th - 10),
                      (x1 + tw + 8, y1),
                      color, cv2.FILLED)
        cv2.putText(img_np.astype(np.uint8),
                    text, (x1 + 4, y1 - 5),
                    font, scale, (255, 255, 255), thickness)

    return img_np.astype(np.uint8)


def save_annotated_image(image: Image.Image,
                         predictions: list[dict],
                         masks: torch.Tensor,
                         output_path: str | Path,
                         ) -> None:
    """
    Draws predictions onto the image and saves the result to disk.

    Args:
        image:       PIL image at SAM resolution.
        predictions: List of prediction dicts from ``run_dino``.
        masks:       SAM mask tensor [N, 1, H, W].
        output_path: Destination file path (JPEG or PNG).
    """
    annotated = draw_predictions(image, predictions, masks)
    # cv2 expects BGR; convert from the RGB array draw_predictions returns
    cv2.imwrite(str(output_path),
                cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"Saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────
# Quick local test
# ─────────────────────────────────────────────────────────────────────────

def _run_local_test():
    """
    Runs the full pipeline on a single hard-coded image and displays the
    result in an OpenCV window.  Not intended for production use.
    """
    # ── Paths — edit these to match your local layout ─────────────────────
    BASE_DIR   = Path(__file__).resolve().parent.parent
    SAM3_ROOT  = BASE_DIR.parent.parent / "sam3"
    DINO_PATH  = (BASE_DIR / "training" / "DINO_grid_models"
                           / "ft-patch_mean_ru-3_bl-4_tmp-0.07_ep-30"
                           / "backbone.pth")
    CLF_PATH   = (BASE_DIR / "training" / "DINO_grid_models"
                           / "ft-patch_mean_ru-3_bl-4_tmp-0.07_ep-30"
                           / "svm_clf.pkl")
    IMAGE_PATH = BASE_DIR / "data" / "images" / "167.JPG"
    OUTPUT_DIR = BASE_DIR / "testing" / "pipeline_out"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load models ───────────────────────────────────────────────────────
    print("Loading models...")
    sam3 = load_sam3(SAM3_ROOT, device=device)
    dino = load_dino(DINO_PATH,  device=device)
    clf  = load_svm(CLF_PATH)

    # ── Run pipeline ──────────────────────────────────────────────────────
    print(f"Processing {IMAGE_PATH.name}...")
    sam_output, image = run_sam(sam3, image_path=IMAGE_PATH)

    predictions, _ = run_dino(
        dino, clf, image,
        sam_output["masks"],
        sam_output["boxes"],
        sam_output["scores"],
        device=device,
    )

    # ── Print summary ─────────────────────────────────────────────────────
    n_rot  = sum(1 for p in predictions if p["predicted_class"] == 0)
    n_ripe = len(predictions) - n_rot
    print(f"\nDetected {len(predictions)} berries — "
          f"{n_ripe} ripe, {n_rot} rot "
          f"({n_rot / max(len(predictions), 1):.0%} rot rate)")

    # ── Display and save ──────────────────────────────────────────────────
    annotated = draw_predictions(image, predictions, sam_output["masks"])
    cv2.imshow("Cranberry Predictions", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_annotated_image(image, predictions, sam_output["masks"],
                         OUTPUT_DIR / f"annotated_{IMAGE_PATH.stem}.jpg")


if __name__ == "__main__":
    _run_local_test()
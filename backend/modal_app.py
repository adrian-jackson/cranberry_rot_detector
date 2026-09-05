# modal_app.py
# Deploy with: modal deploy modal_app.py
# One-time setup: pip install modal && modal setup

import modal
from pathlib import Path
# ── Define the cloud environment ──────────────────────────────────────────
# This replaces requirements.txt + CUDA install entirely.
# Modal builds this image once and caches it.

# Pinned to a specific commit so a redeploy months from now can't silently
# pull a breaking upstream change. Bump deliberately when you want an update.
SAM3_COMMIT = "86ed77094094e5cabb16b0414ec60c5ba9ce0a0f"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "tesseract-ocr", "libzbar0")   # git: needed by pip to clone SAM3; rest for read_label() OCR/barcode
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "fastapi",
        "python-multipart",
        "Pillow",
        "opencv-python-headless",   # headless — no display needed on server
        "numpy>=1.26,<2",
        "scikit-learn==1.8.0",
        "huggingface-hub>=0.23.0",
        "einops>=0.7.0",
        "psutil",
        "pytesseract",
        "pyzbar",
        "setuptools",       # provides pkg_resources — sam3.model_builder uses
                            # pkg_resources.resource_filename() unconditionally
        "pycocotools",      # sam3.model.sam3_video_base (pulled in by
                            # model_builder even though we only use the image
                            # path) does `from sam3.train.masks_ops import
                            # rle_encode`, and that module does `import
                            # pycocotools.mask` at the top — unconditional,
                            # not training-only despite living under train/.
        "triton",           # sam3/model/edt.py does `import triton` at module
                            # top level; reachable from model_builder's import
                            # chain. Usually ships as a transitive dep of
                            # torch's Linux CUDA wheel anyway, but pin it
                            # explicitly rather than rely on that.
    )
    # Install SAM3 straight from the upstream GitHub repo — it's a properly
    # packaged pip module (pyproject.toml, package-data includes the BPE
    # vocab asset), so no local mount/copy is needed. Its own pyproject.toml
    # declares timm/tqdm/ftfy/regex/iopath/typing_extensions/huggingface_hub
    # as base dependencies, so pip resolves those automatically here — but
    # pycocotools and triton above are NOT declared there (a gap in sam3's
    # own packaging), which is exactly what broke this deploy once already.
    # If a future SAM3_COMMIT bump throws another ModuleNotFoundError, re-run
    # the reachability trace (see dev.notes) rather than assuming a package
    # is unused just because it's not in sam3's declared dependencies.
    .pip_install(f"sam3 @ git+https://github.com/facebookresearch/sam3.git@{SAM3_COMMIT}")
    # Bundle your pipeline.py so the container can import it
    .add_local_file("backend/pipeline.py", "/usr/local/pipeline.py")
)

# ── Mount your model files ────────────────────────────────────────────────
# These are uploaded to Modal's storage once and reused across requests.
# Users never touch these files.
model_volume = modal.Volume.from_name("cranberry-models", create_if_missing=True)

app = modal.App("cranberry-inspector", image=image)


@app.cls(
    gpu="T4",                        # cheapest GPU — sufficient for SAM3 + DINOv2
    volumes={"/models": model_volume},
    scaledown_window=300,      # keep warm for 5 min between requests
    secrets=[modal.Secret.from_name("cranberry-api-key")],
    timeout=600,               # generous enough for a full /predict/batch request
)
class CranberryInspector:

    @modal.enter()
    def load_models(self):
        """Runs once when the container starts — loads all models into memory."""
        import torch
        import pickle
        import sys
        sys.path.insert(0, "/usr/local")         # make pipeline.py importable
        from pipeline import load_sam3, load_dino, load_svm, set_seed

        set_seed()   # reproducible outputs across requests/deploys

        device = "cuda"

        self.clf = load_svm("/models/svm_clf.pkl")
        self.sam3 = load_sam3(ckpt_path="/models/sam3_weights/sam3.pt", device=device)
        self.dino = load_dino("/models/backbone.pth", device=device)

        with open("/models/svm_clf.pkl", "rb") as f:
            self.clf = pickle.load(f)

        self.device = device
        print("Models loaded.")


    @modal.asgi_app()
    def fastapi_app(self):
        import os
        from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
        from fastapi.middleware.cors import CORSMiddleware
        import io, base64, cv2
        import numpy as np
        from PIL import Image
        from pipeline import run_sam, run_dino, draw_predictions, read_label, SAM_FALLBACK_PROMPTS

        web_app = FastAPI()
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        MAX_BATCH_SIZE = 25   # keep a single /predict/batch request well inside the class timeout

        def require_api_key(x_api_key: str = Header(default=None)):
            if x_api_key != os.environ["API_KEY"]:
                raise HTTPException(status_code=401, detail="Missing or invalid API key")

        def predict_one(image):
            """Runs the full pipeline on one already-decoded PIL image."""
            cran_error = 0

            sam_output, image_resized, sam_prompt_used = None, image, None
            for prompt in SAM_FALLBACK_PROMPTS:
                sam_output, image_resized = run_sam(self.sam3, image=image, prompt=prompt)
                if sam_output["masks"].shape[0] > 0:
                    sam_prompt_used = prompt
                    break

            detected_label = read_label(image_resized)

            if sam_output["masks"].shape[0] == 0:
                # Every fallback prompt found nothing — skip DINO/SVM (which
                # can't classify zero berries) and report the failure as-is.
                cran_error = 1
                predictions = []
                annotated = np.array(image_resized)
            else:
                predictions, _ = run_dino(
                    self.dino, self.clf, image_resized,
                    sam_output["masks"],
                    sam_output["boxes"],
                    sam_output["scores"],
                    device=self.device,
                )
                annotated = draw_predictions(
                    image_resized, predictions, sam_output["masks"],
                    label_caption=detected_label["value"] if detected_label["type"] != "none" else None,
                )

            _, buf = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            b64    = base64.b64encode(buf).decode()

            n_rot  = sum(1 for p in predictions if p["predicted_class"] == 0)
            n_ripe = len(predictions) - n_rot

            return {
                "error": cran_error,
                "sam_prompt_used": sam_prompt_used,
                "detected_label": detected_label,
                "annotated_image": b64,
                "image_size":      list(image_resized.size),
                "cranberries":     predictions,   # already JSON-serialisable — all floats/ints/strings
                "summary": {
                    "total":   len(predictions),
                    "n_rot":   n_rot,
                    "n_ripe":  n_ripe,
                    "pct_rot": round(n_rot / max(len(predictions), 1) * 100, 1),
                }
            }

        def decode_image(content: bytes) -> Image.Image:
            try:
                return Image.open(io.BytesIO(content)).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

        @web_app.post("/predict")
        async def predict(file: UploadFile = File(...), _=Depends(require_api_key)):
            image = decode_image(await file.read())
            return predict_one(image)

        @web_app.post("/predict/batch")
        async def predict_batch(files: list[UploadFile] = File(...), _=Depends(require_api_key)):
            if len(files) > MAX_BATCH_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch too large ({len(files)} files) — max {MAX_BATCH_SIZE} per request. Split into multiple requests.",
                )

            results = []
            for f in files:
                try:
                    image = decode_image(await f.read())
                    result = predict_one(image)
                except HTTPException as e:
                    # One bad file shouldn't fail the whole batch — record it
                    # and keep processing the rest.
                    result = {"error": 1, "detail": e.detail}
                result["filename"] = f.filename
                results.append(result)

            return {"results": results}

        return web_app
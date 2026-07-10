# modal_app.py
# Deploy with: modal deploy modal_app.py
# One-time setup: pip install modal && modal setup

import modal
from pathlib import Path
# ── Define the cloud environment ──────────────────────────────────────────
# This replaces requirements.txt + CUDA install entirely.
# Modal builds this image once and caches it.

image = (
    modal.Image.debian_slim(python_version="3.11")
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
        # SAM3 / Meta vision dependencies
        "fvcore",
        "omegaconf",
        "hydra-core",
        "timm>=1.0.17",
        "tqdm",
        "ftfy==6.1.1",
        "regex",
        "iopath>=0.1.10",
        "typing_extensions",
        #optional imports from SAM3 - modal still fails without these
        "pytest",
        "pytest-cov",
        "black==24.2.0",
        "ufmt==2.8.0",
        "ruff-api==0.1.0",
        "usort==1.0.2",
        "gitpython==3.1.31",
        "yt-dlp",
        "pandas",
        "pycocotools",
        "numba",
        "python-rapidjson",
    )
    # Install SAM3 from your git repo - currently unused in favor of local mount
    # .pip_install("git+https://github.com/yourname/sam3.git")
        # Bundle SAM3 source directly into the image — no mount needed
    .add_local_dir(
        "C:/Users/Adria/Documents/sam3",  # local path
        "/usr/local/sam3",                 # where it lands in the container
    )
    # Add BPE file into the assets folder SAM3 expects
    .add_local_file(
        "C:/Users/Adria/Documents/github/cranberry_algorithms/cranberry_rot_detector/backend/models/sam3_assets/bpe_simple_vocab_16e6.txt.gz",
        "/usr/local/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    )
    # Also bundle your pipeline.py so the container can import it
    .add_local_file("pipeline.py", "/usr/local/pipeline.py")
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
)
class CranberryInspector:

    @modal.enter()
    def load_models(self):
        """Runs once when the container starts — loads all models into memory."""
        import torch
        import pickle
        import sys
        sys.path.insert(0, "/usr/local/sam3")   # make SAM3 importable
        sys.path.insert(0, "/usr/local")         # make pipeline.py importable
        from pipeline import load_sam3, load_dino, load_svm

        device = "cuda"

        self.clf = load_svm("/models/svm_clf.pkl")
        self.sam3 = load_sam3("/usr/local/sam3", ckpt_path="/models/sam3_weights/sam3.pt", device=device)
        self.dino = load_dino("/models/backbone.pth", device=device)

        with open("/models/svm_clf.pkl", "rb") as f:
            self.clf = pickle.load(f)

        self.device = device
        print("Models loaded.")


    @modal.asgi_app()
    def fastapi_app(self):
        from fastapi import FastAPI, UploadFile, File
        from fastapi.middleware.cors import CORSMiddleware
        import io, base64, cv2
        import numpy as np
        from PIL import Image
        from pipeline import run_sam, run_dino, draw_predictions

        web_app = FastAPI()
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.post("/predict")
        async def predict(file: UploadFile = File(...)):
            content = await file.read()
            image   = Image.open(io.BytesIO(content)).convert("RGB")

            sam_output, image_resized = run_sam(self.sam3, image=image)

            if sam_output["masks"].shape[0] == 0:
                return {"error": "No cranberries detected"}

            predictions, _ = run_dino(
                self.dino, self.clf, image_resized,
                sam_output["masks"],
                sam_output["boxes"],
                sam_output["scores"],
                device=self.device,
            )

            annotated = draw_predictions(image_resized, predictions, sam_output["masks"])
            _, buf    = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            b64       = base64.b64encode(buf).decode()

            n_rot  = sum(1 for p in predictions if p["predicted_class"] == 0)
            n_ripe = len(predictions) - n_rot

            return {
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
        return web_app
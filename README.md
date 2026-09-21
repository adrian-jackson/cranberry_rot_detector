# Cranberry Rot Detector

Automatically detects and classifies cranberries in photos as **ripe** or **rot**,
using SAM3 (segmentation) and DINOv2 (classification). The inference pipeline
runs as a GPU API on [Modal](https://modal.com); the demo web frontend is a
React/Vite app deployed on [Vercel](https://vercel.com).

If you're looking to demo this product or test it out, use the **web frontend**. Go to [cranberry-rot-detector.vercel.app](https://cranberry-rot-detector.vercel.app/), upload an image, and check out the guide on there.

If you're looking to update the backend or frontend, check the corresponding sections below.

If you're looking to use this in-field, use the **Python API**. Details are below in the "Using the API Directly" section.

Finally, if you're looking to adapt this framework or fine-tune the DINO backend for better ripe/rot classification accuracy, look at the "Retraining" section.

## How it works

1. **SAM3** segments every cranberry it can find in the photo. If it finds
   none on the first try, the request automatically retries with alternate
   prompts ("cranberry" → "fruit" → "berry") before giving up.
2. **DINOv2 + an SVM classifier** scores each detected berry as ripe or rot.
3. Any printed label or barcode/QR code visible in the photo is read
   best-effort (barcode/QR first, then OCR) and drawn as a caption on the
   annotated result.

## Repo layout

```
backend/        Modal deployment (modal_app.py) + inference pipeline (pipeline.py)
src/            React frontend (Vite)
pyproject.toml  Python dependencies for local backend development
upload_models.py  One-time script to push model weights to Modal's storage
```

---

## Backend: deploying to Modal

### 1. Install the Modal CLI and authenticate (one-time)

```bash
pip install modal
modal setup
```

### 2. Upload model weights to Modal's storage (one-time)

Place `backbone.pth` and `svm_clf.pkl` under `backend/models/` (see
`upload_models.py` for exact paths), then:

```bash
python upload_models.py
```

### 3. Create the API access key (one-time)

The `/predict` endpoint requires an `X-API-Key` header on every request.
Generate a key and store it as a Modal Secret — **do not commit this value
anywhere in the repo**:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
modal secret create cranberry-api-key API_KEY=<paste the generated value>
```

Keep the value somewhere private (a password manager, not a file in this
repo). You'll share it out-of-band — text, email, Slack DM — with whoever
you want to have access. Anyone without it gets a `401` from the API.

### 4. Deploy

```bash
modal deploy backend/modal_app.py
```

Modal prints a URL like `https://yourname--cranberry-inspector-cranberryinspector-fastapi-app.modal.run` — copy it, you'll need it for the frontend.

If SAM3 needs to be updated later, bump `SAM3_COMMIT` in `backend/modal_app.py`
to a newer commit SHA from [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
and redeploy.

---

## Frontend: local dev and deployment

### Local development

```bash
npm install
cp .env.example .env
# edit .env: set VITE_API_URL to the Modal URL from step 4 above
npm run dev
```

### Deploying to Vercel

Connect the repo in Vercel (it already picks up `vercel.json`'s build
settings) and set `VITE_API_URL` as a Vercel environment variable, pointing
at your Modal deployment URL.

### The access key, from the frontend's side

The API key is **not** baked into the build — a `VITE_`-prefixed env var
would get inlined into the public JS bundle, readable by anyone who visits
the site. Instead, the app prompts for a key on first visit and stores it in
the browser's `localStorage`. Give people the key you created in step 3
above; they paste it in once. A "Change access key" link at the bottom of
the page lets you (or them) re-enter it if it's ever rotated.

---

## Using the web app

1. Choose **Single Image** to upload one photo, or **Folder** to batch-process
   an entire folder of images (processed a few at a time, with a progress bar).
2. Each berry is outlined in **green** (ripe) or **red** (rot); hover over one
   to see its exact probability breakdown.
3. Toggle **Original / Annotated** to compare against the raw photo.
4. Click **Export** to download a `.zip` containing every annotated image
   plus a `results.xlsx` summarizing filename, rot %, ripe %, average DINO
   confidence, and any detected label/barcode for each image.

---

## Using the API directly

Please contact me for the modal backend URL - this is kept private for safety. All requests (both endpoints below) require an `X-API-Key` header. Contact me for this as well.

### `POST /predict`

- **Body**: `multipart/form-data` with a single field, `file`, containing the
  image.
- **Auth**: `X-API-Key: <your key>` header. Missing or wrong key → `401`.

**curl:**

```bash
curl -X POST "modal-backend.modal.run/predict" \
  -H "X-API-Key: $CRANBERRY_API_KEY" \
  -F "file=@photo.jpg"
```

**Python:**

```python
import os
import requests

API_URL = "https://adrian-jackson--cranberry-inspector-cranberryinspector-f-db90e3.modal.run/predict"
API_KEY = <api key>

with open(r"C:\Users\Adria\Documents\github\cranberry_algorithms\cranberry_algorithms\src\cranberry_algorithms\raw_data\jpg\1.JPG", "rb") as f:
    resp = requests.post(
        API_URL,
        headers={"X-API-Key": API_KEY},
        files={"file": f},
    )
print(repr(resp.text))
print(repr(API_KEY))
resp.raise_for_status()
data = resp.json()
print(data["summary"])
```

### `POST /predict/batch`

Same idea, but with repeated `files` fields (up to `MAX_BATCH_SIZE` — currently 25 — per request) instead of a single `file`.

**curl:**

```bash
curl -X POST "https://your-modal-endpoint.modal.run/predict/batch" \
  -H "X-API-Key: $CRANBERRY_API_KEY" \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg" \
  -F "files=@photo3.jpg"
```

**Python:**

```python
import os
import requests

API_URL = "https://your-modal-endpoint.modal.run/predict/batch"
API_KEY = os.environ["CRANBERRY_API_KEY"]  # best practice to not hardcode this - but you can..

image_paths = ["photo1.jpg", "photo2.jpg", "photo3.jpg"] #if you have a folder, use os.listdir()
files = [("files", open(p, "rb")) for p in image_paths]

try:
    resp = requests.post(
        API_URL,
        headers={"X-API-Key": API_KEY},
        files=files,
    )
finally:
    for _, f in files:
        f.close()

resp.raise_for_status()
data = resp.json()
for result in data["results"]:
    if result["error"]:
        print(result["filename"], "->", result.get("detail", "no cranberries detected"))
    else:
        print(result["filename"], "->", f"{result['summary']['pct_rot']}% rot")
```

**Response shape (`/predict` — a batch response wraps one of these per file in `results: [...]`):**

```jsonc
{
  "error": 0,                     // 1 if no berries were found even after retrying
  "sam_prompt_used": "cranberry", // which SAM3 prompt succeeded ("cranberry"/"fruit"/"berry")
  "detected_label": {             // best-effort OCR/barcode read, "none" if nothing found
    "type": "barcode",            // "barcode" | "text" | "none"
    "value": "0123456789"
  },
  "annotated_image": "<base64 PNG>",
  "image_size": [1024, 768],
  "cranberries": [
    {
      "mask_index": 0,
      "sam_score": 0.91,
      "bounding_box": [x1, y1, x2, y2],
      "predicted_class": 1,       // 0 = rot, 1 = ripe
      "label": "ripe",
      "confidence": 0.87,
      "p_rot": 0.13,
      "p_ripe": 0.87
    }
  ],
  "summary": {
    "total": 42,
    "n_rot": 5,
    "n_ripe": 37,
    "pct_rot": 11.9
  }
}
```

### Single image vs. batch

There are two endpoints. Using either with python is the same, except for the endpoint (/predict for single or /predict/batch/ for batch). See above.

- **`POST /predict`** — one image per request, field name `file`. Response
  shape is exactly what's shown above.
- **`POST /predict/batch`** — multiple images in one request, repeated
  `files` fields (e.g. `-F "files=@a.jpg" -F "files=@b.jpg"`). Processes them
  sequentially server-side and returns:

  ```jsonc
  {
    "results": [
      { "filename": "a.jpg", "error": 0, "sam_prompt_used": "cranberry", "...": "same shape as /predict" },
      { "filename": "b.jpg", "error": 1, "detail": "Could not read image: ..." }
    ]
  }
  ```

  Each entry includes a `filename` field (echoing what you sent) so you can
  correlate results back to your inputs. **A batch request is capped at 25
  files** (`MAX_BATCH_SIZE` in `backend/modal_app.py`) — send more than that
  and you get a `400` telling you to split it up. One bad file (corrupt
  image, or `error: 1` because no berries were found even after the prompt
  retries) doesn't fail the rest of the batch — it just shows up as its own
  entry in `results`.

  A batch of 25 is one GPU processing 25 images back-to-back, not 25 in
  parallel — there's no per-image speedup over calling `/predict` 25 times
  yourself, just fewer HTTP round-trips and one JSON response instead of 25.
  If you need real concurrency (multiple GPU containers working at once),
  issue several requests — to either endpoint — in parallel instead of
  relying on batch size.

The web app's **Folder** mode doesn't use `/predict/batch` — it calls
`/predict` once per file through a 3-way concurrency-limited loop
(`BATCH_CONCURRENCY` in `src/app.jsx`) instead, so it can show live
per-file progress and avoid one big request timing out on a huge folder.
`/predict/batch` exists for scripts/integrations that want one round-trip
for a modest batch and don't need progress feedback mid-request.

### Adjusting how predictions are made

None of this is exposed as request parameters — it can't be changed
per-request, only by whoever runs the deployment editing
`backend/pipeline.py` and running `modal deploy backend/modal_app.py` again.
The knobs that exist today:

| Constant                 | Default                             | What it controls                                                                         | Why you'd change it                                                                                                                       |
| ------------------------ | ----------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `SAM_CONF_THRESH`      | `0.1`                             | Minimum SAM3 confidence for a detected mask to count as a berry.                         | Lower to catch more (partially obscured/oddly shaped) berries at the cost of more false positives; raise it to be stricter.               |
| `ROT_THRESHOLD`        | `0.55`                            | `P(rot)` cutoff the SVM uses to call a berry rot instead of ripe.                      | Lower to flag more berries as rot (higher recall, more false rot calls); raise it to require more certainty before calling something rot. |
| `SAM_MAX_DIM`          | `1024`                            | Max image dimension fed to SAM3 (larger photos are downscaled to this before inference). | Raise it if you need finer detail on very high-resolution photos and have GPU memory to spare; lower it to reduce memory use/latency.     |
| `SAM_FALLBACK_PROMPTS` | `["cranberry", "fruit", "berry"]` | Text prompts tried against SAM3, in order, until one finds at least one mask.            | Add/reorder prompts if your photos contain something SAM3 doesn't recognize as any of these terms.                                        |
| `SEED`                 | `0`                               | Seed applied at model-load time so repeated runs are reproducible.                       | Change it if you specifically want a different (but still fixed) run, e.g. for A/B-testing determinism itself.                            |

---

## Retraining DINOv2 and SAM3

This is quite an complex topic, so I'll briefly cover the process for doing this. As for motivation, you'd want to retrain/finetune SAM if your cranberries are failing to get identified as cranberry; you'd want to finetune DINO if your accuracy is low.

Note that SAM has nothing to do with predicting the status of a cranberry once it's been identified- if your classifier accuracy is low, retrain DINO!

The current implementation has no fine-tuning of SAM. For information on how to set that up and hardware requirements, check out [github.com/facebookresearch/sam3/blob/main/README_TRAIN.md](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md). Of course, if a better segmenter comes out in the time between I'm writing this and your usage, feel free to use that. You'll need to ensure you update `run_sam`in `backend/pipeline.py`to modify the output of whatever segmenter you choose to match the current format, which is detailed in `run_sam`.

As for DINOv2, the easiest way to boost accuracy would be to switch to DINOv3- a newer, more powerful feature extractor. You might also consider fitting a more modern classifier head that can make better use of the high-dimensional DINO output. Currently (for the sake of simplicity & because it seemed to work fairly well), a simple SVM is used on top of the DINO output. Note that DINO features are notprojected to a lower dimension before SVM usage.

To retrain DINOv2 (or train a different feature extractor) you need to annotate data. This entails taking a picture of cranberries, marking which pixels correspond to an individual cranberry, and then labeling it as either rotted or ripe. There are plenty of tools that help streamline this process - my favorite is CVAT, though Roboflow is easier to set up. It is very important that the training data is diverse - cranberries should be/have a wide variety of shapes, colors, orientations, spacing from neighbors, lighting conditions, etc. Look through all your available training data before selecting that diverse set so you know how to choose good data. If you'd like my training set to add to yours, contact me at adrianj3@illinois.edu.

There are plenty of guides available online detailing how to finetune DINO. Agents are also quite good at this. Be sure to ask them to set up a grid search to identify the optimal hyperparameter settings- one of the biggest knobs I found was adjusting how many blocks were unfrozen.

## Troubleshooting

**"No cranberries detected" even after retrying**
Your image is likely obscured in some way that DINO didn't encounter in the training set. A variety of lighting conditions were used, but these models are black boxes and can fail silently sometimes. The backend already retries SAM3 with a couple of alternate prompts before giving up ("fruit", "berry", "cranberry"). These images will likely need to be either retaken or manually annotated; retrying is unlikely to work as these models are seeded and should produce repeatable outputs.

**Classifier is wrong**
DINO was trained on a fairly small dataset - around 500 hand-labeled cranberries. This is quite a manual process, and as one person, I didn't have time to do too many. As a result, the classifier isn't always going to be very accurate. In testing, I averaged around 90% accuracy. If you're far below that, your images may simply differ from the training set significantly - consider retraining DINO. Note that each prediction from the algorithm comes with a confidence metric specifying how certain the model is that its prediction is correct. If confidence is low, then something about that specific cranberry probably wasn't familiar to DINO. This happens because rotted cranberries can take many different appearances: they can be shriveled, discolored, spherical, blocthy, etc. These can sometimes either be difficult for DINO to extract meanignful features from; or SAM may simply not realize they are cranberries because they look so different. If this problem occurs frequently, it may be wise to fine-tune SAM on what rotted cranberries look like.

**`401 Unauthorized` from the API**
Your `X-API-Key` header is missing or doesn't match the key stored in the
`cranberry-api-key` Modal Secret. In the web app, use "Change access key" to
re-enter it.

**First request takes a long time**
Normal — Modal spins up a GPU container on the first request after a period
of inactivity (`scaledown_window` is 5 minutes). Subsequent requests are fast.

**`detected_label` is empty or wrong**
OCR/barcode reading is best-effort and implemented somewhat naively. This may need adjusting for your specific use case.

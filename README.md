# Cranberry Inspector

Automatically detects and classifies cranberries in photos as **ripe** or **rot**
using SAM3 (segmentation) and DINOv2 (classification).

Hover over any berry in the result image to see its rot/ripe probability.

---

## First-time setup

You will need:
- A [Modal](https://modal.com) account (free — provides the GPU)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed

### Step 1 — Clone the repo

```bash
git clone https://github.com/yourname/cranberry-inspector.git
cd cranberry-inspector
```

### Step 2 — Create a Modal account and install the CLI

Go to [modal.com](https://modal.com) and sign up for a free account, then run:

```bash
pip install modal
modal setup
```

`modal setup` opens your browser to complete authentication. This is a one-time
step — you will not need to touch Modal again after this.

### Step 3 — Upload the model files to Modal (one-time)

Download the model files from the
[latest release](https://github.com/yourname/cranberry-inspector/releases/latest)
and place them in a `models/` folder:

```
models/
├── backbone.pth
├── svm_clf.pkl
└── sam3_root/          ← SAM3 weights and assets
```

Then upload them to Modal's cloud storage:

```bash
python upload_models.py
```

This uploads once and reuses the files for all future deployments.

### Step 4 — Deploy the inference API to Modal

```bash
modal deploy modal_app.py
```

Modal will print a URL that looks like:

```
https://yourname--cranberry-inspector-predict.modal.run
```

Copy it.

### Step 5 — Configure the app

```bash
cp .env.example .env
```

Open `.env` and paste your Modal URL:

```
MODAL_INFERENCE_URL=https://yourname--cranberry-inspector-predict.modal.run
```

### Step 6 — Start the app

```bash
docker compose up --build
```

The first run downloads and builds the Docker images (~2 minutes).
Subsequent starts take a few seconds.

Open **http://localhost:3000** in your browser.

---

## Using the app

1. Click **Upload Image** and select a cranberry photo.
2. Wait ~30 seconds on the first request (Modal cold start — GPU spins up).
   Subsequent requests take a few seconds.
3. Each berry is outlined in **green** (ripe) or **red** (rot).
4. Hover over a berry to see its exact rot/ripe probability.
5. The summary panel on the right shows total counts and rot percentage.

---

## Stopping the app

```bash
docker compose down
```

Your Modal deployment stays live in the cloud and will respond to requests
whenever the app is running. Modal automatically scales to zero when idle
so you are not charged for time between requests.

---

## Troubleshooting

**"No cranberries detected"**
The image may be too dark, blurry, or at an unusual angle. Try a well-lit
photo taken from directly above the cranberries.

**First request takes a long time**
This is normal — Modal spins up a GPU container on the first request after
a period of inactivity. Subsequent requests in the same session are fast.

**Docker Desktop not running**
Make sure Docker Desktop is open before running `docker compose up`.

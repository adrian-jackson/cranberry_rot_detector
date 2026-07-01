# upload_models.py — run once from your machine (windows only due to forward slashes)
import modal

volume = modal.Volume.from_name("cranberry-models", create_if_missing=True)

with volume.batch_upload() as batch:
    batch.put_file("backend/models/backbone.pth", "/backbone.pth")
    batch.put_file("backend/models/svm_clf.pkl",  "/svm_clf.pkl")
    # SAM3 assets if they can't be installed via pip
    batch.put_directory("backend/models/SAM_assets","/SAM_assets")

print("Models uploaded.")
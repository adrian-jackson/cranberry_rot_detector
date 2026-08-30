# upload_models.py — run once from your machine (windows only due to forward slashes)
import modal

volume = modal.Volume.from_name("cranberry-models", create_if_missing=True)

with volume.batch_upload(force=False) as batch:
    #already installed
    #volume.remove_file("/backbone.pth")
    #volume.remove_file("/svm_clf.pkl")
    batch.put_file("C:/Users/Adria/Documents/github/cranberry_algorithms/cranberry_rot_detector/backend/models/backbone.pth", "/backbone.pth" )
    batch.put_file("C:/Users/Adria/Documents/github/cranberry_algorithms/cranberry_rot_detector/backend/models/svm_clf.pkl",  "/svm_clf.pkl")
    # SAM3 assets if they can't be installed via pip
    # batch.put_directory("backend/models/sam3_assets","/sam3_assets")
    #batch.put_file(f"backend/models/SAM_weights/sam3.pt","/sam3_weights/sam3.pt")
    #batch.put_file(f"backend/models/SAM_weights/config.json", "/sam3_weights/config.json")
    #to be installed

print("Models uploaded.")
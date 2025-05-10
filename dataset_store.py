import os
from fastapi import UploadFile
import shutil

DATASET_DIR = "static/datasets"

os.makedirs(DATASET_DIR, exist_ok=True)

def save_uploaded_dataset(file: UploadFile):
    file_path = os.path.join(DATASET_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file.filename

def list_available_datasets():
    return [f for f in os.listdir(DATASET_DIR) if os.path.isfile(os.path.join(DATASET_DIR, f))]

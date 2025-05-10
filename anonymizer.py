from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import random
import os

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# --- Anonymization techniques ---
def hash_data(value):
    return hash(value)

def generalize_column(value):
    if isinstance(value, int):
        return value // 10 * 10
    elif isinstance(value, float):
        return round(value, -1)
    elif isinstance(value, str):
        return value[:3] + '...'
    return value

def mask_data(value):
    s = str(value)
    return s[:2] + '*' * (len(s) - 2) if len(s) > 2 else '*' * len(s)

def scramble_data(value):
    return ''.join(random.sample(str(value), len(str(value)))) if value else value

def anonymize_column(column: pd.Series, technique: str) -> pd.Series:
    technique_map = {
        "masking": lambda col: col.apply(mask_data),
        "pseudonymization": lambda col: col.apply(lambda x: f"pseudo_{hash(x)}"),
        "generalization": lambda col: col.apply(generalize_column),
        "suppression": lambda col: col.apply(lambda _: None),
        "synthetic": lambda col: col.apply(lambda x: f"synthetic_{random.randint(1000,9999)}"),
        "hashing": lambda col: col.apply(hash_data),
        "k_anonymity": lambda col: col,  # Placeholder
        "data_swapping": lambda col: col.sample(frac=1).reset_index(drop=True),
        "scrambling": lambda col: col.apply(scramble_data),
        "shuffling": lambda col: col.sample(frac=1).reset_index(drop=True),
    }
    return technique_map.get(technique, lambda col: col)(column)

# --- Routes ---
@router.post("/anonymize_and_download", response_class=HTMLResponse)
async def anonymize_and_download(
    request: Request,
    file: UploadFile = File(...),
    technique: str = Form("masking")  # Can be set via form input
):
    df = pd.read_csv(file.file)
    for col in df.columns:
        df[col] = anonymize_column(df[col], technique)

    output_file = "anonymized_result.csv"
    output_path = os.path.join("static", output_file)
    df.to_csv(output_path, index=False)

    return templates.TemplateResponse("anonymize_and_download.html", {
        "request": request,
        "filename": output_file,
        "table": df.head().to_html(classes="table", index=False)
    })

@router.get("/download/{filename}")
async def download_file(filename: str):
    path = os.path.join("static", filename)
    if os.path.exists(path):
        return FileResponse(path=path, filename=filename, media_type='text/csv')
    return {"error": "File not found"}

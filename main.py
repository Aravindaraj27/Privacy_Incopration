from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import shutil
import os
import pickle
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import traceback

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import utilities safely
try:
    from utils.anonymization_utils import (
        anonymize_column,
        check_residual_risk,
        apply_differential_privacy,
    )
except ImportError as e:
    raise ImportError(f"Failed to import anonymization utilities. Original error: {str(e)}")

# Initialize FastAPI app
app = FastAPI()

# Middleware for catching all errors
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# File paths
UPLOAD_FOLDER = 'uploads/uploaded.csv'
DOWNLOAD_FOLDER = 'downloads/anonymized_data.csv'
MODEL_FOLDER = 'downloads/trained_model.pkl'

# Ensure folders exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('downloads', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Technique icons
technique_emojis = {
    "masking": "🎭",
    "differential_privacy": "🔐",
    "generalization": "📦",
    "pseudonymization": "🕵️",
    "suppression": "🚫",
}

technique_descriptions = {
    "masking": "Hides parts of data by replacing with random characters.",
    "differential_privacy": "Adds noise to data to prevent individual identification.",
    "generalization": "Broadens specific values into categories.",
    "pseudonymization": "Replaces identifiers with pseudonyms.",
    "suppression": "Completely removes sensitive data.",
}

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Upload CSV
@app.post("/upload", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile):
    try:
        if not file.filename.endswith('.csv'):
            raise ValueError("Only CSV files are allowed.")

        max_size = 50 * 1024 * 1024  # 50MB limit
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size > max_size:
            raise ValueError("File too large. Max size is 50MB.")

        with open(UPLOAD_FOLDER, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df = pd.read_csv(UPLOAD_FOLDER)
        preview = df.head().to_html(classes="preview-table", index=False)
        columns = df.columns.tolist()

        return templates.TemplateResponse("select_columns.html", {
            "request": request,
            "columns": columns,
            "preview": preview,
        })
    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e)})

# Anonymize selected columns
@app.post("/anonymize", response_class=HTMLResponse)
async def anonymize_columns(
    request: Request,
    columns: List[str] = Form(...),
    apply_dp: bool = Form(False),
    epsilon: float = Form(1.0)
):
    try:
        df = pd.read_csv(UPLOAD_FOLDER)

        invalid_cols = [col for col in columns if col not in df.columns]
        if invalid_cols:
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error": f"Invalid columns: {', '.join(invalid_cols)}"
            })

        applied_techniques = {}

        for col in columns:
            df[col] = anonymize_column(df[col], col)
            applied_techniques[col] = "masking"

        if apply_dp:
            df = apply_differential_privacy(df, epsilon=epsilon)
            for col in df.columns:
                if col not in applied_techniques:
                    applied_techniques[col] = "differential_privacy"

        df.to_csv(DOWNLOAD_FOLDER, index=False)
        preview = df.head().to_html(classes="preview-table", index=False)

        return templates.TemplateResponse("anonymize_and_download.html", {
            "request": request,
            "table": preview,
            "columns": applied_techniques.items(),
            "technique_emojis": technique_emojis,
            "technique_descriptions": technique_descriptions,
            "filename": "anonymized_data.csv"
        })

    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e)})

# Download anonymized CSV
@app.get("/download")
async def download_anonymized():
    return FileResponse(DOWNLOAD_FOLDER, media_type='text/csv', filename="anonymized_data.csv")

# Download trained model
@app.get("/download_model")
async def download_model():
    return FileResponse(MODEL_FOLDER, media_type='application/octet-stream', filename="trained_model.pkl")

# Train model from anonymized data
@app.post("/train_model_result", response_class=HTMLResponse)
async def train_model_result(request: Request, model_choice: str = Form(...)):
    try:
        df = pd.read_csv(DOWNLOAD_FOLDER)

        if len(df.columns) < 2:
            raise ValueError("Dataset must have at least two columns (features and target).")

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X = pd.get_dummies(X)

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "RandomForest":
            model = RandomForestClassifier()
        elif model_choice == "LogisticRegression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "SVM":
            model = SVC(probability=True)
        else:
            raise ValueError("Invalid model selected.")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        accuracy = accuracy_score(y_test, y_pred)

        with open(MODEL_FOLDER, "wb") as f:
            pickle.dump(model, f)

        # Confusion Matrix
        cm_path = "static/confusion_matrix.png"
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(cm_path)
        plt.close()

        # ROC Curve
        roc_path = None
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            roc_path = "static/roc_curve.png"
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.savefig(roc_path)
            plt.close()

        return templates.TemplateResponse("training_result.html", {
            "request": request,
            "model_choice": model_choice,
            "accuracy": f"{accuracy:.2%}",
            "cm_path": f"/{cm_path}",
            "roc_path": f"/{roc_path}" if roc_path else None,
        })

    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e)})

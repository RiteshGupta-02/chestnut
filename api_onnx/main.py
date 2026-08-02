from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import io
from pathlib import Path
import sys
import onnxruntime as ort
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cam_onnx import build_cam_response, preprocess_image, generate_cam_onnx,stable_sigmoid



ALL_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
from contextlib import asynccontextmanager

model_store = {}  # global dict to hold model between requests


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    # Full model — for /predict
    model_store["predict_session"] = ort.InferenceSession(
        Path(__file__).parent.parent/"src/checkpoint/chexnet_full.onnx",
        providers=["CPUExecutionProvider"]
    )

    # Feature extractor — for /cam
    model_store["feature_session"] = ort.InferenceSession(
        Path(__file__).parent.parent /"src/checkpoint/chexnet_features.onnx",
        providers=["CPUExecutionProvider"]
    )

    # Classifier weights — for CAM dot product
    model_store["classifier_weights"] = np.load(
        Path(__file__).parent.parent /"src/checkpoint/classifier_weights.npy"
    )

    print("All models loaded ")
    yield
    model_store.clear()

app = FastAPI(lifespan=lifespan)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(f"{FRONTEND_DIR}/index.html")

@app.get("/health")
async def get_health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read uploaded bytes → PIL Image → tensor
    # 2. Run through model → sigmoid → probabilities
    # 3. Build dict of {disease: probability}
    # 4. Find top finding
    # 5. Return JSONResponse
    # image_tensor = load_single_image(Path(file))
    image_bytes = await file.read()
    
    # 2. Open image from memory bytes
    # image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor_img = preprocess_image(image_bytes)
    
    # 3. Convert to PyTorch Tensor (Shape: C, H, W)
    model = model_store['predict_session']

    output = model.run(
        None,
        {"image": tensor_img},
    )

    output = stable_sigmoid(output[0])
    
    prob = {disease : float(probabilities) for disease,probabilities in zip(ALL_DISEASES, output[0]) }
    top_finding_name = max(prob, key=prob.get)
    top_finding_prob = prob[top_finding_name]
    
    if top_finding_prob < 0.5:
        top_finding   = "No Finding"
        top_prob      = top_finding_prob
        is_normal     = True
    else:
        top_finding   = top_finding_name
        top_prob      = top_finding_prob
        is_normal     = False

    jsonresponse = {
    "predictions": prob,
    "top_finding": top_finding_name,
    "top_probability": float(top_finding_prob),
    "is normal" : is_normal,
    }
    return JSONResponse(status_code=201, content=jsonresponse)

@app.post("/cam")
async def cam_endpoint(
    file: UploadFile = File(...),
    disease: str = Query(default="Pneumonia")
):
    if disease not in ALL_DISEASES:
        raise HTTPException(400, f"Unknown disease. Choose from: {ALL_DISEASES}")

    image_bytes = await file.read()
    image_array = preprocess_image(image_bytes)   # from cam_onnx.py

    target_idx = ALL_DISEASES.index(disease)

    # Get predictions first (needed for panel 3)
    logits = model_store["predict_session"].run(
        ["logits"], {"image": image_array}
    )[0]
    probs = 1 / (1 + np.exp(-logits[0]))   # sigmoid in numpy
    predictions = {d: float(p) for d, p in zip(ALL_DISEASES, probs)}

    # Generate CAM
    cam_map = generate_cam_onnx(
        feature_session=model_store["feature_session"],
        classifier_weights=model_store["classifier_weights"],
        image_array=image_array,
        target_class_idx=target_idx,
    )

    # Build and return PNG
    png_bytes = build_cam_response(image_array, cam_map, predictions, disease)

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png"
    )
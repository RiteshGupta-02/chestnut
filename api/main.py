from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
import io
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import get_model
from cam import generate_cam, overlay_cam_image, load_single_image, INFERENCE_TRANSFORM, visualize_prediction
from chestxray_dataset import ALL_DISEASES

def get_health():
    pass


from contextlib import asynccontextmanager

model_store = {}   # global dict to hold model between requests
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs ONCE when server starts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_model(num_classes=14, device=device)
    ckpt   = torch.load("../src/checkpoint/checkpoint_epoch8.tar", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model_store["model"]  = model
    model_store["device"] = device
    yield
    # runs ONCE when server shuts down — cleanup here if needed
    model_store.clear()

app = FastAPI(lifespan=lifespan)
transform = transforms.ToTensor()

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
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = INFERENCE_TRANSFORM(image)
    
    # 3. Convert to PyTorch Tensor (Shape: C, H, W)
    tensor_img = image.unsqueeze(0)
    tensor_img = tensor_img.to(model_store['device'])
    model = model_store['model']
    prob = None
    top_finding_prob = None
    top_finding_name = None
    with torch.no_grad():
        output = model(tensor_img)
        output.sigmoid_()
        output = output.cpu().numpy()
        prob = {disease : float(probabilities*100) for disease,probabilities in zip(ALL_DISEASES, output[0]) }
        top_finding_prob = max(output[0])*100
    
    for k, v in prob.items():
        if v == top_finding_prob:
            top_finding_name = k
    

    jsonresponse = {
    "predictions": prob,
    "top_finding": top_finding_name,
    "top_probability": float(top_finding_prob)
    }
    return JSONResponse(status_code=201, content=jsonresponse)

@app.post("/cam")
async def cam(file: UploadFile = File(...), disease: str = "Pneumonia"):
    # 1. Validate disease name is in ALL_DISEASES
    # 2. Read image → tensor
    # 3. Call generate_cam() → cam_map
    # 4. Call overlay_cam_on_image() → overlay_img
    # 5. Save to bytes buffer → return as StreamingResponse PNG
    if disease not in ALL_DISEASES:
        return None
    image_bytes = await file.read()

    model = model_store['model']
    device = model_store["device"]
    disease_idx = ALL_DISEASES.index(disease)

    # 2. Open image from memory bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = INFERENCE_TRANSFORM(image)
    
    # 3. Convert to PyTorch Tensor (Shape: C, H, W)
    tensor_img = image.unsqueeze(0)
    print(tensor_img.shape)

    # batched = tensor_img.to(model_store['device'])

    
    cam_map, target_prob = generate_cam(model,tensor_img,disease_idx,device)

    overlay_img, _ = overlay_cam_image(image, cam_map, alpha=0.4)

    buf = io.BytesIO()
    Image.fromarray(overlay_img).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")



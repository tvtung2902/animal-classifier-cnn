from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import torch
import numpy as np
import cv2

from model import SimpleCNN
from animal_dataset import AnimalDataset

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SimpleCNN(num_classes=10).to(device)
checkpoint = torch.load("trained_models/best_cnn.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

categories = AnimalDataset.get_categories()

def preprocess_image(image_bytes, image_size=224):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    image = np.transpose(image, (2, 0, 1)) / 255.0
    image = image[None, :, :, :]
    return torch.tensor(image, dtype=torch.float32).to(device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image_tensor = preprocess_image(contents)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        pred_idx = torch.argmax(probs).item()
        predicted_label = categories[pred_idx]
        confidence = probs[pred_idx].item()

    return JSONResponse({
        "prediction": predicted_label,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
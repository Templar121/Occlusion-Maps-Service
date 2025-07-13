from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import uuid
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Constants ---
label_mapping = {0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc', 4: 'akiec', 5: 'vasc', 6: 'df'}
class_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
MODEL_PATH = Path(__file__).parent / "model" / "best_model.keras"
TEMP_DIR = Path(__file__).parent / "temp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

# --- Load Model ---
model = load_model(str(MODEL_PATH), compile=False)

# --- Utility Functions ---
def preprocess_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def preprocess_image_from_array(img_array):
    img = tf.image.resize(img_array, (64, 64)).numpy().astype(np.float32)
    return np.expand_dims(img, axis=0)

def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

@app.get("/health")
def health_check():
    return {"status": "API is healthy ✅"}

@app.post("/explain")
async def explain_with_occlusion(file: UploadFile = File(...)):
    temp_path = TEMP_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    try:
        # Save uploaded file to disk
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Load and process image
        img = Image.open(temp_path).convert('RGB').resize((64, 64))
        arr = np.asarray(img, dtype=np.float32)
        input_tensor = preprocess_image_from_array(arr)

        preds = model.predict(input_tensor, verbose=0)[0]
        class_index = int(np.argmax(preds))
        confidence = preds[class_index]

        # Occlusion logic
        patch_size = 15
        stride = 8
        mean_pixel = np.mean(arr, axis=(0, 1), keepdims=True)
        heatmap = np.zeros((64, 64), dtype=np.float32)

        for y in range(0, 64 - patch_size + 1, stride):
            for x in range(0, 64 - patch_size + 1, stride):
                occluded = arr.copy()
                occluded[y:y+patch_size, x:x+patch_size] = mean_pixel
                pred = model.predict(preprocess_image_from_array(occluded), verbose=0)[0]
                drop = confidence - pred[class_index]
                heatmap[y:y+patch_size, x:x+patch_size] = drop

        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(arr.astype(np.uint8), alpha=0.6)
        ax.imshow(heatmap, cmap='jet', alpha=0.5)
        ax.set_title(f"Occlusion Map ({class_names[class_index]})")
        ax.axis('off')

        return {"occlusion_base64": fig_to_base64(fig)}

    except Exception as e:
        return {"error": str(e)}

    finally:
        try:
            temp_path.unlink()  # Delete temp file
        except Exception:
            pass

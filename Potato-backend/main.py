from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from pymongo import MongoClient
import uvicorn
app = FastAPI()

# Allow Android emulator access
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://10.0.2.2:8080",      # For emulator
    "http://192.168.50.23:8080", # Your laptop IP
    "http://192.168.50.231"       # Your phone IP
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TensorFlow model
MODEL = tf.keras.models.load_model("C:/Users/User/Desktop/lecture_python/PotatoDiseaseApp/backend/saved_models/1")  # Adjust path
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# MongoDB setup (optional)
client = MongoClient("mongodb://localhost:27017/")
db = client["prediction_db"]
collection = db["predictions"]

@app.get("/ping")
async def ping():
    return {"message": "FastAPI is running"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    # Convert RGBA to RGB if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize((256, 256))
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = read_file_as_image(contents)
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    result = {
        "filename": file.filename,
        "class": predicted_class,
        "confidence": confidence
    }
    collection.insert_one(result)  # Optional: Store in MongoDB

    return {"className": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
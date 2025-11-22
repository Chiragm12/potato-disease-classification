# uvicorn main:app --reload --host 0.0.0.0 --port 8001
#run this for main.py the backend, and for frontend run npm start in the frontend directory
# Make sure to have the model saved in the correct path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost:3000",  # React frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model using SavedModel format (compatible with Keras 3)
LOADED_MODEL = tf.saved_model.load("../models/1")
if 'serving_default' not in LOADED_MODEL.signatures:
    raise ValueError(f"Model does not have 'serving_default' signature. Available: {list(LOADED_MODEL.signatures.keys())}")
MODEL = LOADED_MODEL.signatures['serving_default']
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
# Get the output key from the model signature
output_keys = list(MODEL.structured_outputs.keys())
if not output_keys:
    raise ValueError("Model signature has no output keys")
MODEL_OUTPUT_KEY = output_keys[0]

@app.get("/ping")
async def ping():
    return {"message": "Hello World"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).resize((256, 256))  # Resize to your model's input
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    # Convert to float32 as required by the SavedModel signature (TensorSpec expects float32)
    image_batch = np.expand_dims(image.astype(np.float32), 0)
    # Use the SavedModel signature interface (convert_to_tensor is more efficient than constant)
    predictions = MODEL(tf.convert_to_tensor(image_batch))
    # Extract the output tensor using the dynamic key
    output = predictions[MODEL_OUTPUT_KEY].numpy()
    predicted_class = CLASS_NAMES[np.argmax(output[0])]
    confidence = np.max(output[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

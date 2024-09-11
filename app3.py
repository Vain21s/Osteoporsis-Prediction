from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import os
from fastapi import Form


# Create directories for static files and templates
image_directory = "uploaded_images"
os.makedirs(image_directory, exist_ok=True)
templates_directory = "templates"

# Initialize FastAPI app
app = FastAPI()

# Serve images statically
app.mount("/images", StaticFiles(directory=image_directory), name="images")



# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This should be more restrictive in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 template rendering setup
templates = Jinja2Templates(directory=templates_directory)

# Load the TensorFlow model
class ModelManager:
    def __init__(self, model_paths):
        self.models = {name: self.load_model(path) for name, path in model_paths.items()}
        self.current_model = None
    
    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path)
    
    def set_model(self, model_name):
        self.current_model = self.models.get(model_name, None)
    
    def predict(self, preprocessed_image):
        if self.current_model is None:
            raise ValueError("No model is currently set.")
        return self.current_model.predict(preprocessed_image)

# Define model paths
model_paths = {
    "model1": r"C:\Users\Varun S\Downloads\osteoporosis_model (2).h5",
    "model2": r"C:\Users\Varun S\Downloads\vgg16_model.h5",
    "model3": r"C:\Users\Varun S\Downloads\Resnetmodel"
              r".h5",
}

model_manager = ModelManager(model_paths)
model_manager.set_model("model1")  # Set a default model

def preprocess_image(img):
    """
    Preprocess the image to fit the input requirements of the TensorFlow model and enhance image clarity:
    - Convert to grayscale
    - Apply Gaussian Blur to reduce noise
    - Use adaptive thresholding to enhance image contrast
    - Resize to the model's expected input size (e.g., 224x224)
    - Normalize pixel values to [0, 1]
    """
    img = img.convert('L')  # Convert image to grayscale
    img_array = np.array(img)  # Convert PIL image to numpy array

    # Apply Gaussian Blur
    img_blurred = cv2.GaussianBlur(img_array, (5, 5), 0)

    # Apply adaptive thresholding to enhance contrast
    img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

    # Resize image
    img_resized = cv2.resize(img_thresh, (224, 224), interpolation=cv2.INTER_AREA)

    # Normalize the image
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Expand dimensions to match model's input
    img_expanded = np.expand_dims(img_normalized, axis=0)  # for batch size
    img_expanded = np.expand_dims(img_expanded, axis=-1)   # for channel

    return img_expanded

#def predict(self, preprocessed_image):
      #  prediction = self.model.predict(preprocessed_image)
      #  return "Osteoporosis" if prediction[0][0] > 0.65 else "Not Osteoporosis"

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """ Serve the main page. """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def read_about(request: Request):
    """ Serve the main page. """
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/FAQ", response_class=HTMLResponse)
async def read_FAQ(request: Request):
    """ Serve the main page. """
    return templates.TemplateResponse("FAQ.html", {"request": request})


@app.get("/osteoporosis-specialist", response_class=HTMLResponse)
async def read_osteoporosis(request: Request):
    """ Serve the main page. """
    return templates.TemplateResponse("osteoporosis-specialist.html", {"request": request})

@app.get("/nutrition-specialist", response_class=HTMLResponse)
async def read_nutrition(request: Request):
    """ Serve the main page. """
    return templates.TemplateResponse("nutrition-specialist.html", {"request": request})


@app.get("/Treatment", response_class=HTMLResponse)
async def read_Treatment(request: Request):
    """ Serve the main page. """
    return templates.TemplateResponse("Treatment.html", {"request": request})


@app.get("/model1", response_class=HTMLResponse)
async def read_Treatment(request: Request):
 
    return templates.TemplateResponse("CNN.html", {"request": request})

@app.get("/model2", response_class=HTMLResponse)
async def read_Treatment(request: Request):

    return templates.TemplateResponse("VGG.html", {"request": request})

@app.get("/model3", response_class=HTMLResponse)
async def read_Treatment(request: Request):

    return templates.TemplateResponse("RESNET.html", {"request": request})



@app.post("/predict/", response_class=HTMLResponse)
async def create_upload_file(request: Request, file: UploadFile = File(...), model_choice: str = Form(...)):
    """ Handles image uploads and displays prediction results. """
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    image_path = os.path.join(image_directory, file.filename)
    image.save(image_path)  # Save image to static directory

    # Set the chosen model
    model_manager.set_model(model_choice)

    preprocessed_image = preprocess_image(image)
    prediction_result = model_manager.predict(preprocessed_image)
    prediction_result_text = "Osteoporosis" if prediction_result[0][0] > 0.65 else "Not Osteoporosis"

    return templates.TemplateResponse("result.html", {
        "request": request,
        "filename": file.filename,
        "prediction": prediction_result_text,
        "image_url": f"/images/{file.filename}"
    })
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

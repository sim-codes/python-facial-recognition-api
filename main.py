from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
import face_recognition
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import logging
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Facial Recognition API")

# Create a thread pool executor
executor = ThreadPoolExecutor(max_workers=4)

class ImageURL(BaseModel):
    url: HttpUrl
    
class FaceMatchRequest(BaseModel):
    known_face_url: HttpUrl
    unknown_face_url: HttpUrl
    tolerance: float = 0.6  # Default tolerance value

class FaceDetectionResponse(BaseModel):
    face_count: int
    face_locations: list
    
class FaceMatchResponse(BaseModel):
    matches: bool
    distance: float = None
    processing_time: float = None

# Cache image downloads to avoid re-downloading the same image
@lru_cache(maxsize=100)
def get_image_from_url(url: str):
    """Download an image from a URL and convert it to a numpy array for face_recognition."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        # Convert to RGB if image is in RGBA format (e.g., PNG with transparency)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        logger.error(f"Error downloading or processing image from {url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Could not process image from URL: {str(e)}")

def process_image_for_face_recognition(image):
    """Resize the image to speed up face recognition."""
    # Calculate new dimensions while maintaining aspect ratio
    h, w = image.shape[:2]
    new_width = min(w, 640)  # Limit width to 640 pixels
    new_height = int(h * (new_width / w))
    
    # Use cv2 for faster resizing
    small_image = cv2.resize(image, (new_width, new_height))
    
    # Convert from BGR to RGB if using OpenCV
    if len(small_image.shape) == 3 and small_image.shape[2] == 3:
        small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
        
    return small_image

def get_face_encoding(image_url):
    """Get face encoding from an image URL."""
    image = get_image_from_url(image_url)
    # Resize image for faster processing
    small_image = process_image_for_face_recognition(image)
    # Use model='small' for faster but slightly less accurate face detection
    face_locations = face_recognition.face_locations(small_image, model="cnn")
    if not face_locations:
        raise HTTPException(status_code=400, detail=f"No face found in the image: {image_url}")
    # Use first face found
    # Set num_jitters=1 (default) for speed, increase for accuracy at the cost of speed
    face_encodings = face_recognition.face_encodings(small_image, known_face_locations=[face_locations[0]], num_jitters=1)
    if not face_encodings:
        raise HTTPException(status_code=400, detail=f"Could not encode face in the image: {image_url}")
    return face_encodings[0]

@app.get("/")
def read_root():
    return {"message": "Facial Recognition API is running. Use /detect_faces or /match_faces endpoints."}

@app.post("/match_faces", response_model=FaceMatchResponse)
async def match_faces(request: FaceMatchRequest):
    """Check if two face images from URLs match."""
    start_time = time.time()
    try:
        # Process both images in parallel
        known_future = executor.submit(get_face_encoding, request.known_face_url)
        unknown_future = executor.submit(get_face_encoding, request.unknown_face_url)
        
        # Get results from futures
        known_face_encoding = known_future.result()
        unknown_face_encoding = unknown_future.result()
        
        # Compare faces
        face_distance = face_recognition.face_distance([known_face_encoding], unknown_face_encoding)[0]
        matches = face_distance <= request.tolerance
        
        processing_time = time.time() - start_time
        
        return {
            "matches": bool(matches),
            "distance": float(face_distance),
            "processing_time": processing_time
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error matching faces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error matching faces: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
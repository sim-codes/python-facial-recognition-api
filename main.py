from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Face Comparison API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development. In production, specify your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def load_image_into_numpy_array(data):
    image = Image.open(io.BytesIO(data))
    return np.array(image)

@app.post("/compare-faces/")
async def compare_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    # Check if files are images
    if not image1.content_type.startswith("image/") or not image2.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded files must be images.")
    
    try:
        print("Received Image1 Content-Type:", image1.content_type)
        print("Received Image2 Content-Type:", image2.content_type)

        # Read images
        image1_content = await image1.read()
        image2_content = await image2.read()
        
        # Convert to numpy arrays
        img1_array = await load_image_into_numpy_array(image1_content)
        img2_array = await load_image_into_numpy_array(image2_content)
        
        # Find faces in images
        img1_face_locations = face_recognition.face_locations(img1_array)
        img2_face_locations = face_recognition.face_locations(img2_array)
        
        # Check if faces were found
        if not img1_face_locations:
            return JSONResponse(
                status_code=400,
                content={"message": "No faces found in the first image."}
            )
        
        if not img2_face_locations:
            return JSONResponse(
                status_code=400,
                content={"message": "No faces found in the second image."}
            )
        
        # Generate face encodings
        img1_face_encodings = face_recognition.face_encodings(img1_array, img1_face_locations)
        img2_face_encodings = face_recognition.face_encodings(img2_array, img2_face_locations)
        
        # Compare faces - check if any face in img1 matches any face in img2
        matches = []
        for face1 in img1_face_encodings:
            for i, face2 in enumerate(img2_face_encodings):
                # Compare face encodings
                distance = face_recognition.face_distance([face1], face2)[0]
                match = face_recognition.compare_faces([face1], face2)[0]
                
                # Add match info
                matches.append({
                    "is_match": bool(match),
                    "confidence": float(1 - distance),  # Convert to confidence score
                    "face1_index": len(matches) // len(img2_face_encodings),
                    "face2_index": i
                })
        
        # Prepare response
        result = {
            "faces_in_image1": len(img1_face_locations),
            "faces_in_image2": len(img2_face_locations),
            "matches": matches,
            "match_found": any(match["is_match"] for match in matches)
        }
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Face Comparison API is running. Use /compare-faces/ endpoint to compare two images."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
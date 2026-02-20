from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from .schemas import VerifyResponse
from .model import verify_faces_from_bytes

app = FastAPI()

@app.post("/verify", response_model=VerifyResponse)
async def verify(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    model_name: str = Form(...)
):
    """
    Face verification API endpoint.

    Receives two face images and a model name, performs face verification
    using a preloaded model, and returns a similarity score.

    Parameters
    ----------
    image1 : UploadFile
        First face image.
    image2 : UploadFile
        Second face image.
    model_name : str
        Name of the face recognition model to use.

    Returns
    -------
    VerifyResponse
        JSON response containing the similarity score.
    """
    try:
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()

        similarity = verify_faces_from_bytes(img1_bytes, img2_bytes, model_name)
        return VerifyResponse(similarity=float(similarity))
    except Exception as e:
        import traceback
        import logging
        logging.getLogger(__name__).info("ERROR in verify_faces:", e)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
import numpy as np
import cv2
import logging
from insightface.app import FaceAnalysis
from app.utils import compute_similarity

logger = logging.getLogger(__name__)

logger.info("Loading models...")

models = {
    "buffalo_s": FaceAnalysis(name="buffalo_s"),
    "buffalo_l": FaceAnalysis(name="buffalo_l")
}

for name, model in models.items():
    logger.info(f"Preparing model: {name}")
    model.prepare(ctx_id=-1)  # GPU if possible
    logger.info(f"{name} loaded successfully")

logger.info("All models loaded.")


def get_embedding(model, image):
    """
    Extracts a face embedding from an input image.

    Parameters
    ----------
    model : FaceAnalysis
        Initialized InsightFace model.
    image : numpy.ndarray
        Input image in OpenCV format.

    Returns
    -------
    numpy.ndarray or None
        Face embedding vector if a face is detected,
        otherwise None.
    """
    faces = model.get(image)
    if len(faces) == 0:
        return None

    return faces[0].embedding

def verify_faces_from_bytes(img1_bytes, img2_bytes, model_name):
    """
    Verifies two face images using the selected model.

    The function decodes image bytes, extracts embeddings,
    and computes cosine similarity between them.

    Parameters
    ----------
    img1_bytes : bytes
        First image file as bytes.
    img2_bytes : bytes
        Second image file as bytes.
    model_name : str
        Model identifier ("buffalo_s" or "buffalo_l").

    Returns
    -------
    float
        Cosine similarity score. Returns 0.0 if a face
        is not detected in either image.
    """
    if model_name not in models:
        raise ValueError(f"Model {model_name} not available")

    model = models[model_name]

    # decode directly from bytes
    img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        raise ValueError("Failed to decode image(s)")

    emb1 = get_embedding(model, img1)
    emb2 = get_embedding(model, img2)

    if emb1 is None or emb2 is None:
        return 0.0
    
    similarity = compute_similarity(emb1, emb2)
    return float(similarity)
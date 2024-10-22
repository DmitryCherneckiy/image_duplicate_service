import uuid
import requests
import base64
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import List, Optional

from app.models.image_model import image_to_vector
from app.db import vector_db

router = APIRouter()
logger = logging.getLogger(__name__)


def process_image(image_bytes: bytes) -> List[float]:
    """
    Processes an image and converts it into a vector representation.

    Args:
        image_bytes (bytes): The image data in bytes.

    Returns:
        List[float]: The image vector.

    Raises:
        HTTPException: If the image size exceeds the limit or processing fails.
    """
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size exceeds 10 MB")

    try:
        vector = image_to_vector(image_bytes)
        return vector
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@router.post("/images")
async def add_images(request: Request, files: Optional[List[UploadFile]] = File(None)):
    """
    Adds images to the vector database. Supports images uploaded via multipart/form-data,
    Base64-encoded images, and images provided via URLs.

    Args:
        request (Request): The FastAPI request object.
        files (Optional[List[UploadFile]]): List of uploaded files (optional).

    Returns:
        dict: A dictionary containing the request ID, the number of images added,
              and their system-assigned names.

    Raises:
        HTTPException: If no valid images are provided or processing fails.
    """
    request_id = str(uuid.uuid4())
    vectors = []

    # Determine if the content type is JSON to read data accordingly
    if request.headers.get("Content-Type") == "application/json":
        try:
            data = await request.json()
            logger.info(f"Received JSON data: {data}")
        except Exception as e:
            logger.error(f"Error reading JSON data: {str(e)}")
            data = None
    else:
        data = None

    # Processing images from multipart/form-data
    if files:
        for file in files:
            if file.content_type not in ["image/jpeg", "image/png"]:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

            image_bytes = await file.read()
            vector = process_image(image_bytes)
            vectors.append(vector)

    # Processing Base64-encoded images
    if data and "base64_images" in data:
        for image_b64 in data["base64_images"]:
            try:
                image_bytes = base64.b64decode(image_b64)
                vector = process_image(image_bytes)
                vectors.append(vector)
            except Exception as e:
                logger.error(f"Error processing Base64 image: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing Base64 image: {str(e)}")

    # Processing images from URLs
    if data and "image_urls" in data:
        for image_url in data["image_urls"]:
            logger.info(f"Processing image from URL: {image_url}")
            try:
                response = requests.get(image_url, timeout=10)
                logger.info(f"URL: {image_url}, Status code: {response.status_code}, "
                            f"Content length: {len(response.content)}")

                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {image_url} "
                                                                f"(Status code: {response.status_code})")

                image_bytes = response.content
                if not image_bytes:
                    raise HTTPException(status_code=400, detail=f"Downloaded image is empty from URL: {image_url}")

                vector = process_image(image_bytes)
                vectors.append(vector)
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for URL: {image_url}, Error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing image from URL: {str(e)}")

    if not vectors:
        logger.info("No valid vectors were generated from the provided images.")
        raise HTTPException(status_code=400, detail="No valid images provided")

    vector_db.add_vectors(request_id, vectors)

    start_index = len(vector_db.vectors) - len(vectors)
    end_index = len(vector_db.vectors)
    image_names = [f"image_{i}" for i in range(start_index + 1, end_index + 1)]

    return {"request_id": request_id, "added_images": len(vectors), "system_image_names": image_names}


@router.get("/duplicates/{request_id}")
async def find_duplicates(request_id: str, threshold: float = 1.0, k: int = 3):
    """
    Finds duplicates for the images associated with a given request ID.

    Args:
        request_id (str): The unique ID associated with a set of images.
        threshold (float, optional): The distance threshold for considering images as duplicates. Defaults to 1.0.
        k (int, optional): The number of nearest neighbors to search for. Defaults to 3.

    Returns:
        dict: A dictionary containing the request ID and a list of duplicate image names,
              or a message indicating no duplicates were found.

    Raises:
        HTTPException: If the request ID is not found or an error occurs during search.
    """
    if request_id not in vector_db.request_map:
        raise HTTPException(status_code=404, detail="Request ID not found")

    start_index, _ = vector_db.request_map[request_id]

    try:
        duplicates_indices = vector_db.search_duplicates(request_id, k)
    except Exception as e:
        logger.error(f"Error during duplicate search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

    all_duplicates = set()

    for vector_key, results in duplicates_indices.items():
        indices = results["indices"]
        distances = results["distances"]

        current_vector_number = int(vector_key.split('_')[1])
        current_index = start_index + current_vector_number

        for i in range(len(indices)):
            if indices[i] != current_index and distances[i] <= threshold:
                all_duplicates.add(indices[i])

    if not all_duplicates:
        return {"message": "No duplicates found"}

    duplicate_image_names = [vector_db.get_image_name(idx) for idx in all_duplicates]

    return {"request_id": request_id, "duplicates": duplicate_image_names}

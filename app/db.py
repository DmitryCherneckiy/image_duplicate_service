import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class VectorDB:
    """
    A simple vector database using FAISS for similarity search.
    Manages image vectors, request IDs, and image names.
    """

    def __init__(self, dim: int):
        """
        Initializes the VectorDB with a given vector dimension.

        Args:
            dim (int): The dimension of the vectors to be stored.
        """
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)  # Flat L2 distance index
        self.vectors: List[List[float]] = []  # List to store vectors
        self.request_map: Dict[str, Tuple[int, int]] = {}  # Maps request_id to vector indices
        self.image_names: List[str] = []  # List of image names

    def add_vectors(self, request_id: str, vectors: List[List[float]]):
        """
        Adds vectors to the index and updates mappings.

        Args:
            request_id (str): The unique identifier for the request.
            vectors (List[List[float]]): A list of vectors to add.

        Raises:
            ValueError: If the vectors have incorrect dimensions.
        """
        vectors_np = np.array(vectors, dtype='float32')
        if vectors_np.ndim != 2 or vectors_np.shape[1] != self.dim:
            raise ValueError(f"Expected vector dimension {self.dim}, but got {vectors_np.shape[1]}")

        logger.info(f"Adding vectors for request_id: {request_id}, vectors shape: {vectors_np.shape}")

        # Add vectors to FAISS index
        self.index.add(vectors_np)

        # Update mappings and lists
        start_index = len(self.vectors)
        end_index = start_index + len(vectors)
        new_image_names = [f"image_{i}" for i in range(start_index + 1, end_index + 1)]

        self.image_names.extend(new_image_names)
        self.request_map[request_id] = (start_index, end_index)
        self.vectors.extend(vectors)

        logger.info(f"Total vectors in index after addition: {self.index.ntotal}")

    def search_duplicates(self, request_id: str, k: int = 3) -> Dict[str, Dict[str, List]]:
        """
        Searches for duplicates of vectors associated with a request ID.

        Args:
            request_id (str): The request ID to search duplicates for.
            k (int, optional): The number of nearest neighbors to retrieve. Defaults to 3.

        Returns:
            Dict[str, Dict[str, List]]: A dictionary containing distances and indices of nearest neighbors.

        Raises:
            ValueError: If the request ID is not found.
        """
        if request_id not in self.request_map:
            raise ValueError("Request ID not found")

        start_index, end_index = self.request_map[request_id]
        query_vectors = np.array(self.vectors[start_index:end_index], dtype='float32')
        logger.info(f"Searching duplicates for request_id: {request_id}, query vectors shape: {query_vectors.shape}")

        # Search for nearest neighbors
        distances, indices = self.index.search(query_vectors, k)
        logger.info(f"Search results for request_id: {request_id}, distances: {distances}, indices: {indices}")

        duplicates = {}
        for i, (dist, ind) in enumerate(zip(distances, indices)):
            duplicates[f"vector_{i}"] = {
                "indices": ind.tolist(),
                "distances": dist.tolist()
            }

        return duplicates

    def get_image_name(self, index: int) -> str:
        """
        Retrieves the image name corresponding to a given index.

        Args:
            index (int): The index of the image.

        Returns:
            str: The image name if found, else None.
        """
        if 0 <= index < len(self.image_names):
            return self.image_names[index]
        return None

    def reset(self):
        """
        Resets the vector database by clearing the index and internal data structures.
        """
        self.index.reset()
        self.vectors.clear()
        self.request_map.clear()
        self.image_names.clear()


# Initialize a global instance of VectorDB with vector dimension 2048
vector_db = VectorDB(2048)

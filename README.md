# Image Duplicate Detection Service

This project is a simple REST API service that detects duplicate images. It uses machine learning models to convert images into vector representations and then finds duplicates based on these vectors.

## Requirements

- Python 3.11
- Docker & Docker Compose (for containerized usage)

## Installation and Running Locally

1. **Clone the repository:**

   ```sh
   git clone https://github.com/DmitryCherneckiy/image_duplicate_service.git
   cd image_duplicate_service
   ```
2. **Create a virtual environment and activate it:**

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```
4. **Run the application:**

   ```sh
   uvicorn app.main:app --reload --port 8000
   ```
This will start the service at http://localhost:8000

## Running with Docker

**Build and run the container:**

   ```sh
   docker-compose up --build
   ```
This will start the service at http://localhost:8000

## Running tests
### Using Docker Compose
**To run the tests in a Docker container, you can use Docker Compose:**

   ```sh
   docker-compose run tests
   ```
### Running Locally
**To run the tests locally, use:**

   ```sh
   pytest tests/test_api.py
   ```

## API Endpoints
### 1. **Add Images**
- **Endpoint:** `POST /images`
- **Description:** Accepts a list of images and converts each to a vector representation.
- **Input:** Images in multipart/form-data, base64-encoded strings, or URLs.
- **Response:** Returns a `request_id`, information about successfully added images and system name of image.

### 2. **Find Duplicates**
- **Endpoint:** `GET /duplicates/{request_id}`
- **Description:** Searches for duplicate images based on a provided `request_id`.
- **Response:** Returns a list of system name of images that are duplicates or a message indicating that no duplicates were found.

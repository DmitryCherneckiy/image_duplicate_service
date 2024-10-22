import io
import base64
import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from app.main import app
from app.db import vector_db

# Constants for test images
TEST_IMAGE_DIR = 'test_images'
TEST_IMAGE_FILENAMES = ['test_image_1.jpg', 'test_image_2.jpg', 'test_image_3.jpg']


@pytest.fixture
def test_client():
    return TestClient(app)


# Fixture to reset the vector database before each test
@pytest.fixture(autouse=True)
def reset_vector_db():
    vector_db.reset()


@pytest.fixture
def test_images():
    """
    Fixture to load test images into memory.
    Returns a list of tuples containing filename and image content.
    """
    images = []
    for filename in TEST_IMAGE_FILENAMES:
        with open(f'{TEST_IMAGE_DIR}/{filename}', 'rb') as f:
            images.append((filename, f.read()))
    return images


def test_add_images_via_multipart(test_client, test_images):
    """
    Test adding images via multipart/form-data.
    """
    files = [('files', (filename, io.BytesIO(content), 'image/jpeg')) for filename, content in test_images]

    response = test_client.post("/images", files=files)
    assert response.status_code == 200
    data = response.json()
    assert 'request_id' in data
    assert data['added_images'] == len(test_images)
    assert len(data['system_image_names']) == len(test_images)


def test_add_images_via_base64(test_client, test_images):
    """
    Test adding images via Base64-encoded JSON data.
    """
    base64_images = [base64.b64encode(content).decode('utf-8') for _, content in test_images]

    json_data = {
        "base64_images": base64_images
    }

    response = test_client.post("/images", json=json_data)
    assert response.status_code == 200
    data = response.json()
    assert 'request_id' in data
    assert data['added_images'] == len(test_images)
    assert len(data['system_image_names']) == len(test_images)


def test_add_images_via_urls(test_client, test_images):
    """
    Test adding images via image URLs.
    """
    # Mock URLs for testing
    image_urls = [
        "http://example.com/test_image_1.jpg",
        "http://example.com/test_image_2.jpg",
        "http://example.com/test_image_3.jpg"
    ]

    # Map URLs to image content
    url_to_content = dict(zip(image_urls, [content for _, content in test_images]))

    # Mock function for requests.get
    def mock_requests_get(url, *args, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = url_to_content[url]
        return mock_response

    with patch('requests.get', side_effect=mock_requests_get):
        json_data = {
            "image_urls": image_urls
        }

        response = test_client.post("/images", json=json_data)
        assert response.status_code == 200
        data = response.json()
        assert 'request_id' in data
        assert data['added_images'] == len(test_images)
        assert len(data['system_image_names']) == len(test_images)


def test_find_duplicates_with_duplicates(test_client, test_images):
    """
    Test finding duplicates when duplicates are present.
    """
    # Add the same image twice
    content = test_images[0][1]
    files = [
        ('files', ('duplicate1.jpg', io.BytesIO(content), 'image/jpeg')),
        ('files', ('duplicate2.jpg', io.BytesIO(content), 'image/jpeg'))
    ]

    response = test_client.post("/images", files=files)
    assert response.status_code == 200
    data = response.json()
    request_id = data['request_id']
    assert data['added_images'] == 2

    # Search for duplicates
    response = test_client.get(f"/duplicates/{request_id}", params={'threshold': 0.0, 'k': 2})
    assert response.status_code == 200
    data = response.json()
    assert 'duplicates' in data
    assert len(data['duplicates']) >= 1


def test_find_duplicates_no_duplicates(test_client, test_images):
    """
    Test finding duplicates when no duplicates are present.
    """
    # Add different images
    files = [('files', (filename, io.BytesIO(content), 'image/jpeg')) for filename, content in test_images]

    response = test_client.post("/images", files=files)
    assert response.status_code == 200
    data = response.json()
    request_id = data['request_id']
    assert data['added_images'] == len(test_images)

    # Search for duplicates
    response = test_client.get(f"/duplicates/{request_id}", params={'threshold': 0.0, 'k': 2})
    assert response.status_code == 200
    data = response.json()
    assert 'message' in data
    assert data['message'] == 'No duplicates found'

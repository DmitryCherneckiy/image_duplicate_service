import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import io
from typing import List

# Initialize the pre-trained ResNet50 model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Define the image transformation pipeline once at the module level
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def image_to_vector(image_bytes: bytes) -> List[float]:
    """
    Converts image bytes into a feature vector using a pre-trained ResNet50 model.

    Args:
        image_bytes (bytes): The image data in bytes format.

    Returns:
        List[float]: A list of floats representing the image feature vector.

    Raises:
        IOError: If the image cannot be opened or read.
    """
    try:
        # Open the image from bytes
        image = Image.open(io.BytesIO(image_bytes))
    except IOError as e:
        raise IOError(f"Unable to open image: {e}")

    # Ensure the image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply the transformation pipeline
    tensor = transform(image).unsqueeze(0)

    # Get the feature vector without tracking gradients
    with torch.no_grad():
        vector = model(tensor)

    # Flatten the vector and convert to a list of floats
    return vector.flatten().numpy().astype('float32').tolist()

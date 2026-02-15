import torch
from src.preprocess import preprocess_image
from PIL import Image

def test_preprocess_output_shape(tmp_path):
    # Create dummy image
    img = Image.new("RGB", (300, 300), color="white")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    tensor = preprocess_image(str(img_path))

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
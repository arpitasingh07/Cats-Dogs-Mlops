from preprocess import preprocess_image
import numpy as np

def test_preprocess():
    tensor = preprocess_image("sample.jpg")
    assert tensor.shape == (1, 3, 224, 224)
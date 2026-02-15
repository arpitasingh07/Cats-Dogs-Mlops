import torch
from src.model import build_model

def test_model_output_shape():
    model = build_model()
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape[1] == 2
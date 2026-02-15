from inference import load_model

def test_model_load():
    model = load_model()
    assert model is not None
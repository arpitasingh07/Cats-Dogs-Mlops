import torch
from src.model import build_model
from src.preprocess import preprocess_image

def load_model(model_path="artifacts/best_model.pt"):
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def predict_image(image_path):
    model = load_model()
    img = preprocess_image(image_path)
    preds = model(img)
    prob = torch.softmax(preds, dim=1)
    label = torch.argmax(prob).item()
    return label, prob.tolist()
from torchvision import transforms
from PIL import Image
import torch

transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    return transform_224(img).unsqueeze(0)
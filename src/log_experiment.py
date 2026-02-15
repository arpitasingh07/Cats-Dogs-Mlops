import mlflow
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from src.model import build_model
from src.dataset import CatsDogsDataset
from src.preprocess import transform_224

# Force local tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("CatsVsDogs_Local")

device = "cpu"

# Load trained model
model = build_model()
model.load_state_dict(torch.load("artifacts/best_model.pt", map_location=device))
model.eval()

# Use small subset for fast evaluation
dataset = CatsDogsDataset("data/val", transform=transform_224)
loader = DataLoader(dataset, batch_size=32)

mlflow.start_run()

mlflow.log_param("model", "ResNet18")
mlflow.log_param("evaluation_device", "cpu")
mlflow.log_param("dataset_split", "validation")
mlflow.log_param("epochs_trained", 3)
mlflow.log_param("batch_size", 64)
mlflow.log_param("learning_rate", 0.001)
mlflow.log_param("optimizer", "Adam")
mlflow.log_param("loss_function", "CrossEntropyLoss")
mlflow.log_param("input_size", "224x224")
mlflow.log_param("data_augmentation", "HorizontalFlip + Rotation(10)")

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in loader:
        preds = model(imgs)
        preds = torch.argmax(preds, dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
mlflow.log_metric("validation_accuracy", acc)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

mlflow.log_artifact("confusion_matrix.png")
mlflow.log_artifact("artifacts/best_model.pt")

mlflow.end_run()

print("Experiment logged successfully.")
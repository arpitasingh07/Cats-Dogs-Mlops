import requests
import os

API_URL = "http://localhost:8000/predict"
EVAL_DIR = "eval_samples"

def infer_image(img_path):
    """Send one image to FastAPI prediction endpoint."""
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
        response = requests.post(API_URL, files=files)
    return response.json()

def evaluate():
    total = 0
    correct = 0

    print("\n Running Post-Deployment Evaluation.\n")

    for filename in os.listdir(EVAL_DIR):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        # true labels: 0=cat, 1=dog
        true_label = 0 if "cat" in filename.lower() else 1

        img_path = os.path.join(EVAL_DIR, filename)
        result = infer_image(img_path)

        print("RAW RESPONSE:", result)

        if "prediction" not in result:
            print(f" ERROR: API returned no prediction for {filename}")
            continue

        predicted_name = result["prediction"]
        pred_label = 0 if predicted_name == "cat" else 1

        print(f"File: {filename} | True: {true_label} | Predicted: {pred_label}")

        total += 1
        if pred_label == true_label:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    print("\n===============================")
    print(f"   Evaluation Complete")
    print(f"   Total Samples: {total}")
    print(f"   Correct: {correct}")
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    print("===============================\n")

if __name__ == "__main__":
    evaluate()
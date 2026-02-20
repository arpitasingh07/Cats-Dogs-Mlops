
# Cats and Dogs – End‑to‑End MLOps Pipeline (Assignment 2)

Use case: Binary image classification (Cats vs Dogs) for a pet adoption platform

---

## 1. Project Goal
This project provides an end-to-end MLOps pipeline for binary image classification (Cats vs Dogs) for a pet adoption platform using machine learning.
It covers data preprocessing, model training, experiment tracking, data and model versioning, testing, CI/CD automation, containerization, deployment, and monitoring using open-source tools.

---

## 2. Project Overview
- Dataset: Cats vs Dogs Image Classification Dataset (Kaggle)
- Model used: ResNet18 CNN (Transfer Learning)
- Frameworks: PyTorch, Torchvision, FastAPI
- MLOps tools: DVC, Pytest, GitHub Actions, Docker, Kubernetes
- Deployment: Docker Compose & Kubernetes
- Monitoring: Application logging and request latency tracking

---

## 3. DataSet Information
 - Dataset: Cats and Dogs Binary Classification Dataset (Kaggle)
 - Pre‑processing:Images are resized to 224 × 224 RGB
 - Train , Validation , Test split (80 , 10 , 10)
 - Data augmentation is used for generalization
 - Versioning: Dataset and processed artifacts are tracked using DVC

---

## 4. Assignment Coverage (M1–M5)

 ### M1 : Model Development & Experiment Tracking 
 - Source code is versioned using Git
 - Dataset and features versioned using DVC
 - Baseline CNN model implemented
 - Trained model stored in serialized format (.pt / .h5 / .pkl)
 - MLflow used to log parameters, metrics, and artifacts

 ### M2 : Model Packaging & Containerization 
 - Inference service built using FastAPI
 - REST endpoints:GET/health – service status
 - POST/predict – image classification
 - Dependencies pinned in requirements.txt
 - Application containerized using Docker

 ### M3 : CI Pipeline 
 - Unit tests written using pytest for preprocessing and inference
 - GitHub Actions CI pipeline:Install dependencies
 - Run tests
 - Build Docker image
 - Push image to registry

 ### M4 : CD Pipeline & Deployment 
 - Deployment supported via:Docker Compose (local)(docker-compose.yaml)
 - Kubernetes (manifests provided)(k8s-deploy.yaml)
 - Automatic deployment on main branch updates
 - Post‑deployment smoke tests (health + prediction)

 ### M5 : Monitoring & Final Submission 
 - Request/response logging enabled
 - Metrics exposed (request count, latency)
 - Prometheus & Grafana configuration provided
 - Final submission includes code, configs, artifacts, and demo video

## 5. Repository Structure (Aligned to This Repo)
```
.
├── .github/workflows        # CI/CD pipelines (GitHub Actions)
├── api                      # FastAPI inference service
├── src                      # Training, preprocessing, utilities
├── tests                    # pytest unit tests
├── data                     # DVC‑tracked datasets
├── models                   # Trained model artifacts
├── monitoring               # Prometheus & Grafana configs
├── k8s                      # Kubernetes manifests
├── reports                  # Assignment report & results
├── scripts                  # Helper / automation scripts
├── docs                     # Detailed documentation
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Local deployment
├── dvc.yaml                 # DVC pipeline stages
├── params.yaml              # Training parameters
├── requirements.txt         # Pinned dependencies
├── QUICKSTART.md            # Quick setup guide
├── RUN_STEPS.md             # Step‑by‑step execution
├── VERIFICATION.md          # Runtime & deployment evidence
└── README.md                # Project documentation

```
## 5. How to Run

### Install dependencies
```
pip install -r requirements.txt

```

### Train model

```
python -m src.train

```

 ### Run API locally

 ```

uvicorn api.main:app --host 0.0.0.0 --port 8000

```

 ### Run with Docker Compose

```

docker-compose up --build

```

 ### Smoke test
```
curl http://localhost:8000/health
# curl -X POST http://localhost:8000/predict -F "file=@sample.jpg"

```




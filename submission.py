import requests
import torch
import torch.nn as nn
from torchvision import models
import os

# Make sure this directory exists and contains your trained model
model_path = "saved_models/PGD_madry_model.pt"
assert os.path.exists(model_path), f"Model not found at {model_path}"

# Load model
allowed_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}
model_name = "resnet18"  # <--- Make sure this matches your trained model

model = allowed_models[model_name](weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes

# Load trained weights
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=True)
model.eval()

# Check model output shape
dummy_input = torch.randn(1, 3, 32, 32)
out = model(dummy_input)
assert out.shape == (1, 10), f"Invalid output shape: {out.shape}"

# Submit to evaluation server
response = requests.post(
    "http://34.122.51.94:9090/robustness",
    files={"file": open(model_path, "rb")},
    headers={
        "token": "09596680",  # <--- your team token
        "model-name": model_name,
    }
)

print(response.json())

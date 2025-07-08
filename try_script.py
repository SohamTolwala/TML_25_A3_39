# import torch
# import torch.nn as nn
# from torchvision import models
# from models.resnet_wrapper import ResNetWrapper  # your wrapper

# # === Load your trained wrapper model ===
# wrapper_model = ResNetWrapper("resnet18", num_classes=10)
# wrapper_model.load_state_dict(torch.load("saved_models\\PGD_madry_model_optim.pt", map_location="cpu"))
# wrapper_model.eval()

# # === Init clean torchvision resnet18 ===
# submission_model = models.resnet18(weights=None)
# submission_model.fc = nn.Linear(submission_model.fc.in_features, 10)

# # === Copy weights from wrapper to plain resnet18 ===
# # This works only if the architectures match exactly (which they do in your case)
# submission_model.load_state_dict(wrapper_model.state_dict(), strict=True)

# # === Save new clean model ===
# torch.save(submission_model.state_dict(), "submission_resnet18_clean.pt")
# print("✅ Saved clean model for submission.")




import requests

response = requests.post(
    "http://34.122.51.94:9090/robustness",
    files={"file": open("saved_models\\PGD_8020.pt", "rb")},
    headers={"token": "09596680", "model-name": "resnet18"}  # <- important!
)

print(response.status_code)
print(response.json())


# import requests, torch, torch.nn as nn
# from torchvision import models

# MODEL_PATH = "saved_models\\PGD_madry_model_optim_ggg.pt"  # <- your trained file
# TOKEN      = "09596680"                                   # <- your token

# model = models.resnet18(weights=None)
# model.fc = nn.Linear(model.fc.in_features, 10)
# state = torch.load(MODEL_PATH, map_location="cpu")        # MUST load strict=True
# model.load_state_dict(state, strict=True)
# model.eval()                                              # just to be safe

# # Upload
# with open(MODEL_PATH, "rb") as f:
#     r = requests.post(
#         "http://34.122.51.94:9090/robustness",
#         files={"file": f},
#         headers={"token": TOKEN, "model-name": "resnet18"},
#         timeout=600,
#     )
# print(r.status_code, r.text)


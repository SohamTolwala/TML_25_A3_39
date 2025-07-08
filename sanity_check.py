import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from models.resnet_wrapper import ResNetWrapper
from torchvision.transforms.functional import to_pil_image
# # --- Begin: Dataset class definitions needed for torch.load ---
import torch
from torch.utils.data import Dataset
from typing import Tuple

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)
# --- End: Dataset class definitions ---



from PIL import Image

class DataWrapper(Dataset):
    """Wraps provided TaskDataset (idx, PIL.Image, label) → (Tensor, label)."""
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        rec = self.base[idx]
        if len(rec) == 3:
            _, img, label = rec
        elif len(rec) == 2:
            img, label = rec
        else:
            raise ValueError(f"Unexpected tuple length {len(rec)}")

        # Convert to RGB if not already
        if isinstance(img, torch.Tensor):
            if img.max() > 1: img = img.float() / 255.
            img = to_pil_image(img)

        if img.mode != "RGB":
            img = img.convert("RGB")

        img = self.transform(img)
        return img, int(label)


# # ───────────────────────────────  your DataWrapper here
#       # or inline class
raw_ds = torch.load("data/Train.pt", map_location="cpu", weights_only=False)
val_set = DataWrapper(raw_ds, transform=transforms.ToTensor())  # only ToTensor!
val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

# model = models.resnet18(weights=None)
# model.fc = nn.Linear(model.fc.in_features, 10)
# state = torch.load("saved_models\\PGD_madry_model_optim.pt", map_location="cpu", weights_only=True)
# model.load_state_dict(state, strict=True)
# model.eval()

# correct, total = 0, 0
# with torch.no_grad():
#     for x, y in val_loader:
#         out = model(x)
#         correct += (out.argmax(1) == y).sum().item()
#         total   += y.size(0)

# print(f"Clean accuracy on Train.pt = {100*correct/total:.2f} %")



# quick check on 100 samples
wrapper = ResNetWrapper("resnet18", 10)
wrapper.load_state_dict(torch.load("saved_models\\PGD_madry_model_optim.pt", map_location="cpu"))
wrapper.eval()

correct = 0
for i in range(100):
    _, img, lbl = raw_ds[i]
    img = transforms.ToTensor()(img.convert("RGB")).unsqueeze(0)
    pred = wrapper(img).argmax(1).item()
    correct += (pred == lbl)
print("Wrapper accuracy on first 100:", correct)   # expect >> 10

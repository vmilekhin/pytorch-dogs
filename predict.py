"""Инференс: определяем породу собаки по фотографии."""

import sys
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# ---------------------------------------------------------------------------
# 1. Загружаем сохранённую модель
# ---------------------------------------------------------------------------

checkpoint = torch.load("dog_breeds_v3.pth", map_location="cpu", weights_only=False)
classes = checkpoint["classes"]
print("classes:", classes)

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ---------------------------------------------------------------------------
# 2. Transform (тот же, что при обучении)
# ---------------------------------------------------------------------------

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

# ---------------------------------------------------------------------------
# 3. Загружаем картинку
# ---------------------------------------------------------------------------

image_path = sys.argv[1] if len(sys.argv) > 1 else "test_images/dog.jpg"
image = Image.open(image_path).convert("RGB")

x = transform(image)         # [3, 224, 224]
x = x.unsqueeze(0)           # [1, 3, 224, 224]

# ---------------------------------------------------------------------------
# 4. Inference
# ---------------------------------------------------------------------------

with torch.no_grad():
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)
    class_id = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, class_id].item()

print(f"\nФайл: {image_path}")

if classes[class_id] == "other":
    print(f"Порода: other (не из 5 известных)")
    print(f"Уверенность: {confidence:.2%}")
else:
    print(f"Порода: {classes[class_id]}")
    print(f"Уверенность: {confidence:.2%}")

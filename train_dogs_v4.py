"""v4: ResNet50 + weighted loss + mixup + layer4 fine-tuning. Финальная версия."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import v2

# ---------------------------------------------------------------------------
# 1. Device
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---------------------------------------------------------------------------
# 2. Классы: 10 пород собак + other (кошки)
# ---------------------------------------------------------------------------

dog_breeds = [
    "Beagle", "Pug", "Samoyed", "Shiba Inu", "Yorkshire Terrier",
    "Boxer", "Chihuahua", "Havanese", "Pomeranian", "Newfoundland",
]

cat_breeds = [
    "Abyssinian", "Bengal", "Birman", "Bombay",
    "British Shorthair", "Egyptian Mau", "Maine Coon",
    "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
]

classes = dog_breeds + ["other"]
print(f"Всего классов: {len(classes)}")

# ---------------------------------------------------------------------------
# 3. Датасет
# ---------------------------------------------------------------------------

train_all = datasets.OxfordIIITPet(root="./data", split="trainval",
                                   target_types="category", download=True)
test_all  = datasets.OxfordIIITPet(root="./data", split="test",
                                   target_types="category", download=True)
print("train_all:", len(train_all), "test_all:", len(test_all))

# ---------------------------------------------------------------------------
# 4. Маппинг
# ---------------------------------------------------------------------------

name_to_old = {name: i for i, name in enumerate(train_all.classes)}
for name in dog_breeds + cat_breeds:
    if name not in name_to_old:
        raise SystemExit(f"Порода '{name}' не найдена!")

dog_old_to_new = {name_to_old[n]: new for new, n in enumerate(dog_breeds)}
cat_old_set = {name_to_old[n] for n in cat_breeds}

def remap(old_y: int) -> int:
    if old_y in dog_old_to_new:
        return dog_old_to_new[old_y]
    if old_y in cat_old_set:
        return len(dog_breeds)  # other
    return -1

# ---------------------------------------------------------------------------
# 5. Трансформы
# ---------------------------------------------------------------------------

weights = ResNet50_Weights.DEFAULT

train_transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.RandomResizedCrop(224, scale=(0.7, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])
test_transform = weights.transforms()

# ---------------------------------------------------------------------------
# 6. Dataset
# ---------------------------------------------------------------------------

class MultiClasses(Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform
        self.items = []
        for i, (_, old_y) in enumerate(base):
            new_y = remap(old_y)
            if new_y >= 0:
                self.items.append((i, new_y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, new_y = self.items[idx]
        image, _ = self.base[i]
        return self.transform(image), new_y

train_ds = MultiClasses(train_all, train_transform)
test_ds  = MultiClasses(test_all,  test_transform)
print("train_ds:", len(train_ds), "test_ds:", len(test_ds))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4)

# ---------------------------------------------------------------------------
# 7. Модель: ResNet50, разморозить layer4 + fc
# ---------------------------------------------------------------------------

model = resnet50(weights=weights)

for name, p in model.named_parameters():
    if name.startswith("layer4") or name.startswith("fc"):
        p.requires_grad = True
    else:
        p.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Обучаемых: {trainable:,} / {total:,}")

# ---------------------------------------------------------------------------
# 8. Weighted loss для проблемных классов
# ---------------------------------------------------------------------------

class_weights = torch.ones(len(classes))
class_weights[classes.index("Chihuahua")]  = 1.5
class_weights[classes.index("Pomeranian")] = 1.3
class_weights = class_weights.to(device)
print("class_weights:", class_weights.cpu().numpy())

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ---------------------------------------------------------------------------
# 9. Optimizer + Scheduler
# ---------------------------------------------------------------------------

optimizer = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(),     "lr": 1e-3},
], weight_decay=1e-4)

EPOCHS = 20
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------------------------------------------------------------------------
# 10. Mixup
# ---------------------------------------------------------------------------

def mixup_batch(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam

# ---------------------------------------------------------------------------
# 11. Training loop
# ---------------------------------------------------------------------------

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        X_mix, y_a, y_b, lam = mixup_batch(X, y, alpha=0.2)

        optimizer.zero_grad()
        logits = model(X_mix)
        loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"  epoch {epoch+1:02d} batch {batch_idx:03d}/{len(train_loader)} "
                  f"loss={loss.item():.4f}")

    scheduler.step()
    avg = running_loss / len(train_loader)
    print(f"epoch={epoch+1:02d} avg_loss={avg:.4f}")

# ---------------------------------------------------------------------------
# 12. Evaluation
# ---------------------------------------------------------------------------

model.eval()
correct = total_pred = 0
n = len(classes)
confusion = torch.zeros(n, n, dtype=torch.int64)

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X).argmax(dim=1)
        correct += (pred == y).sum().item()
        total_pred += y.size(0)
        for t, p in zip(y.cpu(), pred.cpu()):
            confusion[t, p] += 1

accuracy = correct / total_pred
print(f"\naccuracy = {accuracy:.2%}")

print("\nConfusion matrix:")
print("          " + "  ".join(f"{c[:8]:>8}" for c in classes))
for i, c in enumerate(classes):
    row = "  ".join(f"{int(confusion[i, j]):>8}" for j in range(n))
    print(f"{c[:8]:>8}  {row}")

# ---------------------------------------------------------------------------
# 13. Сохранение
# ---------------------------------------------------------------------------

torch.save(
    {"model_state": model.state_dict(), "classes": classes},
    "dog_breeds_v4.pth",
)
print("\nsaved: dog_breeds_v4.pth")

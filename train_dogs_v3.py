"""Обучение v3: 10 пород + other, разморозка layer4, аугментация, scheduler."""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights
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
print(f"Породы: {dog_breeds}")
print("Класс 'other': кошки и всё остальное")

# ---------------------------------------------------------------------------
# 3. Датасет
# ---------------------------------------------------------------------------

print("\nЗагружаю trainval...")
train_all = datasets.OxfordIIITPet(
    root="./data", split="trainval", target_types="category", download=True,
)
print("Загружаю test...")
test_all = datasets.OxfordIIITPet(
    root="./data", split="test", target_types="category", download=True,
)

print("train_all:", len(train_all))
print("test_all:", len(test_all))

# ---------------------------------------------------------------------------
# 4. Маппинг old_class -> new_class
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
        return len(dog_breeds)  # индекс "other"
    return -1  # остальные — игнорируем

# ---------------------------------------------------------------------------
# 5. Трансформы — сильнее, чем в v2
# ---------------------------------------------------------------------------

weights = ResNet18_Weights.DEFAULT

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

print("train_ds:", len(train_ds))
print("test_ds: ", len(test_ds))

# ---------------------------------------------------------------------------
# 7. DataLoader
# ---------------------------------------------------------------------------

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=4)

# ---------------------------------------------------------------------------
# 8. Модель: размораживаем layer4 + fc
# ---------------------------------------------------------------------------

model = resnet18(weights=weights)

for name, p in model.named_parameters():
    if name.startswith("layer4") or name.startswith("fc"):
        p.requires_grad = True
    else:
        p.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Обучаемых параметров: {trainable:,} / {total:,}")

# ---------------------------------------------------------------------------
# 9. Optimizer с разными LR + Scheduler
# ---------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(),     "lr": 1e-3},
], weight_decay=1e-4)

EPOCHS = 15
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------------------------------------------------------------------------
# 10. Training loop
# ---------------------------------------------------------------------------

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"  epoch {epoch+1:02d} batch {batch_idx:03d}/{len(train_loader)} "
                  f"loss={loss.item():.4f}")

    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    lrs = [g["lr"] for g in optimizer.param_groups]
    print(f"epoch={epoch+1:02d} avg_loss={avg_loss:.4f} lrs={[f'{lr:.2e}' for lr in lrs]}")

# ---------------------------------------------------------------------------
# 11. Evaluation + confusion matrix
# ---------------------------------------------------------------------------

model.eval()
correct = total = 0
n = len(classes)
confusion = torch.zeros(n, n, dtype=torch.int64)

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        for t, p in zip(y.cpu(), pred.cpu()):
            confusion[t, p] += 1

accuracy = correct / total
print(f"\naccuracy = {accuracy:.2%}")

print("\nConfusion matrix (строки = true, столбцы = pred):")
header = "          " + "  ".join(f"{c[:8]:>8}" for c in classes)
print(header)
for i, c in enumerate(classes):
    row = "  ".join(f"{int(confusion[i, j]):>8}" for j in range(n))
    print(f"{c[:8]:>8}  {row}")

# ---------------------------------------------------------------------------
# 12. Сохранение
# ---------------------------------------------------------------------------

torch.save(
    {"model_state": model.state_dict(), "classes": classes},
    "dog_breeds_v3.pth",
)
print("\nsaved: dog_breeds_v3.pth")

"""Обучение 6-классовой модели: 5 пород собак + класс 'other' (кошки)."""

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
# 2. Классы: 5 пород собак + "other" (кошки)
# ---------------------------------------------------------------------------

dog_breeds = [
    "Beagle", "Pug", "Samoyed", "Shiba Inu", "Yorkshire Terrier",
]

cat_breeds = [
    "Abyssinian", "Bengal", "Birman", "Bombay",
    "British Shorthair", "Egyptian Mau", "Maine Coon",
    "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
]

# Итоговые классы: 0..4 — собаки, 5 — other
classes = dog_breeds + ["other"]

# ---------------------------------------------------------------------------
# 3. Датасет
# ---------------------------------------------------------------------------

print("Загружаю trainval...")
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

dog_old = {name_to_old[n] for n in dog_breeds}
cat_old = {name_to_old[n] for n in cat_breeds}

# Проверка
for name in dog_breeds + cat_breeds:
    if name not in name_to_old:
        raise SystemExit(f"Порода '{name}' не найдена в датасете!")

dog_old_to_new = {old: new for new, old in enumerate([name_to_old[n] for n in dog_breeds])}

def remap(old_y: int) -> int:
    """Собака -> 0..4, кошка -> 5."""
    if old_y in dog_old_to_new:
        return dog_old_to_new[old_y]
    if old_y in cat_old:
        return 5  # other
    return -1  # другие породы (например, оставшиеся собаки) — игнорируем

# ---------------------------------------------------------------------------
# 5. Трансформы
# ---------------------------------------------------------------------------

weights = ResNet18_Weights.DEFAULT

train_transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])
test_transform = weights.transforms()

# ---------------------------------------------------------------------------
# 6. Dataset
# ---------------------------------------------------------------------------

class SixClasses(Dataset):
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

train_ds = SixClasses(train_all, train_transform)
test_ds  = SixClasses(test_all,  test_transform)

print("train_ds:", len(train_ds))
print("test_ds:",  len(test_ds))

# ---------------------------------------------------------------------------
# 7. DataLoader
# ---------------------------------------------------------------------------

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=2)

# ---------------------------------------------------------------------------
# 8. Модель
# ---------------------------------------------------------------------------

model = resnet18(weights=weights)

# Замораживаем весь backbone, обучаем только fc
for p in model.parameters():
    p.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(classes))  # 6 выходов
model = model.to(device)

# ---------------------------------------------------------------------------
# 9. Optimizer + Scheduler
# ---------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# ---------------------------------------------------------------------------
# 10. Training loop
# ---------------------------------------------------------------------------

EPOCHS = 10
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

        if batch_idx % 5 == 0:
            print(f"  epoch {epoch+1} batch {batch_idx}/{len(train_loader)} "
                  f"loss={loss.item():.4f}")

    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    print(f"epoch={epoch+1:02d} avg_loss={avg_loss:.4f}")

# ---------------------------------------------------------------------------
# 11. Evaluation
# ---------------------------------------------------------------------------

model.eval()
correct = 0
total = 0

# confusion matrix для понимания: где модель ошибается
n_classes = len(classes)
confusion = torch.zeros(n_classes, n_classes, dtype=torch.int64)

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

        for t, p in zip(y.cpu(), pred.cpu()):
            confusion[t, p] += 1

accuracy = correct / total
print(f"\naccuracy = {accuracy:.2%}")

print("\nConfusion matrix (строки = true, столбцы = pred):")
print("       ", "  ".join(f"{c[:8]:>8}" for c in classes))
for i, c in enumerate(classes):
    row = "  ".join(f"{int(confusion[i,j]):>8}" for j in range(n_classes))
    print(f"{c[:8]:>8}  {row}")

# ---------------------------------------------------------------------------
# 12. Сохранение
# ---------------------------------------------------------------------------

torch.save(
    {
        "model_state": model.state_dict(),
        "classes": classes,
    },
    "dog_breeds_v2.pth",
)
print("\nsaved: dog_breeds_v2.pth")

"""Обучение классификатора 5 пород собак на базе ResNet18."""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image  # noqa: F401  (нужен внутри Dataset)

# ---------------------------------------------------------------------------
# 1. Устройство (CPU, потому что GPU у нас нет)
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---------------------------------------------------------------------------
# 2. Выбранные породы (правильные имена из датасета!)
# ---------------------------------------------------------------------------

selected = ["Beagle", "Pug", "Samoyed", "Shiba Inu", "Yorkshire Terrier"]

# ---------------------------------------------------------------------------
# 3. Загрузка полного Oxford-IIIT Pet
# ---------------------------------------------------------------------------

print("Загружаю trainval...")
train_all = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=True,
)

print("Загружаю test...")
test_all = datasets.OxfordIIITPet(
    root="./data",
    split="test",
    target_types="category",
    download=True,
)

print("train_all:", len(train_all))
print("test_all:", len(test_all))
print("classes:", train_all.classes)

# ---------------------------------------------------------------------------
# 4. Маппинг: имя класса -> старый индекс -> новый индекс (0..4)
# ---------------------------------------------------------------------------

name_to_old = {name: i for i, name in enumerate(train_all.classes)}

# На всякий случай проверим, что все 5 пород есть в датасете
for name in selected:
    if name not in name_to_old:
        raise SystemExit(f"Порода '{name}' не найдена в датасете!")

selected_old = [name_to_old[name] for name in selected]
old_to_new = {old: new for new, old in enumerate(selected_old)}

print("selected_old:", selected_old)

# ---------------------------------------------------------------------------
# 5. Transform (ResNet18 требует 224x224 и нормализацию ImageNet)
# ---------------------------------------------------------------------------

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

# ---------------------------------------------------------------------------
# 6. Dataset только для 5 пород
# ---------------------------------------------------------------------------

class FiveBreeds(Dataset):
    def __init__(self, base, transform, selected_old, old_to_new):
        self.base = base
        self.transform = transform
        self.selected_old = selected_old
        self.old_to_new = old_to_new
        self.indices = [
            i for i, (_, y) in enumerate(base) if y in selected_old
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, old_y = self.base[self.indices[i]]
        image = self.transform(image)
        new_y = self.old_to_new[old_y]
        return image, new_y

train_ds = FiveBreeds(train_all, transform, selected_old, old_to_new)
test_ds = FiveBreeds(test_all, transform, selected_old, old_to_new)

print("train_ds:", len(train_ds))
print("test_ds:", len(test_ds))

# ---------------------------------------------------------------------------
# 7. DataLoader
# ---------------------------------------------------------------------------

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

X, y = next(iter(train_loader))
print("X.shape:", X.shape)
print("y.shape:", y.shape)

# ---------------------------------------------------------------------------
# 8. Модель: pretrained ResNet18 + новый fc на 5 выходов
# ---------------------------------------------------------------------------

model = resnet18(weights=weights)

for p in model.parameters():
    p.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(selected))
model = model.to(device)

# ---------------------------------------------------------------------------
# 9. Loss и optimizer
# ---------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)

# ---------------------------------------------------------------------------
# 10. Training loop
# ---------------------------------------------------------------------------

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 5 == 0:
            print(f"  epoch {epoch+1} batch {batch_idx}/{len(train_loader)} loss={loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"epoch={epoch+1} avg_loss={avg_loss:.4f}")

# ---------------------------------------------------------------------------
# 11. Evaluation
# ---------------------------------------------------------------------------

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        pred = logits.argmax(dim=1)

        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total
print(f"accuracy = {accuracy:.2%}")

# ---------------------------------------------------------------------------
# 12. Сохранение модели
# ---------------------------------------------------------------------------

torch.save(
    {
        "model_state": model.state_dict(),
        "classes": selected,
    },
    "dog_breeds.pth",
)
print("saved: dog_breeds.pth")

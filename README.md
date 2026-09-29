# PyTorch Dog Breed Classifier

Классификатор 5 пород собак с поддержкой класса `other` (open-set recognition).

## Классы

- **0–4:** Beagle, Pug, Samoyed, Shiba Inu, Yorkshire Terrier
- **5:** `other` — все остальные животные (кошки, другие породы собак)

## Эволюция модели

| Версия | Классов | Backbone | Что нового | Accuracy |
|---|---|---|---|---|
| v1 | 5 | ResNet18 | Transfer learning, backbone заморожен | **99.60%** |
| v2 | 6 | ResNet18 | + класс `other` (open-set recognition) | **99.47%** |
| v3 | 11 | ResNet18 | + 5 пород, fine-tuning `layer4`, аугментация | **97.80%** |
| **v4** | **11** | **ResNet50** | **+ weighted loss, mixup** | **99.50%** |

### v4 — финальная версия
- **ResNet50** вместо ResNet18 (глубже, точнее).
- **Weighted loss:** Chihuahua ×1.5, Pomeranian ×1.3 — компенсация слабой точности в v3.
- **Mixup** α=0.2 — регуляризация, защита от переобучения.
- 20 эпох.

#### Улучшения по сравнению с v3:

| Порода | v3 | v4 | Δ |
|---|---|---|---|
| Chihuahua | 86% | **96%** | **+10%** |
| Pomeranian | 91% | **99%** | **+8%** |
| Shiba Inu | 97% | **100%** | +3% |
| Yorkshire Terrier | 96% | **100%** | +4% |
| Boxer | 96% | **99%** | +3% |
| **Общая accuracy** | 97.80% | **99.50%** | **+1.7%** |

## Примеры предсказаний

| Фото | Порода | Уверенность |
|---|---|---|
| beagle.jpg | Beagle | 93.35% |
| chihuahua_1.jpg | Chihuahua | 96.10% |
| Persian_1.jpg | other | 98.18% |

## Примеры предсказаний

| Фото | Порода | Уверенность |
|---|---|---|
| beagle.jpg | Beagle | 98.51% |
| chihuahua_1.jpg | Chihuahua | 97.32% |
| boxer_1.jpg | Boxer | 100.00% |
| Persian_1.jpg | other | 99.52% |
### v3 — финальная

- **10 пород собак:** Beagle, Pug, Samoyed, Shiba Inu, Yorkshire Terrier, Boxer, Chihuahua, Havanese, Pomeranian, Newfoundland
- **+ класс `other`:** кошки и всё остальное
- **Fine-tuning:** разморожен `layer4` backbone (не только `fc`)
- **Optimizer:** AdamW с разными learning rates для backbone и `fc`
- **Scheduler:** CosineAnnealingLR
- **Аугментация:** RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter

Примеры предсказаний:
## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow
```

## Обучение

```bash
python train_dogs_v2.py
```

- Датасет: Oxford-IIIT Pet (скачивается автоматически, ~800 МБ)
- Модель: ResNet18, pretrained, backbone заморожен
- Обучение: ~15 минут на CPU
- Результат: `dog_breeds_v2.pth`

## Предсказание

```bash
python predict.py путь/к/фото.jpg
```

Примеры:

```
# Своя порода:
python predict.py test_images/beagle.jpg
→ Порода: Beagle, уверенность: 95.84%

# Чужая (кошка):
python predict.py data/oxford-iiit-pet/images/Persian_1.jpg
→ Порода: other (не из 5 известных), уверенность: 99.41%
```

## Архитектура решения

1. **Transfer learning:** ResNet18 с весами ImageNet, обучаем только `fc`.
2. **Аугментация:** RandomResizedCrop, HorizontalFlip, ColorJitter — только для train.
3. **Optimizer:** AdamW с `lr=1e-3` (для `fc`).
4. **Scheduler:** CosineAnnealingLR.
5. **Open-set:** обучение с 6-м классом `other` вместо пороговой эвристики.

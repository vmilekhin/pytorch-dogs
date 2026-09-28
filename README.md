# PyTorch Dog Breed Classifier

Классификатор 5 пород собак с поддержкой класса `other` (open-set recognition).

## Классы

- **0–4:** Beagle, Pug, Samoyed, Shiba Inu, Yorkshire Terrier
- **5:** `other` — все остальные животные (кошки, другие породы собак)

## Версии модели

| Версия | Классов | Accuracy | Особенности |
|---|---|---|---|
| v1 | 5 | 99.60% | SGD, backbone заморожен, без other |
| v2 | 6 | 99.47% | + класс `other` (open-set recognition), AdamW |
| **v3** | **11** | **97.80%** | **10 пород + other, layer4 разморожен, аугментация, scheduler** |

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

# PyTorch Dog Breed Classifier

Классификатор 5 пород собак: Beagle, Pug, Samoyed, Shiba Inu, Yorkshire Terrier.

## Что использовано
- Датасет: Oxford-IIIT Pet (torchvision)
- Модель: ResNet18 (pretrained, transfer learning)
- Backbone заморожен, обучается только последний слой `fc`
- Accuracy на test set: **99.60%**

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow
```

## Запуск

### Обучение
```bash
python train_dogs.py
```
Обучает модель, сохраняет `dog_breeds.pth`.

### Предсказание
```bash
python predict.py путь/к/фото.jpg
```
Определяет породу на фото.

## Результаты

Пример:
```
Файл: /mnt/d/Пользователи/Рабочий стол/dog.jpg
Порода: Beagle
Уверенность: 100.00%

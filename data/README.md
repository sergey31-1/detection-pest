# Данные для проекта Crop Pest Detector

## Датасет: AgroPest-12

Датасет содержит 12 классов вредителей сельскохозяйственных культур с разметкой в формате YOLO.

- Источник: [Kaggle – Crop Pests Dataset](https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset)
- Классы: ants, bees, beetles, caterpillars, earthworms, earwigs, grasshoppers, moths, slugs, snails, wasps, weevils
- Формат аннотаций: YOLO (class x_center y_center width height)

## Скачивание и подготовка

1. Перейдите на страницу датасета: https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset
2. Скачайте архив `detection-pest.zip`.
3. Распакуйте архив в папку `data/` этого проекта.
4. После распаковки структура должна выглядеть так:
data/
images/
train/
val/
test/
labels/
train/
val/
test/
agropest.yaml (уже есть в репозитории)
README.md (этот файл)
5. Убедитесь, что папки `images` и `labels` находятся на одном уровне с `agropest.yaml`.

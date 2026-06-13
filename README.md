#Crop Pest Detector — Обнаружение вредителей полей с помощью YOLOv8

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-blue)](https://github.com/ultralytics/ultralytics)

**Crop Pest Detector** — это проект по обучению нейросети для автоматического обнаружения и классификации вредителей сельскохозяйственных культур на фотографиях с полей.

 **Цель** — помочь фермерам и агрономам быстро выявлять очаги вредителей на ранних стадиях для снижения потерь урожая и оптимизации применения пестицидов.

## Используемый подход

- **Архитектура:** Детекция объектов на основе **YOLOv8** (модель `yolov8n` или `yolov8s`).
- **Обучение:** Предобученная на ImageNet модель дообучается на датасете AgroPest-12 для точного выделения вредителей на изображениях с полей.
- **Аугментация данных:** Для повышения устойчивости модели применяются случайные повороты, изменения яркости и масштабирования.

##  Датасет

В проекте используется датасет **[AgroPest-12](https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset)** с платформы Kaggle.

Основные характеристики:
- **12 классов** вредителей: Муравьи, Пчёлы, Жуки, Гусеницы, Дождевые черви, Уховёртки, Кузнечики, Мотыльки, Слизни, Улитки, Осы, Долгоносики[reference:0].
- **Разметка в формате YOLO:** Каждое изображение сопровождается `.txt`-файлом с аннотациями (`class x_center y_center width height`)[reference:1].
- Прямая совместимость с обучением моделей семейства YOLO.

>  Сам датасет не хранится в репозитории. Инструкция по его загрузке приведена ниже.

## Установка и запуск
```bash
# Клонируем репозиторий
git clone https://github.com/sergey31-1/detection-pest.git
cd crop-pest-detector

# Создаём виртуальное окружение
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Устанавливаем зависимости
pip install -r requirements.txt

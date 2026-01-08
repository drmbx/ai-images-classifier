# AI Images Classifier

Сервис для распознавания ИИ-изображений.

## Описание

Проект представляет собой модель глубокого обучения для классификации изображений на две категории:
- **AI-generated** - изображения, сгенерированные искусственным интеллектом
- **Real** - реальные фотографии

Модель использует предобученные архитектуры (EfficientNet-B0 или ResNet-50) в качестве backbone и дообучается на ваших данных.

## Ключевые технологии

- **PyTorch Lightning** - удобный фреймворк для обучения моделей
- **Hydra** - управление конфигурацией экспериментов
- **TensorBoard** - визуализация метрик и логирование
- **Ruff** - быстрый линтер и форматтер для Python
- **UV** - быстрый менеджер пакетов для Python

## Структура проекта

```
ai-images-classifier/
├── model.py              # Архитектура модели
├── lightning_module.py   # PyTorch Lightning модуль для обучения
├── data_module.py        # DataModule для загрузки данных
├── train.py             # Скрипт для обучения (с Hydra)
├── predict.py           # Скрипт для предсказаний
├── conf/                # Конфигурация Hydra
│   ├── config.yaml      # Основной конфиг
│   ├── model/           # Конфигурации моделей
│   ├── data/            # Конфигурации данных
│   ├── training/        # Конфигурации обучения
│   ├── lightning/       # Конфигурации Lightning
│   ├── logger/          # Конфигурации логгеров (TensorBoard)
│   └── callbacks/       # Конфигурации callbacks
├── pyproject.toml      # Конфигурация проекта (uv, ruff)
├── requirements.txt     # Зависимости (legacy)
├── Makefile            # Команды для разработки
├── scripts/             # Вспомогательные скрипты
│   ├── lint.sh         # Скрипт проверки кода
│   └── format.sh       # Скрипт форматирования
└── README.md           # Документация
```

## Установка

### Вариант 1: Использование UV (рекомендуется)

1. Установите UV (если еще не установлен):
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Установите зависимости:
```bash
uv pip install -e .
```

3. Для разработки (с ruff):
```bash
uv pip install -e ".[dev]"
```

### Вариант 2: Использование pip

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Установите ruff отдельно (опционально):
```bash
pip install ruff
```

### Подготовка данных

Подготовьте данные в следующей структуре:
```
data/
├── train/
│   ├── ai/          # ИИ-изображения для обучения
│   └── real/        # Реальные изображения для обучения
├── val/             # (опционально)
│   ├── ai/
│   └── real/
└── test/            # (опционально)
    ├── ai/
    └── real/
```

Если папки `val` и `test` отсутствуют, данные будут автоматически разделены на train/val/test.

## Использование

### Обучение модели с Hydra

Базовый запуск (использует конфигурацию по умолчанию):
```bash
python train.py
```

С переопределением параметров через командную строку:
```bash
python train.py \
    data.batch_size=64 \
    training.learning_rate=2e-4 \
    training.max_epochs=100 \
    model.backbone_name=resnet50
```

Выбор другой модели:
```bash
python train.py model=resnet50
```

### Примеры использования Hydra

Переопределение нескольких параметров:
```bash
python train.py \
    data.batch_size=64 \
    data.image_size=256 \
    training.learning_rate=1e-3 \
    training.max_epochs=100 \
    lightning.devices=2 \
    experiment_name=my_experiment
```

Использование разных конфигураций:
```bash
# EfficientNet-B0 с большим batch size
python train.py model=efficientnet_b0 data.batch_size=128

# ResNet-50 с меньшим learning rate
python train.py model=resnet50 training.learning_rate=5e-5
```

### Мульти-запуск (Hyperparameter Sweep)

Hydra позволяет легко запускать множественные эксперименты:
```bash
python train.py -m \
    training.learning_rate=1e-4,5e-5,1e-5 \
    data.batch_size=32,64,128 \
    model.dropout=0.3,0.5,0.7
```

### Предсказание на новых изображениях

```bash
python predict.py \
    --model_path checkpoints/best-model-epoch=XX-val_loss=X.XX.ckpt \
    --image_path path/to/image.jpg \
    --device cuda
```

## Конфигурация

Все параметры настраиваются через файлы в директории `conf/`:

- `conf/config.yaml` - основной конфигурационный файл
- `conf/model/` - конфигурации моделей (efficientnet_b0, resnet50)
- `conf/data/` - параметры данных
- `conf/training/` - параметры обучения
- `conf/lightning/` - параметры PyTorch Lightning
- `conf/logger/` - настройки логгеров (TensorBoard)
- `conf/callbacks/` - настройки callbacks

### Пример создания собственной конфигурации

Создайте `conf/model/custom.yaml`:
```yaml
backbone_name: efficientnet_b0
num_classes: 2
dropout: 0.3
pretrained: true
```

Используйте её:
```bash
python train.py model=custom
```

## Параметры модели

- **backbone**: `efficientnet_b0` или `resnet50` (по умолчанию: `efficientnet_b0`)
- **num_classes**: Количество классов (по умолчанию: 2)
- **dropout**: Вероятность dropout (по умолчанию: 0.5)
- **pretrained**: Использовать предобученные веса ImageNet (по умолчанию: True)

## Особенности

- ✅ PyTorch Lightning для удобного процесса обучения
- ✅ **Hydra** для гибкого управления конфигурацией
- ✅ **TensorBoard** для визуализации метрик
- ✅ Автоматическое разделение данных на train/val/test
- ✅ Data augmentation для улучшения обобщения
- ✅ Callbacks: ModelCheckpoint, EarlyStopping, LearningRateMonitor
- ✅ Поддержка GPU/CPU
- ✅ Mixed precision training (16-bit)
- ✅ Метрики: Accuracy, Precision, Recall, F1-Score
- ✅ Hyperparameter sweeps через Hydra
- ✅ **Ruff** для линтинга и форматирования кода
- ✅ **UV** для быстрого управления зависимостями

## Результаты обучения

Во время обучения создаются:
- **checkpoints/**: Сохраненные модели (лучшие 3 + последняя)
- **outputs/**: Результаты Hydra (конфигурации и логи)
- **logs/**: TensorBoard логи для визуализации

### Просмотр результатов

Запустите TensorBoard для просмотра метрик:
```bash
tensorboard --logdir logs
```

## Полезные команды Hydra

```bash
# Показать конфигурацию без запуска
python train.py --cfg job

# Сохранить конфигурацию в файл
python train.py --cfg job --config-path=conf --config-name=config

# Запуск с разными seed
python train.py seed=42,123,456 -m
```

## Разработка

### Использование Ruff

Ruff используется для линтинга и форматирования кода.

**Проверка кода:**
```bash
# С помощью Makefile
make lint

# Или напрямую
ruff check .

# Проверка форматирования
ruff format --check .
```

**Форматирование кода:**
```bash
# С помощью Makefile
make format

# Или напрямую
ruff format .
```

**Автоматическое исправление:**
```bash
# С помощью Makefile
make fix

# Или напрямую
ruff check --fix .
ruff format .
```

**Использование скриптов:**
```bash
# Linux/Mac
./scripts/lint.sh
./scripts/format.sh

# Windows (PowerShell)
bash scripts/lint.sh
bash scripts/format.sh
```

### Использование UV

UV - быстрый менеджер пакетов для Python.

**Установка зависимостей:**
```bash
# Установка проекта
uv pip install -e .

# Установка с dev зависимостями
uv pip install -e ".[dev]"

# Синхронизация зависимостей
uv pip sync
```

**Создание виртуального окружения:**
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Makefile команды

Для удобства доступны команды через Makefile:

```bash
make help        # Показать справку
make install      # Установить зависимости через uv
make install-dev  # Установить зависимости для разработки
make lint         # Проверить код
make format       # Форматировать код
make check        # Проверить код (lint + format check)
make fix          # Автоматически исправить проблемы
make clean        # Очистить кэш и временные файлы
```

## Лицензия

MIT

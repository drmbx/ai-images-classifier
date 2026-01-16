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
- **MLflow** - отслеживание экспериментов и логирование метрик
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

### Вариант 1: Использование UV (локальная установка)

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

#### Ручная подготовка

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

#### Загрузка данных из Hugging Face

Для загрузки данных из Hugging Face используйте:
```bash
python -m src.ai_images_classifier.utils.download_data
```

## Использование

### MLflow

Проект использует MLflow для отслеживания экспериментов. По умолчанию используется MLflow logger.

#### Настройка MLflow

MLflow настраивается через конфигурацию `conf/logger/mlflow.yaml`:
```yaml
_target_: pytorch_lightning.loggers.MLFlowLogger
experiment_name: ${experiment_name}
tracking_uri: http://127.0.0.1:8080
log_model: false
```

#### Запуск MLflow сервера

Перед обучением убедитесь, что MLflow сервер запущен:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 8080
```

Или используйте удаленный tracking server (укажите его адрес в конфиге).

#### Что логируется в MLflow

- **Гиперпараметры**: все параметры модели, обучения и данных
- **Метрики**: train_loss, val_loss, train_acc, val_acc, val_f1, val_precision, val_recall
- **Git commit ID**: автоматически определяется и логируется
- **Графики**: графики loss, accuracy и F1-score сохраняются в `plots/` и логируются в MLflow

#### Переключение между логгерами

Используйте Hydra для выбора логгера:
```bash
# MLflow (по умолчанию)
python train.py logger=mlflow

# TensorBoard
python train.py logger=tensorboard
```

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

## Infer

### Использование обученной модели

После обучения модели вы можете использовать её для предсказаний:

```bash
python predict.py \
    --model_path checkpoints/best-model-epoch=XX-val_loss=X.XX.ckpt \
    --image_path path/to/image.jpg \
    --device cuda
```

### Использование Triton Inference Server

Проект поддерживает развертывание через Triton Inference Server.

#### Подготовка модели для Triton

1. Экспортируйте модель в ONNX:
```bash
python export_to_onnx.py \
    --model_path checkpoints/best-model.ckpt \
    --copy_to_triton
```

2. Или подготовьте конфигурацию Triton вручную:
```bash
python prepare_triton.py --model_path checkpoints/best-model.ckpt
```

#### Запуск Triton Inference Server

```bash
# Используйте готовый скрипт
bash start_triton.sh
```

#### Использование Triton клиента

```bash
python triton_predict.py \
    --image_path path/to/image.jpg \
    --model_name ai_classifier \
    --url localhost:8000
```

## Production preparation

### Экспорт в ONNX

Экспорт модели в ONNX формат для использования в продакшене:

```bash
python export_to_onnx.py \
    --model_path checkpoints/best-model-epoch=XX-val_loss=X.XX.ckpt \
    --output_dir models/onnx \
    --image_size 224
```

Опции:
- `--model_path`: Путь к checkpoint модели
- `--output_dir`: Директория для сохранения ONNX модели
- `--image_size`: Размер входного изображения (по умолчанию 224)
- `--opset`: Версия ONNX opset (по умолчанию 18)
- `--copy_to_triton`: Скопировать модель в Triton директорию

### Использование Triton Inference Server

Проект поддерживает развертывание через Triton Inference Server. Подробности в разделе [Infer](#infer).

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
- ✅ **MLflow** для отслеживания экспериментов и логирования метрик
- ✅ **TensorBoard** для визуализации метрик
- ✅ Автоматическое разделение данных на train/val/test
- ✅ Data augmentation для улучшения обобщения
- ✅ Callbacks: ModelCheckpoint, EarlyStopping, LearningRateMonitor
- ✅ Поддержка GPU/CPU
- ✅ Mixed precision training (16-bit)
- ✅ Метрики: Accuracy, Precision, Recall, F1-Score
- ✅ Hyperparameter sweeps через Hydra
- ✅ Автоматическое логирование git commit ID
- ✅ Сохранение графиков метрик в `plots/`
- ✅ **Ruff** для линтинга и форматирования кода
- ✅ **UV** для быстрого управления зависимостями

## Результаты обучения

Во время обучения создаются:
- **checkpoints/**: Сохраненные модели (лучшие 3 + последняя)
- **outputs/**: Результаты Hydra (конфигурации и логи)
- **logs/**: TensorBoard логи для визуализации
- **plots/**: Графики метрик (loss, accuracy, F1-score)
- **MLflow**: Эксперименты логируются в MLflow (по умолчанию http://127.0.0.1:8080)

### Просмотр результатов

#### MLflow UI

Запустите MLflow сервер для просмотра экспериментов:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 8080
```

Или если сервер уже запущен:
```bash
# Откройте в браузере
http://127.0.0.1:8080
```

В MLflow вы можете:
- Просматривать метрики и гиперпараметры
- Сравнивать эксперименты
- Просматривать графики метрик
- Видеть git commit id для каждого эксперимента

#### TensorBoard

Запустите TensorBoard для просмотра метрик:
```bash
tensorboard --logdir logs
```

#### Графики

Графики метрик сохраняются в папке `plots/`:
- `{experiment_name}_loss.png` - график loss
- `{experiment_name}_accuracy.png` - график accuracy
- `{experiment_name}_f1_score.png` - график F1, Precision, Recall

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
src .venv/bin/activate  # Linux/Mac
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

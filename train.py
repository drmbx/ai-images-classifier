"""
Скрипт для обучения модели классификации ИИ-изображений
С использованием Hydra для конфигурации
"""

import os
import sys
import random
from pathlib import Path

# Отключаем triton на Windows (не поддерживается)
if os.name == "nt":  # Windows
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import hydra
import lightning as L
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import MLFlowLogger

from src.ai_images_classifier.modules.data_module import AIImageDataModule
from src.ai_images_classifier.modules.lightning_module import AIImageClassifierModule
from src.ai_images_classifier.utils.metrics_collector import MetricsCollectorCallback
from src.ai_images_classifier.utils.mlflow_utils import log_hyperparameters_to_mlflow


def set_seed(seed: int):
    """Установка seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """
    Основная функция обучения

    Args:
        cfg: Конфигурация из Hydra
    """
    # Установка seed
    if hasattr(cfg, "seed"):
        set_seed(cfg.seed)
        print(f"Seed установлен: {cfg.seed}")

    # Вывод конфигурации
    print("Конфигурация эксперимента:")
    print(OmegaConf.to_yaml(cfg))

    # Информация о заморозке
    if hasattr(cfg.model, "freeze_backbone") and cfg.model.freeze_backbone:
        print("🧊 Backbone БУДЕТ ЗАМОРОЖЕН на всё время обучения")
        print("   (обучается только классификатор)")
    else:
        print("🔥 Backbone НЕ заморожен")
        print("   (обучается вся модель)")

    print("=" * 60 + "\n")

    # Инициализация DataModule
    data_module = AIImageDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
    )

    # Инициализация модели
    print("🧠 Инициализация модели...")
    model = AIImageClassifierModule(
        backbone_name=cfg.model.backbone_name,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.get("freeze_backbone", False),
    )

    # Создание callbacks из конфигурации
    callbacks = []

    # ModelCheckpoint
    if "checkpoint" in cfg.callbacks:
        checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint)
        callbacks.append(checkpoint_callback)

    # EarlyStopping
    if "early_stopping" in cfg.callbacks:
        early_stopping = hydra.utils.instantiate(cfg.callbacks.early_stopping)
        callbacks.append(early_stopping)

    # LearningRateMonitor
    if "lr_monitor" in cfg.callbacks:
        lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)
        callbacks.append(lr_monitor)

    # MetricsCollector для сбора метрик и создания графиков
    metrics_collector = MetricsCollectorCallback(plots_dir=Path("plots"))
    callbacks.append(metrics_collector)

    # Создание logger из конфигурации
    logger = None
    mlflow_logger = None
    if "logger" in cfg and cfg.logger is not None:
        logger_config = cfg.logger.copy()
        # Заменяем переменные в конфигурации logger
        if "name" in logger_config:
            logger_config.name = cfg.experiment_name
        logger = hydra.utils.instantiate(logger_config)

        # Проверяем, является ли logger MLflowLogger
        if isinstance(logger, MLFlowLogger):
            mlflow_logger = logger
            # Инициализируем MLflow run
            tracking_uri = cfg.logger.get("tracking_uri") or cfg.get("mlflow", {}).get("tracking_uri", "http://127.0.0.1:8080")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(cfg.logger.experiment_name)
            try:
                mlflow.start_run()
                # Логируем гиперпараметры и git commit id
                log_hyperparameters_to_mlflow(cfg)
            except Exception as e:
                print(f"⚠️  Не удалось подключиться к MLflow серверу: {e}")
                print("   Обучение продолжится без логирования в MLflow")
                mlflow_logger = None

    # Trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.lightning.accelerator,
        devices=cfg.lightning.devices,
        precision=cfg.lightning.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.lightning.log_every_n_steps,
        val_check_interval=cfg.lightning.val_check_interval,
        enable_progress_bar=cfg.lightning.enable_progress_bar,
        enable_model_summary=cfg.lightning.enable_model_summary,
    )

    # Обучение
    print("\n" + "=" * 50)
    print("Начало обучения")
    print("=" * 50 + "\n")
    trainer.fit(model, data_module)

    # Тестирование
    print("\n" + "=" * 50)
    print("Запуск тестирования")
    print("=" * 50 + "\n")
    trainer.test(model, data_module)

    # Вывод информации о лучшей модели
    if "checkpoint" in cfg.callbacks and hasattr(
        checkpoint_callback, "best_model_path"
    ):
        print(f"\nЛучшая модель сохранена в: {checkpoint_callback.best_model_path}")

    # Логируем финальные метрики в MLflow
    if mlflow_logger is not None:
        try:
            # Логируем финальные метрики из callback_metrics
            if hasattr(trainer, "callback_metrics"):
                for metric_name, metric_value in trainer.callback_metrics.items():
                    if isinstance(metric_value, torch.Tensor):
                        mlflow.log_metric(metric_name, metric_value.item())
                    elif isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)

            print("✅ Метрики залогированы в MLflow")
        except Exception as e:
            print(f"⚠️  Ошибка при логировании в MLflow: {e}")

    # Закрываем MLflow run
    if mlflow_logger is not None:
        mlflow.end_run()


if __name__ == "__main__":
    main()

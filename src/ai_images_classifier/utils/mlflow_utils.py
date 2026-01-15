"""
Утилиты для работы с MLflow
"""

import subprocess
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
from omegaconf import DictConfig


def get_git_commit_id() -> Optional[str]:
    """
    Получает текущий git commit id

    Returns:
        Git commit hash или None если не удалось получить
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def log_hyperparameters_to_mlflow(cfg: DictConfig) -> None:
    """
    Логирует гиперпараметры в MLflow

    Args:
        cfg: Конфигурация Hydra
    """
    # Преобразуем конфигурацию в плоский словарь для MLflow
    hyperparameters = {}

    # Основные параметры модели
    if hasattr(cfg, "model"):
        hyperparameters["model.backbone_name"] = cfg.model.backbone_name
        hyperparameters["model.num_classes"] = cfg.model.num_classes
        hyperparameters["model.dropout"] = cfg.model.dropout
        hyperparameters["model.pretrained"] = cfg.model.pretrained
        hyperparameters["model.freeze_backbone"] = cfg.model.get("freeze_backbone", False)

    # Параметры обучения
    if hasattr(cfg, "training"):
        hyperparameters["training.learning_rate"] = cfg.training.learning_rate
        hyperparameters["training.weight_decay"] = cfg.training.weight_decay
        hyperparameters["training.max_epochs"] = cfg.training.max_epochs

    # Параметры данных
    if hasattr(cfg, "data"):
        hyperparameters["data.batch_size"] = cfg.data.batch_size
        hyperparameters["data.image_size"] = cfg.data.image_size
        hyperparameters["data.train_split"] = cfg.data.train_split
        hyperparameters["data.val_split"] = cfg.data.val_split
        hyperparameters["data.test_split"] = cfg.data.test_split

    # Lightning параметры
    if hasattr(cfg, "lightning"):
        hyperparameters["lightning.accelerator"] = cfg.lightning.accelerator
        hyperparameters["lightning.devices"] = cfg.lightning.devices
        hyperparameters["lightning.precision"] = cfg.lightning.precision

    # Seed
    if hasattr(cfg, "seed"):
        hyperparameters["seed"] = cfg.seed

    # Логируем гиперпараметры
    mlflow.log_params(hyperparameters)

    # Логируем git commit id
    commit_id = get_git_commit_id()
    if commit_id:
        mlflow.log_param("git_commit_id", commit_id)
        print(f"📝 Git commit ID: {commit_id}")


def save_metrics_plots(
    metrics_history: dict,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """
    Сохраняет графики метрик в папку plots/

    Args:
        metrics_history: Словарь с историей метрик {metric_name: [values]}
        output_dir: Директория для сохранения графиков
        experiment_name: Название эксперимента
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Создаем графики для основных метрик
    metrics_to_plot = {
        "loss": ["train_loss", "val_loss"],
        "accuracy": ["train_acc", "val_acc"],
        "f1_score": ["val_f1"],
    }

    for plot_name, metric_names in metrics_to_plot.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        for metric_name in metric_names:
            if metric_name in metrics_history:
                values = metrics_history[metric_name]
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, label=metric_name, marker="o")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title(f"{plot_name.capitalize()} over epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = plots_dir / f"{experiment_name}_{plot_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Логируем график в MLflow
        mlflow.log_artifact(str(plot_path), "plots")

        print(f"📊 График сохранен: {plot_path}")











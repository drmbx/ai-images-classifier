"""
Callback для сбора метрик во время обучения
"""

from collections import defaultdict
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import mlflow


class MetricsCollectorCallback(L.Callback):
    """
    Callback для сбора метрик и создания графиков
    """

    def __init__(self, plots_dir: Path = Path("plots")):
        """
        Args:
            plots_dir: Директория для сохранения графиков
        """
        super().__init__()
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = defaultdict(list)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Собираем метрики после каждой эпохи обучения"""
        metrics = trainer.callback_metrics

        # Собираем метрики обучения
        if "train_loss" in metrics:
            self.metrics_history["train_loss"].append(
                metrics["train_loss"].item() if hasattr(metrics["train_loss"], "item") else float(metrics["train_loss"])
            )
        if "train_acc" in metrics:
            self.metrics_history["train_acc"].append(
                metrics["train_acc"].item() if hasattr(metrics["train_acc"], "item") else float(metrics["train_acc"])
            )

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Собираем метрики после каждой эпохи валидации"""
        metrics = trainer.callback_metrics

        # Собираем метрики валидации
        if "val_loss" in metrics:
            self.metrics_history["val_loss"].append(
                metrics["val_loss"].item() if hasattr(metrics["val_loss"], "item") else float(metrics["val_loss"])
            )
        if "val_acc" in metrics:
            self.metrics_history["val_acc"].append(
                metrics["val_acc"].item() if hasattr(metrics["val_acc"], "item") else float(metrics["val_acc"])
            )
        if "val_f1" in metrics:
            self.metrics_history["val_f1"].append(
                metrics["val_f1"].item() if hasattr(metrics["val_f1"], "item") else float(metrics["val_f1"])
            )
        if "val_precision" in metrics:
            self.metrics_history["val_precision"].append(
                metrics["val_precision"].item()
                if hasattr(metrics["val_precision"], "item")
                else float(metrics["val_precision"])
            )
        if "val_recall" in metrics:
            self.metrics_history["val_recall"].append(
                metrics["val_recall"].item()
                if hasattr(metrics["val_recall"], "item")
                else float(metrics["val_recall"])
            )

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Создаем графики после окончания обучения"""
        if not self.metrics_history:
            return

        experiment_name = getattr(trainer.logger, "experiment_name", "experiment") if trainer.logger else "experiment"

        # Создаем графики
        self._create_loss_plot(experiment_name)
        self._create_accuracy_plot(experiment_name)
        self._create_f1_plot(experiment_name)

    def _create_loss_plot(self, experiment_name: str) -> None:
        """Создает график loss"""
        if "train_loss" not in self.metrics_history and "val_loss" not in self.metrics_history:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if "train_loss" in self.metrics_history:
            epochs = range(1, len(self.metrics_history["train_loss"]) + 1)
            ax.plot(epochs, self.metrics_history["train_loss"], label="train_loss", marker="o")

        if "val_loss" in self.metrics_history:
            epochs = range(1, len(self.metrics_history["val_loss"]) + 1)
            ax.plot(epochs, self.metrics_history["val_loss"], label="val_loss", marker="s")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss over epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = self.plots_dir / f"{experiment_name}_loss.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Логируем в MLflow
        try:
            mlflow.log_artifact(str(plot_path), "plots")
        except Exception:
            pass  # MLflow может быть не инициализирован

        print(f"📊 График loss сохранен: {plot_path}")

    def _create_accuracy_plot(self, experiment_name: str) -> None:
        """Создает график accuracy"""
        if "train_acc" not in self.metrics_history and "val_acc" not in self.metrics_history:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if "train_acc" in self.metrics_history:
            epochs = range(1, len(self.metrics_history["train_acc"]) + 1)
            ax.plot(epochs, self.metrics_history["train_acc"], label="train_acc", marker="o")

        if "val_acc" in self.metrics_history:
            epochs = range(1, len(self.metrics_history["val_acc"]) + 1)
            ax.plot(epochs, self.metrics_history["val_acc"], label="val_acc", marker="s")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy over epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = self.plots_dir / f"{experiment_name}_accuracy.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Логируем в MLflow
        try:
            mlflow.log_artifact(str(plot_path), "plots")
        except Exception:
            pass

        print(f"📊 График accuracy сохранен: {plot_path}")

    def _create_f1_plot(self, experiment_name: str) -> None:
        """Создает график F1 score"""
        if "val_f1" not in self.metrics_history:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(self.metrics_history["val_f1"]) + 1)
        ax.plot(epochs, self.metrics_history["val_f1"], label="val_f1", marker="o", color="green")

        if "val_precision" in self.metrics_history:
            epochs = range(1, len(self.metrics_history["val_precision"]) + 1)
            ax.plot(epochs, self.metrics_history["val_precision"], label="val_precision", marker="s", color="blue")

        if "val_recall" in self.metrics_history:
            epochs = range(1, len(self.metrics_history["val_recall"]) + 1)
            ax.plot(epochs, self.metrics_history["val_recall"], label="val_recall", marker="^", color="red")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("F1, Precision, Recall over epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = self.plots_dir / f"{experiment_name}_f1_score.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Логируем в MLflow
        try:
            mlflow.log_artifact(str(plot_path), "plots")
        except Exception:
            pass

        print(f"📊 График F1 score сохранен: {plot_path}")











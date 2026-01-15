"""
PyTorch Lightning модуль для обучения модели классификации ИИ-изображений
"""

import torch
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall, F1Score
import lightning as L
from ..model.model import AIImageClassifier


class AIImageClassifierModule(L.LightningModule):
    """
    Lightning модуль для обучения модели классификации ИИ-изображений
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 2,
        dropout: float = 0.5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            backbone_name: Название backbone модели
            num_classes: Количество классов
            dropout: Dropout вероятность
            learning_rate: Скорость обучения
            weight_decay: Weight decay для оптимизатора
            pretrained: Использовать ли предобученные веса
            freeze_backbone: Замораживать ли веса backbone (навсегда)
        """
        super().__init__()
        self.save_hyperparameters()

        # Инициализация модели
        self.model = AIImageClassifier(
            backbone_name=backbone_name,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

        # Loss функция: используем CrossEntropyLoss для всех случаев
        self.criterion = nn.CrossEntropyLoss()

        # Метрики (используем режим multiclass даже для 2 классов,
        # чтобы работать с целочисленными метками 0/1)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.val_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        """Forward pass"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Шаг обучения"""
        images, labels = batch

        # Forward pass
        logits = self(images)

        # Вычисление loss и предсказаний (общий случай multiclass, в том числе 2 класса)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        # Обновление метрик
        acc = self.train_accuracy(preds, labels)

        # Логирование
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        # Логируем статус заморозки только на первом шаге
        if self.current_epoch == 0 and batch_idx == 0:
            backbone_status = 0.0 if self.model.freeze_backbone else 1.0
            # Используем self.log() для универсальной совместимости с любым логгером
            self.log("backbone_status", backbone_status, on_step=False, on_epoch=False)
            if self.model.freeze_backbone:
                print("🧊 Backbone заморожен - обучается только классификатор")
            else:
                print("🔥 Backbone разморожен - обучается вся модель")

        return loss

    def validation_step(self, batch, batch_idx):
        """Шаг валидации"""
        images, labels = batch

        # Forward pass
        logits = self(images)

        # Вычисление loss и предсказаний
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        # Обновление метрик
        self.val_accuracy(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)

        # Логирование
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Шаг тестирования"""
        images, labels = batch

        # Forward pass
        logits = self(images)

        # Вычисление loss и предсказаний
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        # Обновление метрик
        self.test_accuracy(preds, labels)

        # Логирование
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Настройка оптимизатора - обучаем только размороженные параметры"""

        # Собираем только те параметры, которые требуют градиентов
        trainable_params = []
        frozen_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
            else:
                frozen_params.append(param)

        # Логируем информацию
        print("📊 Параметры модели:")
        print(
            f"   Всего параметров: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        print(f"   Обучаемых параметров: {sum(p.numel() for p in trainable_params):,}")
        print(f"   Замороженных параметров: {sum(p.numel() for p in frozen_params):,}")
        print(
            f"   Обучается: {sum(p.numel() for p in trainable_params) / sum(p.numel() for p in self.model.parameters()) * 100:.1f}% модели"
        )

        # Создаем оптимизатор только для обучаемых параметров
        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

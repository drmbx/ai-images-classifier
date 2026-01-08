"""
PyTorch Lightning модуль для обучения модели классификации ИИ-изображений
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score
import pytorch_lightning as pl
from model import AIImageClassifier


class AIImageClassifierModule(pl.LightningModule):
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
        pretrained: bool = True
    ):
        """
        Args:
            backbone_name: Название backbone модели
            num_classes: Количество классов
            dropout: Dropout вероятность
            learning_rate: Скорость обучения
            weight_decay: Weight decay для оптимизатора
            pretrained: Использовать ли предобученные веса
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Инициализация модели
        self.model = AIImageClassifier(
            backbone_name=backbone_name,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained
        )
        
        # Loss функция (для бинарной классификации используем BCEWithLogitsLoss)
        if num_classes == 2:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Метрики
        self.train_accuracy = Accuracy(task="binary" if num_classes == 2 else "multiclass", 
                                       num_classes=num_classes)
        self.val_accuracy = Accuracy(task="binary" if num_classes == 2 else "multiclass", 
                                     num_classes=num_classes)
        self.test_accuracy = Accuracy(task="binary" if num_classes == 2 else "multiclass", 
                                      num_classes=num_classes)
        
        self.val_precision = Precision(task="binary" if num_classes == 2 else "multiclass", 
                                       num_classes=num_classes)
        self.val_recall = Recall(task="binary" if num_classes == 2 else "multiclass", 
                                 num_classes=num_classes)
        self.val_f1 = F1Score(task="binary" if num_classes == 2 else "multiclass", 
                             num_classes=num_classes)
        
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
        
        # Вычисление loss
        if self.hparams.num_classes == 2:
            # Для бинарной классификации преобразуем labels в float
            labels = labels.float().unsqueeze(1)
            loss = self.criterion(logits, labels)
            preds = torch.sigmoid(logits) > 0.5
            preds = preds.long().squeeze(1)
            labels = labels.squeeze(1).long()
        else:
            loss = self.criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
        
        # Обновление метрик
        acc = self.train_accuracy(preds, labels)
        
        # Логирование
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Шаг валидации"""
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        
        # Вычисление loss
        if self.hparams.num_classes == 2:
            labels_float = labels.float().unsqueeze(1)
            loss = self.criterion(logits, labels_float)
            preds = torch.sigmoid(logits) > 0.5
            preds = preds.long().squeeze(1)
        else:
            loss = self.criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
        
        # Обновление метрик
        self.val_accuracy(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        
        # Логирование
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Шаг тестирования"""
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        
        # Вычисление loss
        if self.hparams.num_classes == 2:
            labels_float = labels.float().unsqueeze(1)
            loss = self.criterion(logits, labels_float)
            preds = torch.sigmoid(logits) > 0.5
            preds = preds.long().squeeze(1)
        else:
            loss = self.criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
        
        # Обновление метрик
        self.test_accuracy(preds, labels)
        
        # Логирование
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Настройка оптимизатора и scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


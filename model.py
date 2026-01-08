"""
Модель для классификации ИИ-изображений
Использует предобученный backbone для извлечения признаков
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights


class AIImageClassifier(nn.Module):
    """
    Модель для классификации ИИ-изображений.
    Использует предобученный EfficientNet или ResNet как backbone.
    """
    
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        """
        Args:
            backbone_name: Название backbone модели ('efficientnet_b0', 'resnet50')
            num_classes: Количество классов (2 для бинарной классификации: AI/Real)
            dropout: Dropout вероятность
            pretrained: Использовать ли предобученные веса
        """
        super(AIImageClassifier, self).__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Выбор backbone модели
        if backbone_name == "efficientnet_b0":
            if pretrained:
                self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
            # Заменяем классификатор
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == "resnet50":
            if pretrained:
                self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                self.backbone = models.resnet50(weights=None)
            # Заменяем классификатор
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Входные изображения [batch_size, 3, height, width]
            
        Returns:
            Логиты для классов [batch_size, num_classes]
        """
        # Извлечение признаков
        features = self.backbone(x)
        
        # Классификация
        logits = self.classifier(features)
        
        return logits
    
    def extract_features(self, x):
        """
        Извлечение признаков без классификации
        
        Args:
            x: Входные изображения
            
        Returns:
            Признаки [batch_size, num_features]
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features


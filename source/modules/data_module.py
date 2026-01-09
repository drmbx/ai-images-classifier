"""
PyTorch Lightning DataModule для загрузки данных
"""
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import lightning as L


class AIImageDataModule(L.LightningDataModule):
    """
    DataModule для загрузки и подготовки данных для классификации ИИ-изображений
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1
    ):
        """
        Args:
            data_dir: Путь к директории с данными
            batch_size: Размер батча
            num_workers: Количество воркеров для загрузки данных
            image_size: Размер изображений после ресайза
            train_split: Доля данных для обучения
            val_split: Доля данных для валидации
            test_split: Доля данных для тестирования
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Трансформации для обучения (с аугментацией)
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Трансформации для валидации и тестирования (без аугментации)
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self, stage=None):
        """
        Настройка датасетов для разных стадий
        
        Ожидаемая структура данных:
        data/
            train/
                ai/          # ИИ-изображения
                real/        # Реальные изображения
            val/ (опционально)
                ai/
                real/
            test/ (опционально)
                ai/
                real/
        """
        if stage == "fit" or stage is None:
            # Проверяем структуру данных
            train_dir = self.data_dir / "train"
            
            if train_dir.exists():
                # Если есть папка train, используем её
                self.train_dataset = datasets.ImageFolder(
                    root=str(train_dir),
                    transform=self.train_transform
                )
                
                # Если есть папка val, используем её
                val_dir = self.data_dir / "val"
                if val_dir.exists():
                    self.val_dataset = datasets.ImageFolder(
                        root=str(val_dir),
                        transform=self.val_transform
                    )
                else:
                    # Разделяем train на train и val
                    total_size = len(self.train_dataset)
                    val_size = int(total_size * self.val_split)
                    train_size = total_size - val_size
                    self.train_dataset, self.val_dataset = random_split(
                        self.train_dataset,
                        [train_size, val_size],
                        generator=torch.Generator().manual_seed(42)
                    )
                    # Применяем трансформации к разделенным датасетам
                    self.train_dataset.dataset.transform = self.train_transform
                    self.val_dataset.dataset.transform = self.val_transform
            else:
                # Если нет папки train, ищем все изображения в data_dir
                # Предполагаем структуру: data/ai/ и data/real/
                all_dataset = datasets.ImageFolder(
                    root=str(self.data_dir),
                    transform=self.train_transform
                )
                
                # Разделяем на train, val, test
                total_size = len(all_dataset)
                train_size = int(total_size * self.train_split)
                val_size = int(total_size * self.val_split)
                test_size = total_size - train_size - val_size
                
                self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                    all_dataset,
                    [train_size, val_size, test_size],
                    generator=torch.Generator().manual_seed(42)
                )
        
        if stage == "test" or stage is None:
            test_dir = self.data_dir / "test"
            if test_dir.exists():
                self.test_dataset = datasets.ImageFolder(
                    root=str(test_dir),
                    transform=self.val_transform
                )
            elif not hasattr(self, 'test_dataset'):
                # Если нет test папки и не был создан test_dataset ранее
                # Используем часть данных из train
                if not hasattr(self, 'train_dataset'):
                    self.setup("fit")
                
                # Разделяем train на train, val, test
                if not hasattr(self, 'test_dataset'):
                    all_dataset = datasets.ImageFolder(
                        root=str(self.data_dir / "train" if (self.data_dir / "train").exists() else self.data_dir),
                        transform=self.train_transform
                    )
                    total_size = len(all_dataset)
                    train_size = int(total_size * self.train_split)
                    val_size = int(total_size * self.val_split)
                    test_size = total_size - train_size - val_size
                    
                    _, _, self.test_dataset = random_split(
                        all_dataset,
                        [train_size, val_size, test_size],
                        generator=torch.Generator().manual_seed(42)
                    )
                    self.test_dataset.dataset.transform = self.val_transform
    
    def train_dataloader(self):
        """DataLoader для обучения"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        """DataLoader для валидации"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        """DataLoader для тестирования"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

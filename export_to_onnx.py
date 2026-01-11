"""
Экспорт модели в ONNX формат
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import onnx
import torch

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai_images_classifier.modules.lightning_module import AIImageClassifierModule


def export_to_onnx(model_path, output_path, image_size=224, opset_version=18):
    """
    Экспорт модели в ONNX формат

    Args:
        model_path: Путь к checkpoint PyTorch Lightning
        output_path: Путь для сохранения ONNX модели
        image_size: Размер входного изображения
        opset_version: Версия ONNX opset
    """
    print("🔄 Экспорт модели в ONNX...")

    # Создаем директорию если нет
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем модель из checkpoint
    print(f"📦 Загрузка модели: {model_path}")
    pl_model = AIImageClassifierModule.load_from_checkpoint(model_path)
    model = pl_model.model
    model.eval()

    # Проверяем параметры
    print("✅ Модель загружена:")
    print(f"   - Backbone: {model.backbone_name}")
    print(f"   - Классы: {model.num_classes}")
    print(f"   - Freeze backbone: {model.freeze_backbone}")

    # Создаем dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)

    # Экспорт в ONNX
    print("\n📤 Экспорт в ONNX...")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output path: {output_path}")
    print(f"   Opset version: {opset_version}")

    # Входные и выходные имена
    input_names = ["input_image"]
    output_names = ["output_logits"]

    # Экспорт
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        verbose=False,
        external_data=False,
    )

    print(f"\n✅ ONNX модель сохранена: {output_path}")

    # Проверка ONNX модели
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        # Получаем информацию о модели
        input_info = onnx_model.graph.input[0]
        output_info = onnx_model.graph.output[0]

        print("\n📊 Информация о ONNX модели:")
        print(f"   Input:  {input_info.name}")
        print(f"   Output: {output_info.name}")

        # Проверяем реальный opset
        if onnx_model.opset_import:
            for opset in onnx_model.opset_import:
                if opset.domain == "":
                    print(f"   Opset version: {opset.version}")

        print(f"   Операций: {len(onnx_model.graph.node)}")
        print(f"   Размер: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"⚠️  Ошибка при проверке ONNX: {e}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Экспорт модели в ONNX")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Путь к checkpoint модели (.ckpt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/onnx",
        help="Директория для сохранения ONNX модели",
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Размер входного изображения"
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument(
        "--copy_to_triton",
        action="store_true",
        help="Скопировать модель в Triton директорию",
    )

    args = parser.parse_args()

    # Имя файла
    model_name = Path(args.model_path).stem
    output_path = Path(args.output_dir) / f"{model_name}.onnx"

    # Экспорт
    onnx_path = export_to_onnx(
        model_path=args.model_path,
        output_path=str(output_path),
        image_size=args.image_size,
        opset_version=args.opset,
    )

    # Копируем модель в triton директорию (опционально)
    if args.copy_to_triton:
        triton_onnx_path = Path("triton/models/ai_classifier/1/model.onnx")
        triton_onnx_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(onnx_path, triton_onnx_path)
        print(f"\n📁 Модель скопирована для Triton: {triton_onnx_path}")


if __name__ == "__main__":
    main()

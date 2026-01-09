"""
Скрипт для предсказания на новых изображениях
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms

from src.ai_images_classifier.modules.lightning_module import AIImageClassifierModule


def load_image(image_path, image_size=224):
    """Загрузка и предобработка изображения"""
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Добавляем batch dimension
    return image


def predict_image(model_path, image_path, device="cpu", image_size=224):
    """
    Предсказание для одного изображения

    Args:
        model_path: Путь к checkpoint модели
        image_path: Путь к изображению
        device: Устройство для вычислений
        image_size: Размер изображения
    """
    # Загрузка модели из checkpoint
    model = AIImageClassifierModule.load_from_checkpoint(model_path)
    model.eval()
    model.to(device)

    # Загрузка изображения
    image = load_image(image_path, image_size)
    image = image.to(device)

    # Предсказание
    with torch.no_grad():
        logits = model(image)

        if model.hparams.num_classes == 2:
            probs = torch.sigmoid(logits)
            prob_ai = probs[0, 0].item() if probs.shape[1] == 1 else probs[0, 1].item()
            prob_real = 1 - prob_ai
            prediction = "AI-generated" if prob_ai > 0.5 else "Real"
        else:
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            prediction = f"Class {pred_class}"
            prob_ai = probs[0, pred_class].item()
            prob_real = 1 - prob_ai

    return prediction, prob_ai, prob_real


def main():
    parser = argparse.ArgumentParser(description="Предсказание на изображениях")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Путь к checkpoint модели"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Путь к изображению"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Устройство для вычислений",
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Размер изображения"
    )

    args = parser.parse_args()

    # Проверка доступности GPU
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, используется CPU")
        args.device = "cpu"

    # Предсказание
    prediction, prob_ai, prob_real = predict_image(
        args.model_path, args.image_path, args.device, args.image_size
    )

    print(f"\nИзображение: {args.image_path}")
    print(f"Предсказание: {prediction}")
    print(f"Вероятность AI-generated: {prob_ai:.4f}")
    print(f"Вероятность Real: {prob_real:.4f}")


if __name__ == "__main__":
    main()

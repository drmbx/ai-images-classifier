#!/usr/bin/env python3
"""
Основной скрипт для инференса через Triton Server.
Использование: python triton_predict.py --image photo.jpg
"""

import argparse
from src.ai_images_classifier.triton.triton_client import TritonImageClassifier


def main():
    # Простой интерфейс, похожий на predict.py
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Путь к изображению")
    parser.add_argument("--url", default="localhost:8000", help="URL Triton сервера")
    parser.add_argument("--model", default="ai_classifier", help="Имя модели")

    args = parser.parse_args()

    # Используем высокоуровневый клиент
    client = TritonImageClassifier(args.url, args.model)
    result = client.predict(args.image)

    # Красивый вывод
    print("\n📊 Результат классификации:")
    print(f"   Изображение:   {args.image}")
    print(f"   Предсказание:  {result['prediction']}")
    print(f"   Вероятность AI:  {result['ai_prob']:.3f}")
    print(f"   Вероятность Real: {result['real_prob']:.3f}")
    print(f"   Уверенность:   {result['confidence']:.1%}")


if __name__ == "__main__":
    main()

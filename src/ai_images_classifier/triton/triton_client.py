"""
Основной клиент Triton Inference Server
"""

from typing import Dict, Optional

import numpy as np
import torchvision.transforms as transforms
import tritonclient.http as httpclient
from PIL import Image


class TritonImageClassifier:
    """Высокоуровневый клиент для классификации изображений"""

    def __init__(self, url: str = "localhost:8000", model_name: str = "ai_classifier"):
        self.client = httpclient.InferenceServerClient(url=url)
        self.model_name = model_name
        self.transform = self._get_default_transform()

    def _get_default_transform(self):
        """Трансформации, идентичные тренировочным"""
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image_path: str) -> Dict:
        """Основной метод предсказания"""
        # 1. Препроцессинг
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).numpy().astype(np.float32)

        # 2. Подготовка запроса
        inputs = [httpclient.InferInput("input_image", tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(tensor)
        outputs = [httpclient.InferRequestedOutput("output_logits")]

        # 3. Инференс
        response = self.client.infer(self.model_name, inputs, outputs=outputs)

        # 4. Постпроцессинг
        return self._postprocess(response)

    def _postprocess(self, response) -> Dict:
        """Преобразование ответа Triton в читаемый формат"""
        logits = response.as_numpy("output_logits")
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        return {
            "prediction": "AI" if probs[0, 1] > 0.5 else "Real",
            "ai_prob": float(probs[0, 1]),
            "real_prob": float(probs[0, 0]),
            "confidence": float(max(probs[0, 0], probs[0, 1])),
            "logits": logits[0].tolist(),
        }

    def is_server_ready(self) -> bool:
        """Проверка доступности сервера"""
        return self.client.is_server_ready()

    def is_model_ready(self) -> bool:
        """Проверка готовности модели"""
        return self.client.is_model_ready(self.model_name)

    def get_model_config(self) -> Optional[Dict]:
        """Получение конфигурации модели"""
        try:
            return self.client.get_model_config(self.model_name)
        except Exception:
            return None

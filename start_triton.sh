#!/bin/bash

echo "🚀 Запуск Triton Inference Server"

# Проверяем модель
if [ ! -f "triton/models/ai_classifier/1/model.onnx" ]; then
    echo "❌ Модель не найдена в triton/models/ai_classifier/1/model.onnx"
    echo "   Сначала подготовь: python prepare_triton.py"
    exit 1
fi

# Проверяем конфиг
if [ ! -f "triton/models/ai_classifier/config.pbtxt" ]; then
    echo "❌ Конфиг не найден"
    exit 1
fi

echo "✅ Модель найдена: $(ls -la triton/models/ai_classifier/1/model.onnx)"

# Запускаем Triton с монтированием моделей
echo "🐳 Запуск Triton из официального образа..."
echo "📁 Модели монтируются из: $(pwd)/triton/models"

docker run \
    --rm \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v $(pwd)/triton/models:/models \
    nvcr.io/nvidia/tritonserver:25.12-py3 \
    tritonserver \
    --model-repository=/models \
    --strict-model-config=false \
    --log-verbose=1
"""
Подготовка модели для Triton
"""

import shutil
import sys
from pathlib import Path

# Находим последний checkpoint
checkpoints = list(Path("checkpoints").glob("*.ckpt"))
if not checkpoints:
    print("❌ Нет checkpoint файлов")
    sys.exit(1)

latest_checkpoint = sorted(checkpoints)[-1]
print(f"📦 Используем checkpoint: {latest_checkpoint}")

# Ищем ONNX модель
onnx_files = list(Path("models/onnx").glob("*.onnx"))
if not onnx_files:
    print("❌ Нет ONNX моделей. Сначала экспортируй:")
    print("   python export_to_onnx.py --model_path checkpoints/best-model.ckpt")
    sys.exit(1)

latest_onnx = sorted(onnx_files)[-1]

# Создаем структуру
model_dir = Path("triton/models/ai_classifier/1")
model_dir.mkdir(parents=True, exist_ok=True)

# Копируем модель
shutil.copy(latest_onnx, model_dir / "model.onnx")
print(f"✅ Модель скопирована в: {model_dir}/model.onnx")

# Создаем конфиг если нет
config_file = Path("triton/models/ai_classifier/config.pbtxt")
if not config_file.exists():
    config_file.parent.mkdir(parents=True, exist_ok=True)

    config = """name: "ai_classifier"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {
    name: "input_image"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "output_logits"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 100
}
"""

    with open(config_file, "w") as f:
        f.write(config)

    print(f"✅ Конфиг создан: {config_file}")

print("✅ Triton модель подготовлена!")

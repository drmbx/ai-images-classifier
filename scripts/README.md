# Скрипты для разработки

## lint.sh

Скрипт для проверки кода с помощью ruff.

**Использование:**
```bash
# Linux/Mac
./scripts/lint.sh

# Windows (PowerShell)
bash scripts/lint.sh
```

Выполняет:
- `ruff check .` - проверка кода
- `ruff format --check .` - проверка форматирования

## format.sh

Скрипт для форматирования кода с помощью ruff.

**Использование:**
```bash
# Linux/Mac
./scripts/format.sh

# Windows (PowerShell)
bash scripts/format.sh
```

Выполняет:
- `ruff format .` - форматирование кода
- `ruff check --fix .` - автоматическое исправление проблем

## Альтернатива: Makefile

Вместо скриптов можно использовать Makefile:

```bash
make lint    # Проверка кода
make format  # Форматирование
make fix     # Исправление
make check   # Проверка (lint + format check)
```


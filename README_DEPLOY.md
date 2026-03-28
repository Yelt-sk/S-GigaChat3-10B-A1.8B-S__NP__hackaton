# README_DEPLOY — что нужно от человека и как развернуть решение

Ниже — максимально практичный чеклист. Я постарался оставить человеку только то, что невозможно сделать заранее в репозитории (доступ к GPU/модели/секретам).

## 1. Что нужно от человека (минимум)

1. Доступ к машине с GPU (целевой профиль: A100 80GB, как в условиях).
2. Локальный путь к модели `ai-sage/GigaChat3-10B-A1.8B-bf16`.
3. (Опционально) доступ к приватным весам/реестру, если модель хранится не локально.

Все остальное (установка зависимостей, запуск пайплайна, оценка) выполняется командами ниже.

---

## 2. Подготовка окружения

## 2.1 Системные требования
- Linux x86_64
- Python 3.10+ (рекомендую 3.11)
- CUDA toolkit, совместимый с вашей версией PyTorch
- GPU-драйверы

## 2.2 Создание окружения

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2.3 Установка зависимостей

Если есть `requirements.txt`:

```bash
pip install -r requirements.txt
```

Если файла пока нет, минимальный набор:

```bash
pip install torch transformers scikit-learn numpy scipy tqdm pyyaml
```

(При переносе baseline в `src/` рекомендуется зафиксировать версии в `requirements.txt`.)

---

## 3. Что должен проверить человек перед запуском

1. Что файл данных доступен:
   - `knowledge_bench_public.csv`
   - Это же значение используется как `--data-path` по умолчанию в `src/train.py`.
2. Что модель существует локально, например:
   - `$HOME/models/GigaChat3-10B-A1.8B-bf16`
3. Что хватает VRAM для выбранного batch size.

Проверка CUDA:

```bash
python - <<'PY'
import torch
print('cuda_available=', torch.cuda.is_available())
print('device_count=', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device_name=', torch.cuda.get_device_name(0))
PY
```

---

## 4. Запуск (целевой сценарий)

Ниже пример для будущих скриптов (`src/train.py`, `src/infer.py`) после переноса baseline из ноутбука.

## 4.1 Обучение детектора

```bash
python -m src.train \
  --data-path knowledge_bench_public.csv \
  --model-dir "$HOME/models/GigaChat3-10B-A1.8B-bf16" \
  --output-dir artifacts/run_001 \
  --probe-layers 0 5 10 15 20 25 \
  --seed 42
```

Если файл не найден, скрипт завершится ранней и понятной ошибкой `FileNotFoundError` с подсказкой передать корректный `--data-path`.

Ожидаемые артефакты:
- `artifacts/run_001/model.pkl`
- `artifacts/run_001/metrics.json`
- `artifacts/run_001/config.yaml`
- `artifacts/run_001/latency.json`

## 4.2 Инференс

```bash
python -m src.infer \
  --model-dir "$HOME/models/GigaChat3-10B-A1.8B-bf16" \
  --detector-path artifacts/run_001/model.pkl \
  --input-file input.jsonl \
  --output-file output.jsonl
```

Формат `input.jsonl` (пример строки):

```json
{"prompt":"...","model_answer":"..."}
```

Формат `output.jsonl` (пример строки):

```json
{"hallucination_score":0.731,"is_hallucination":true}
```

---

## 5. Контроль качества и скорости (обязательно)

1. Качество:
   - проверить `PR-AUC` на валидации.
2. Скорость:
   - проверить `avg/p95 latency`;
   - отдельно отметить, учитывался ли forward pass базовой модели согласно правилам задачи.

Целевой контрольный шаблон отчета:
- `PR-AUC`:
- `Latency avg (ms)`:
- `Latency p95 (ms)`:
- `Hardware`:
- `Torch/CUDA versions`:

---

## 6. Что делать при типовых проблемах

1. **CUDA OOM**
   - уменьшить batch size;
   - включить fp16/bf16;
   - сократить число probe-слоев.

2. **Медленный инференс**
   - профилировать extraction vs classifier;
   - упростить признаки;
   - включить батчирование и кэш токенизации.

3. **Плавающие метрики**
   - зафиксировать seed;
   - сохранить сплиты;
   - проверить одинаковость preprocessing.

---

## 7. Что человек НЕ должен делать вручную

- Ручной разметки данных не требуется.
- Ручной проверки каждого ответа не требуется.
- Внешние API для факт-чекинга подключать не требуется (и в рамках задачи это ограничено).

---

## 8. Чеклист перед сдачей

- [ ] Скрипты train/infer запускаются без ноутбука.
- [ ] Есть артефакты и конфиг запуска.
- [ ] Посчитаны PR-AUC и latency.
- [ ] Ограничения задачи соблюдены.
- [ ] Инструкция воспроизводима на чистой машине.

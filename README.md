# Captcha-MAI — финальная PP-OCRv6 small

В проекте одна модель, один ноутбук и одна точка запуска.
Единственная ветка — `codex/initial-experiments`.

**PP-OCRv6 small**, CPU, распознавание заранее вырезанной строки. Детектор текста,
другие модели, Gemini, платные API, архивы и старые экспериментальные ноутбуки
удалены из текущего дерева. Размеченная выборка уже лежит обычными изображениями.

Подтверждённая точность: **1876/1984 = 94,56%** на публичном holdout.
На семи визуально размеченных целевых изображениях — **4/7**. Этого недостаточно,
чтобы обещать качество на реальном потоке; [ограничения и выбор модели](research/REPORT.md).

## Запуск

Windows, Python 3.12. Из корня репозитория:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 scripts/run.py review
```

Просмотр результатов не загружает модель. Для нового полного локального прогона:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run.py benchmark
```

Команда проверяет SHA-256 изображений, выполняет единственный ноутбук настоящим
inference и обновляет его выводы и `research/results/`. Ядро Jupyter выбирается
автоматически из текущего Python. Первый запуск скачивает только публичные веса
распознавателя, около 21 МБ; их контрольная сумма закреплена в `research/model.json`.

Распознать свою **вырезанную область с символами**:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run.py predict path/to/crop.png
```

## Состав

- `notebooks/final_ppocr_small.ipynb` — единственный рабочий ноутбук.
- `captcha_lab/model.py` — только PP-OCRv6 small, ONNX Runtime на CPU.
- `captcha_lab/evaluation.py` — реальный прогон, exact match, CER и Wilson CI.
- `scripts/run.py` — `review`, `benchmark`, `predict`.
- `datasets/holdout/` — 1984 настоящих изображения с метками и SHA-256.
- `datasets/target/` — 9 целевых вырезок: 7 читаемых и 2 спорных.
- `research/results/` — ответы, эталоны, задержки и параметры единственной модели.
- `research/comparison.csv` — сохранённое сравнение, по которому выбрана small;
  это история исследования, а не дополнительные исполняемые модели.
- `research/REPORT.md` — итог и ограничения; `research/THEORY_AND_PAPERS.md` —
  архитектурные основания и первоисточники.
- `tests/` — проверки метрик и целостности данных.

Прямые зависимости закреплены в `requirements.txt`; точный состав проверенного
Windows-окружения — в `requirements-lock.txt`. PyTorch и альтернативные OCR-пакеты
не нужны. Тесты: `python -m unittest discover -s tests -v`.

Старое полное исследование пяти моделей и исторического Gemini сохранено в
[истории Git](https://github.com/Regqwe/Captcha-MAI/tree/43737f7).
Обучение и подстройка preprocessing на test не выполнялись. ADB, поиск области
на полном экране и подтверждение ответа сторонним сервисом не реализованы.

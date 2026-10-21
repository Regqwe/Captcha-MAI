# Captcha-MAI: размеченный benchmark шести гипотез

**Новые платные вызовы Gemini отключены по требованию пользователя.** Наличие
ключа не запускает запросы. Сохранённые результаты можно просматривать бесплатно.

Последний дополнительный пилот отменён до запуска: [статус и подготовленный план](research/results_v3/STATUS.md).

Лучший из проверенных локальных кандидатов — **PP-OCRv6 small**:
1876/1984 (94,56%) на публичном holdout, медиана 25,3 мс на CPU.
На семи размеченных исходных изображениях — 4/7; перенос на реальный поток не доказан.

- [Результаты, ограничения и решение](research/REPORT_V2.md).
- [Выбор Gemini до дальнейших расходов](research/MODEL_SELECTION.md).
- [Структура моделей и исследования](research/THEORY_AND_PAPERS.md).
- [Полный датасет: 10 000 картинок и метки](datasets/captcha-images/README.md).
- [Девять целевых изображений и разметка](datasets/target/README.md).
- [Сводные реальные измерения](research/results_v2/summary.csv).

## Шесть ноутбуков

| Гипотеза | Ноутбук | Состояние |
|---|---|---|
| H1: tiny достигает 95% | [1](notebooks/1_h1_ppocr_tiny.ipynb) | Реальный прогон 1984+9 |
| H2: small точнее tiny на 2 п.п. | [2](notebooks/2_h2_ppocr_small.ipynb) | Реальный прогон 1984+9 |
| H3: ddddocr превосходит small | [3](notebooks/3_h3_ddddocr_beta.ipynb) | Реальный прогон 1984+9 |
| H4: CRNN отстаёт от small не более чем на 2 п.п. | [4](notebooks/4_h4_crnn_base.ipynb) | Реальный прогон 1984+9 |
| H5: Transformer точнее CRNN на 2 п.п. | [5](notebooks/5_h5_conv_transformer_base.ipynb) | Реальный прогон 1984+9 |
| H6: Gemini отстаёт от small не более чем на 2 п.п. | [6](notebooks/6_h6_gemini.ipynb) | Чтение сохранённых ответов |

Первые пять содержат вывод выполненных ячеек, при повторном запуске действительно
исполняют локальную модель. Исторические шесть выполненных ноутбуков сохранены
в [research/executed_v2](research/executed_v2). У H6 новые вызовы заблокированы;
читаются реальные ответы уже состоявшейся серии. Старый пилот — `notebooks/pilot`.

## Воспроизведение без платного API

Windows, Python 3.12, CPU. Из корня репозитория:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name captcha-mai --display-name "Python (Captcha-MAI)"
.\.venv\Scripts\python.exe -X utf8 scripts\prepare_benchmark.py
.\.venv\Scripts\python.exe -X utf8 scripts\execute_notebooks.py
```

Распаковка проверяет SHA-256 полных архивов, уже находящихся в `datasets`.
Модели при первом запуске скачивают веса; платного inference API нет.
Runner всегда пропускает Gemini. H6 открывается отдельно для чтения CSV.
Тесты проверяют метрики и запрет API; они не заменяют benchmark.
Пакетирование результатов: `scripts/finalize_v2.py`, без inference.

1984 holdout-примера получены исключением 16 ранее просмотренных. Общие 200
выбраны до прогона. Метки — строки с ведущими нулями. Обучение и подбор
preprocessing на test не выполнялись. Неизвестно, входил ли публичный набор
в обучение готовых весов. ADB, автоматический поиск ROI и принятие ответа
сторонним приложением этим benchmark не проверялись.

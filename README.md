# Captcha-MAI: первичные офлайн-эксперименты

Шесть ноутбуков для сравнения распознавания изображений на обычном Windows-ПК.
Gemini подключён отдельным модулем через настраиваемый прокси. Проект не содержит
цикла автоматического прохождения проверок в стороннем приложении.

## Ноутбуки

1. `notebooks/00_data_audit.ipynb` — файлы, дубликаты, визуальная проверка ручной ROI, пустая разметка.
2. `notebooks/01_onnx_baselines.ipynb` — ddddocr default/beta и PP-OCRv6 tiny/small на CPU.
3. `notebooks/02_hf_pretrained.ipynb` — четыре обученные модели Graf-J, зафиксированные ревизии.
4. `notebooks/03_gemini_proxy.ipynb` — проверка подключения и до девяти запросов с изображениями.
5. `notebooks/04_hf_datasets.ipynb` — метаданные и ограниченные образцы публичных датасетов.
6. `notebooks/05_compare_results.ipynb` — сохранённые предсказания, время, метрики после разметки.

Исследование моделей и датасетов: [research/HUGGING_FACE.md](research/HUGGING_FACE.md).
Результаты текущего прогона и ограничения: [research/EXPERIMENT.md](research/EXPERIMENT.md).

## Установка

Проверено с Python 3.12 на Windows. Команды PowerShell из корня репозитория:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name captcha-mai --display-name "Python (Captcha-MAI)"
```

Открывайте `.ipynb` в редакторе с поддержкой Jupyter и выбирайте Python из `.venv`.
Для браузерного JupyterLab дополнительно установите `jupyterlab` и запустите
`.\.venv\Scripts\python.exe -m jupyterlab`.

Полный снимок проверенного окружения — `requirements-lock.txt`.
Для необязательного TrOCR дополнительно `pip install -r requirements-trocr.txt`.
По умолчанию большой TrOCR отключён; небольшие модели работают без Transformers.

## Данные и разметка

Поместите исходные изображения в `data/local/` либо задайте `CAPTCHA_SOURCE_DIR`
в переменных среды / `.env.local` перед запуском ноутбука 00. Изображения не
публикуются в Git. Их имена при импорте заменяются на префикс SHA-256.

Проверьте вырезки в ноутбуке 00. Текущая ROI подобрана под предоставленный макет,
это не универсальный детектор. Для готовых отдельных картинок используйте `roi=None`.

В `data/labels.csv` запишите точную строку в `label`, сохраняя ведущие нули,
и поставьте `label_status=verified` только после проверки. Для сомнительных
примеров — `ambiguous`. Пустая/непроверенная метка не участвует в точности.
Числа в CSV при чтении должны оставаться строками.

## Gemini

Скопируйте `.env.example` в `.env.local`, укажите `GEMINI_API_KEY`,
`GEMINI_PROXY_URL` и, при необходимости, `GEMINI_MODEL`. В рабочей локальной
копии эти данные уже настроены. В репозиторий и выдаваемый архив они не включены.

Поддерживается SOCKS5 с разрешением DNS через прокси (`socks5h`). При ошибке
прокси прямое подключение не выполняется. Ключ передаётся в HTTP-заголовке,
исключения и repr не содержат секретов. Alias модели разрешается сервисом;
реальная версия ответа сохраняется в результатах.

Ноутбук 03 делает один короткий текстовый запрос. Если он успешен и
`RUN_IMAGE_TEST=True`, выполняется до `MAX_IMAGES` запросов с вырезками (по умолчанию 9).
Повторный запуск расходует квоту/баланс заново. Пересылка полных экранов не нужна.

## Повторяемость и проверки

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\execute_notebooks.py --skip-api
# С реальными запросами Gemini:
.\.venv\Scripts\python.exe scripts\execute_notebooks.py --pattern 03_gemini_proxy.ipynb
```

Результаты и исполненные копии сохраняются только локально в `results/` и
`notebooks/executed/`. Модели загружаются в `models/` или cache библиотек.
Инициализация и скачивание не включены во время предсказания. У локальных моделей
есть прогрев; у Gemini прогрев выключен, чтобы не отправлять лишний запрос.

Три повтора на девяти картинках — пилотный замер. `p95_image_median_ms` означает
95-й перцентиль медиан отдельных изображений. Это не производственный p95 всех запросов.
Согласие моделей и высокая confidence не заменяют эталон. Без verified-меток
точность отображается как отсутствующая. Ошибки учитываются в знаменателе.

Сторонний Python-код из HF не выполняется. Адаптеры небольших моделей используют
локальную просмотренную архитектуру и safetensors, со строгой проверкой весов.
Оригинальные источники и ревизии: `THIRD_PARTY_NOTICES.md`, `research/hf_lock.json`.

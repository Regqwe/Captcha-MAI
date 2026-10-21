"""Package existing measured results. OFFLINE: no model/API inference is called."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import nbformat
import pandas as pd
from dotenv import dotenv_values
from captcha_lab.evaluation_v2 import all_summaries,hypothesis_results,manifest,RESULTS


def table(frame,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines += ['| '+' | '.join(str(row[c]) for c in columns)+' |' for _,row in frame.iterrows()]
    return '\n'.join(lines)


def main():
    published=ROOT/'research/results_v2';published.mkdir(exist_ok=True)
    for model in ['ppocr-tiny','ppocr-small','ddddocr-beta','crnn-base','conv-transformer-base','gemini']:
        source=RESULTS/model
        destination=published/model;destination.mkdir(exist_ok=True)
        for path in source.iterdir():
            if path.suffix in {'.csv','.json'} and '.partial.' not in path.name:
                shutil.copy2(path,destination/path.name)
    figures=published/'figures';figures.mkdir(exist_ok=True)
    shutil.copy2(RESULTS/'figures/ppocr-small-errors.png',figures/'ppocr-small-errors.png')
    summary=all_summaries();hypotheses,paired=hypothesis_results()
    summary.to_csv(published/'summary.csv',index=False)
    hypotheses.to_csv(published/'hypotheses.csv',index=False)
    paired.to_csv(published/'paired_comparisons.csv',index=False)
    serial=pd.read_csv(RESULTS/'gemini/shared_200_serial_diagnostic.csv',dtype=str,keep_default_na=False)
    accounting=dict(primary_attempts=209,completed_logged_serial_attempts=len(serial),
        possible_unlogged_inflight_at_stop=1,primary_model='gemini-3.8-flash',
        serial_status='stopped_at_user_request',billing_total='unknown; logs are not a provider invoice',
        previous_pilot_excluded=True,new_paid_calls_disabled=True)
    (published/'spending_stop.json').write_text(json.dumps(accounting,indent=2),encoding='utf-8')

    # Preserve the actual executed notebooks, including historical H6, unchanged.
    evidence=ROOT/'research/executed_v2';evidence.mkdir(exist_ok=True)
    checked=[]
    for path in sorted((ROOT/'notebooks/executed').glob('[1-6]_*.ipynb')):
        notebook=nbformat.read(path,4)
        code=[c for c in notebook.cells if c.cell_type=='code']
        assert code and all(c.execution_count is not None for c in code)
        assert not any(o.output_type=='error' for c in code for o in c.outputs)
        shutil.copy2(path,evidence/path.name)
        if not path.name.startswith('6_'):
            shutil.copy2(path,ROOT/'notebooks'/path.name)
        checked.append(dict(notebook=path.name,executed_code_cells=len(code),errors=0))
    assert len(checked)==6
    (published/'notebook_execution.json').write_text(json.dumps(checked,indent=2),encoding='utf-8')
    # H6 now only reads the archived experiment. No paid opt-in switch exists.
    h6=nbformat.read(evidence/'6_h6_gemini.ipynb',4)
    h6.cells[2].source=h6.cells[2].source.replace("print('No prediction caches: inference cell below executes the actual model.')",
        "print('READ ONLY: historical results, paid requests are disabled.')")
    h6.cells[5].source=('## Сохранённый реальный прогон: новые запросы отключены\n\n'
        'Исторический запуск уже состоялся и находится в research/executed_v2. '
        'Эта версия читает его настоящие ответы из CSV. Она не проверяет подключение, '
        'не повторяет ошибки и не отправляет изображения. Платная серия остановлена '
        'пользователем; сначала требуется выбор модели и бюджета.')
    h6.cells[6].source=("from captcha_lab.evaluation_v2 import load_predictions\n"
        "frames={name:load_predictions('gemini',name) for name in ['shared_200','target']}\n"
        "print('Saved real predictions loaded; no API calls.')")
    h6.cells[7].source=h6.cells[7].source.replace("joined.to_csv(destination,index=False)",
        "# Read-only review: do not overwrite historical results.")
    for cell in h6.cells:
        if cell.cell_type=='code':cell.execution_count=None;cell.outputs=[]
    nbformat.write(h6,ROOT/'notebooks/6_h6_gemini.ipynb')

    full=summary[summary.dataset.eq('holdout')].copy()
    rows=[]
    for item in full.itertuples():
        rows.append({'Модель':item.model,'Верно':f'{item.correct}/{item.n}',
            'Exact match':f'{item.exact_match:.2%}',
            '95% CI':f'{item.ci95_low:.2%}–{item.ci95_high:.2%}',
            'CER':f'{item.cer:.2%}','Медиана, мс':f'{item.median_ms:.1f}',
            'p95, мс':f'{item.p95_ms:.1f}'})
    fulltable=table(pd.DataFrame(rows),list(rows[0]))
    shared=summary[summary.dataset.eq('shared_200')]
    sharedtable=table(pd.DataFrame([{'Модель':r.model,'Верно':f'{r.correct}/{r.n}',
        'Exact match':f'{r.exact_match:.1%}','Ошибок выполнения':r.errors}
        for r in shared.itertuples()]),['Модель','Верно','Exact match','Ошибок выполнения'])
    ht=table(pd.DataFrame([{'Гипотеза':r.hypothesis,'Оценка':f'{r.estimate:.2%}',
        'Интервал':f'{r.ci_low:.2%}–{r.ci_high:.2%}',
        'Результат':{'supported':'поддержана','rejected':'отвергнута','inconclusive':'неопределённо'}[r.verdict]}
        for r in hypotheses.itertuples()]),['Гипотеза','Оценка','Интервал','Результат'])

    report=f'''# Выбор модели по завершённым измерениям

Дата: 21 октября 2026. Новых вызовов Gemini при подготовке этого отчёта нет.

**Текущий выбор для следующего локального этапа — PP-OCRv6 small.** На публичном
holdout: 1876/1984 полностью правильных строк (94,56%), медиана 25,3 мс на CPU.
Но на семи размеченных исходных примерах — только 4/7. Поэтому это лучший из
проверенных кандидатов, а не готовое универсальное решение для телефона.
Оптимальная Gemini-модель по сочетанию качества и цены пока НЕ установлена:
дешёвые варианты не сравнивались на нашей выборке. Выбор latest/Flash до сравнения
стоимости и массовый платный запуск были ошибкой исполнителя.

## Что реально лежит в поставке

Полные train/validation/test архивы: 6000/2000/2000 изображений из
[project-sloth/captcha-images](https://huggingface.co/datasets/project-sloth/captcha-images),
ревизия `eeaf2b6ec9086645f270f7e2aaa9ea90730683de`. Это скачанные картинки, не
мок-ответы и не preview. Метки взяты из шестизначного префикса оригинального имени.
Все 10 000 файлов читаются; точных и декодированных пиксельных дубликатов между
split нет. 966 меток начинаются с нуля; все хранятся строками.

16 ранее просмотренных test-примеров исключены. Основной holdout — **1984**;
фиксированная общая подвыборка — **200**, seed 20261021. Архивы, SHA-256,
CSV-метки, список исключений и аудит: [datasets](../datasets/captcha-images/README.md).
В отдельном `test-labeled-1984.zip` есть сами изображения и `labels.csv`.
Распознаватели получают только пиксели; имя файла и эталон в запрос не передаются.

Исходные 9 изображений отдельно размечены визуально ассистентом: 7 читаемых,
2 спорных с кандидатами ответа, исключёнными из точной оценки. Это не независимая
человеческая экспертиза; изображения ранее использовались, поэтому target —
development-проверка. [Изображения и разметка](../datasets/target/README.md).
Неизвестно, использовали ли авторы весов публичный набор для обучения.

## Локальные модели: один и тот же holdout

{fulltable}

Пять настоящих прогонов по 1984 изображения, всего 9920 holdout inference,
плюс target и отдельный прогрев. Frozen weights, без обучения на test.
Windows, Ryzen 5 220, CPU, batch=1. Время включает чтение изображения,
preprocessing, inference, decoding; инициализация отдельно в run.json.
PP-OCR/HF: 2 потока; ddddocr: настройки библиотеки. Это сравнение конкретных
конфигураций, не изолированный аппаратный benchmark архитектур.
Exact match — совпадение всей строки; CER — суммарная посимвольная ошибка.
API-отказы и пропуски входят в знаменатель, ведущие нули сохраняются.

Small заметно точнее остальных и остаётся быстрым на обычном CPU.
Tiny выигрывает лишь около 6 мс медианы, теряя 45,46 процентного пункта точности.
ddddocr beta быстрее small примерно на 7 мс, но хуже на 20,72 п.п.
CRNN самый быстрый в этом запуске, однако 33,52% недостаточно.
Conv-Transformer лучше CRNN на 10,89 п.п., но больше и медленнее small при худшей
точности. Более сложная архитектура сама по себе не обеспечивает перенос.

## Общие 200 изображений и Gemini

{sharedtable}

У Gemini первично: 114 сетевых отказов, 2 ответа неподходящего формата,
84 допустимых ответа, из которых 82 верных. 82/200 = 41% описывает этот путь
через сеть/прокси; по нему нельзя заключить, что качество самой VLM равно 41%.
Условные 82/84 нельзя выдавать за полноту решения всех 200 задач.
Медиана успешного допустимого запроса около 12,4 с; сеть и прокси входят во время.
Все 9 первичных target-запросов завершились сетевой ошибкой.

Модель первичного опыта: gemini-3.8-flash, temperature=0, output limit=1024,
4 worker. Это не сравнение всех Gemini и не доказательство оптимальности Flash.
Первичные результаты не переписаны успешными повторами.

## Расходы и остановка

Зафиксировано **209 первичных попыток и {len(serial)} завершённых записанных повторов**
в дополнительном последовательном сетевом контроле. Ещё один запрос мог быть
в полёте при остановке. Предыдущий пилот сюда не включён; отдельно был GET списка
моделей. Это число попыток, не число оплаченных ответов. При сетевой ошибке сервер
мог уже обработать запрос, поэтому по CSV невозможно восстановить точный счёт.
Для суммы списаний нужен биллинг провайдера; такую сумму мы не выдумываем.

Пользователь остановил платную работу. Процесс повторов завершён; новые POST
и generate заблокированы в клиенте, пакетный runner пропускает Gemini, H6 читает
сохранённые ответы. Никаких переключателей автоматического разрешения расходов
по наличию ключа нет. Исторические выполненные ноутбуки сохранены отдельно.
Частичный serial diagnostic не завершён и не используется как новый итог H6.

## Шесть проверенных гипотез

[Протокол](PROTOCOL_V2.md) зафиксирован отдельным commit до запусков.
Пороги 95%, 2 п.п. и задержки — предложенные инженерные критерии.
Для H1 таблица показывает accuracy, для H2–H6 — разность accuracy моделей.

{ht}

H1: tiny не достигает 95%. H2: small лучше tiny минимум на 2 п.п. и p95<150 мс.
H3: ddddocr не превосходит small. H4: CRNN не укладывается в отставание 2 п.п.
H5: Conv-Transformer превосходит CRNN минимум на 2 п.п.; разные веса/обучение
не позволяют приписать весь эффект только attention. H6: не выполнена
эксплуатационная неухудшаемость Gemini; оценка VLM отдельно от транспорта не установлена.
Парные интервалы и exact McNemar с Holm:
[paired_comparisons.csv](results_v2/paired_comparisons.csv). Численный p=0
при крайнем расхождении означает underflow и должен читаться как p<1e-300.

## Перенос и ошибки

Target: tiny 2/7, small 4/7, ddddocr 2/7, CRNN 0/7, Conv-Transformer 0/7;
Gemini 0/7 из-за сетевых отказов. Размер и происхождение разметки не позволяют
оценить производственную точность. Отдельный target-test потребуется до обещаний.

У small на публичном holdout 108 ошибок: 30 исправлялись бы удалением пунктуации,
8 — восстановлением ведущего нуля; 60 ответов неверной длины, 44 неверных
шестизначных ответа. Категории пересекаются. Эти наблюдения получены ПОСЛЕ test:
мы не применяем исправления и не пересчитываем ими исходный результат.
Примеры с настоящими эталонами и предсказаниями:

![Ошибки small](results_v2/figures/ppocr-small-errors.png)

## Теория и исследования

[Разбор структур и статей](THEORY_AND_PAPERS.md) объясняет CNN, BiLSTM,
self-attention, CTC и генеративное декодирование, размеры конкретных графов,
отличие обучающих ветвей от inference и пределы сведений о закрытой Gemini.
В частности, учтены [CTC, 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf),
[CRNN, 2015](https://arxiv.org/abs/1507.05717),
[Ye et al., CCS 2018](https://eprints.whiterose.ac.uk/id/eprint/151526/1/ccs18.pdf),
[PP-OCRv6](https://arxiv.org/abs/2606.13108) и
[COGNITION, USENIX 2026](https://www.usenix.org/conference/usenixsecurity26/presentation/wang-junyu).
Для каждого источника указаны глубина чтения и то, что именно учтено.
Результаты авторов на других данных не выданы за наши измерения.

## Решение до дальнейших экспериментов

Следующий локальный кандидат — small; сначала нужна новая независимая разметка
реальных изображений данного семейства. Публичный корпус пригоден для сравнения
реализаций, но не заменяет target-test. Обучение на целевых данных в этой работе
ещё не выполнялось. Ни real-time ADB, ни автоматический поиск ROI, ни принятие
ответа сторонним сервисом этим benchmark не проверялись.

Для Gemini сначала изучены доступность и стоимость:
[MODEL_SELECTION.md](MODEL_SELECTION.md). Новых API-прогонов не запланировано
и не разрешено. Если платная оценка понадобится позднее, до неё должны быть
выбраны модель, общий денежный предел и маленький диагностический этап;
сетевая неудача не должна автоматически запускать платный повтор.
'''
    (ROOT/'research/REPORT_V2.md').write_text(report,encoding='utf-8')
    (ROOT/'research/NETWORK_CONTROL_V2.md').write_text(
        '# Прерванный сетевой контроль\n\nПосле первичной серии был запущен post-hoc контроль: '
        'один worker и постоянная сессия, те же модель и prompt. '
        f'Сохранено {len(serial)} попыток повторной отправки транспортных отказов. '
        'Пользователь остановил работу; процесс завершён, контроль не закончен. '
        'Основные CSV не изменены. Этот частичный контроль не заменяет H6 '
        'и не доказывает причину сетевых отказов. Скрипт заблокирован.\n',encoding='utf-8')

    # Export only named deliverables, never .env, model weights or the virtualenv.
    output=ROOT.parents[1]/'outputs';output.mkdir(exist_ok=True)
    bundle=output/'Captcha-MAI-v2';bundle.mkdir(exist_ok=True)
    for directory in ['captcha_lab','datasets','notebooks','research','scripts','tests']:
        for path in (ROOT/directory).rglob('*'):
            if not path.is_file() or any(x in {'__pycache__','executed','.ipynb_checkpoints'} for x in path.parts):continue
            dest=bundle/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    for name in ['README.md','requirements.txt','requirements-lock.txt','.env.example','.gitignore','.gitattributes']:
        path=ROOT/name
        if path.exists():shutil.copy2(path,bundle/name)
    shutil.copy2(ROOT/'research/REPORT_V2.md',output/'MODEL_ANALYSIS.md')
    shutil.copy2(ROOT/'research/MODEL_SELECTION.md',output/'GEMINI_MODEL_SELECTION.md')
    # Scan bytes silently against known local secrets before any export/push.
    config=dotenv_values(ROOT/'.env.local')
    secrets=[v.encode() for k,v in config.items() if v and ('KEY' in k or 'PROXY' in k)]
    from urllib.parse import urlsplit
    proxy=urlsplit(config.get('GEMINI_PROXY_URL',''))
    secrets.extend(v.encode() for v in [proxy.password,proxy.username] if v)
    for path in bundle.rglob('*'):
        if path.is_file():
            content=path.read_bytes()
            if any(s in content for s in secrets):raise RuntimeError('Secret scan failed: '+str(path.relative_to(bundle)))
    with zipfile.ZipFile(output/'test-labeled-1984.zip','w',zipfile.ZIP_DEFLATED) as archive:
        labels=manifest('holdout').copy()
        for row in labels.itertuples():
            raw=Path(row.path).read_bytes();assert hashlib.sha256(raw).hexdigest()==row.sha256
            archive.writestr('images/'+Path(row.path).name,raw)
        labels['original_image_path']=labels.image_path
        labels['image_path']=labels.path.map(lambda p:'images/'+Path(p).name)
        archive.writestr('labels.csv',labels.drop(columns=['path']).to_csv(index=False).encode('utf-8'))
        archive.writestr('README.txt',('1984 actual labeled holdout images. Read labels.csv with dtype=str. '
            'Labels are dataset-provided six-digit filename prefixes. '
            'Source project-sloth/captcha-images revision eeaf2b6ec9086645f270f7e2aaa9ea90730683de. '
            '16 prior viewed images excluded; see full research package.').encode())
    with zipfile.ZipFile(output/'Captcha-MAI-v2.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in bundle.rglob('*'):
            if path.is_file():archive.write(path,Path(bundle.name)/path.relative_to(bundle))
    for name in ['test-labeled-1984.zip','Captcha-MAI-v2.zip']:
        with zipfile.ZipFile(output/name) as archive:assert archive.testzip() is None
    print(json.dumps(dict(output=str(output),six_executed_notebooks=checked,
        holdout_images=len(labels),paid_calls_in_packager=0,spending=accounting),indent=2))


if __name__=='__main__':main()

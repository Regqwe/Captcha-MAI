"""Local OCR: labeled evaluation, image recognition and Android screen watcher."""
import argparse
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from captcha_lab.logging_config import configure_logging

logger = logging.getLogger('captcha_lab.cli')


def review():
    from captcha_lab.evaluation import RESULTS, summary_frame

    summary = summary_frame()
    summary.to_csv(RESULTS/'summary.csv', index=False)
    columns = ['dataset', 'n', 'correct', 'exact_match', 'ci95_low', 'ci95_high', 'median_ms']
    logger.info('Сохранённые результаты оценки:\n%s', summary[columns].to_string(index=False))


def benchmark():
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from captcha_lab.data import save_json

    class CurrentPythonKernel(KernelManager):
        def format_kernel_cmd(self, extra_arguments=None):
            return [sys.executable, '-m', 'ipykernel_launcher', '-f', self.connection_file,
                    *list(extra_arguments or [])]

    path = ROOT/'notebooks/final_ppocr_small.ipynb'
    logger.info('Начат полный benchmark на размеченных изображениях',
                extra={'fields': {'notebook': path.name}})
    notebook = nbformat.read(path, 4)
    client = NotebookClient(
        notebook, timeout=600, kernel_name='python3', kernel_manager_class=CurrentPythonKernel,
        resources={'metadata': {'path': str(ROOT)}},
    )
    result = {'notebook': path.name, 'status': 'running'}
    try:
        client.execute(env={**os.environ, 'CAPTCHA_LOG_LEVEL': logging.getLevelName(logger.getEffectiveLevel())})
        result.update(status='passed', code_cells=sum(c.cell_type == 'code' for c in notebook.cells))
    except Exception as exc:
        result['status'] = type(exc).__name__
        raise
    finally:
        nbformat.write(notebook, path)
        save_json(ROOT/'research/results/notebook_validation.json', result)
    logger.info('Benchmark завершён', extra={'fields': result})
    review()


def build_parser():
    log_options = argparse.ArgumentParser(add_help=False)
    log_options.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                             default=argparse.SUPPRESS, help='Уровень журнала, по умолчанию INFO')
    log_options.add_argument('--log-file', type=Path, default=argparse.SUPPRESS,
                             help='Общий JSONL-журнал, по умолчанию runs/app.jsonl')
    parser = argparse.ArgumentParser(description=__doc__, parents=[log_options])
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('review', parents=[log_options], help='Показать сохранённую оценку модели')
    sub.add_parser('benchmark', parents=[log_options], help='Заново проверить точность и скорость на датасете')
    sub.add_parser('phone-eval', parents=[log_options], help='Распознать размеченные примеры из Drive')
    pred = sub.add_parser('predict', parents=[log_options], help='Распознать готовую вырезку')
    pred.add_argument('image', type=Path)
    sub.add_parser('phone-check', parents=[log_options], help='Проверить подключение и авторизацию ADB')
    cap = sub.add_parser('phone-capture', parents=[log_options], help='Сохранить снимок телефона и UI XML')
    cap.add_argument('--serial')
    cap.add_argument('--output', type=Path, default=ROOT/'runs/capture')
    phone = sub.add_parser('phone', parents=[log_options], help='Наблюдать и вводить ответы; Ctrl+C останавливает')
    phone.add_argument('--serial')
    phone.add_argument('--profile', type=Path, default=ROOT/'phone-profile.json')
    phone.add_argument('--observe', action='store_true', help='Распознавать без ввода и нажатий')
    phone.add_argument('--duration', type=float, default=0, help='Секунды; 0 — до Ctrl+C')
    phone.add_argument('--output', type=Path, help='Папка сеанса, по умолчанию runs/дата-время')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    log_path = configure_logging(getattr(args, 'log_level', 'INFO'), getattr(args, 'log_file', None))
    command = args.command or 'review'
    logger.info('Запуск команды %s', command, extra={'fields': {'log_file': log_path.resolve()}})
    if command == 'review':
        review()
    elif command == 'benchmark':
        benchmark()
    elif command == 'predict':
        from PIL import Image
        from captcha_lab.model import load_model

        predict = load_model()
        with Image.open(args.image) as image:
            text, score = predict(image)
        logger.info('Изображение распознано', extra={'fields': {
            'image': str(args.image), 'prediction': text, 'score': score}})
    elif command == 'phone-check':
        from captcha_lab.android import device_inventory

        devices = device_inventory()
        logger.info('Список ADB-устройств', extra={'fields': {'devices': devices}})
        if not devices:
            logger.warning('Телефон не найден. Подключите USB и разрешите отладку на телефоне.')
    elif command == 'phone-capture':
        from captcha_lab.android import capture

        capture(args.serial, args.output)
    elif command == 'phone-eval':
        from captcha_lab.android import evaluate_phone

        evaluate_phone()
    else:
        from captcha_lab.android import watch

        if args.duration < 0:
            parser.error('--duration must be >= 0')
        watch(args.serial, args.profile, observe=args.observe, duration=args.duration, output=args.output)
    logger.info('Команда %s завершена', command)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info('Команда остановлена пользователем')
        raise SystemExit(130)
    except Exception:
        logger.exception('Команда завершилась с ошибкой')
        raise SystemExit(1)

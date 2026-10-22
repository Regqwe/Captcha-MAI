"""Verify persistent logs, rotation, exception diagnostics and the real CLI."""
import io
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from captcha_lab.data import ROOT
from captcha_lab.logging_config import configure_logging, session_logging


class LoggingContracts(unittest.TestCase):
    def setUp(self):
        self.app = logging.getLogger('captcha_lab')
        self.saved = (self.app.handlers[:], self.app.level, self.app.propagate)
        self.app.handlers = []
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.console = io.StringIO()

    def tearDown(self):
        for handler in self.app.handlers[:]:
            self.app.removeHandler(handler)
            handler.close()
        self.app.handlers, level, propagate = self.saved
        self.app.setLevel(level)
        self.app.propagate = propagate
        self.temp.cleanup()

    def test_levels_utf8_structured_fields_and_traceback_survive_in_jsonl(self):
        path = configure_logging('INFO', self.folder/'app.jsonl', stream=self.console)
        log = logging.getLogger('captcha_lab.test')
        log.debug('hidden diagnostic')
        log.warning('Нечитаемая капча', extra={'fields': {'status': 'rejected_prediction', 'score': float('nan')}})
        try:
            raise ValueError('Проверка traceback')
        except ValueError:
            log.exception('Ошибка чтения')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual([row['level'] for row in rows], ['WARNING', 'ERROR'])
        self.assertEqual(rows[0]['message'], 'Нечитаемая капча')
        self.assertEqual(rows[0]['status'], 'rejected_prediction')
        self.assertIsNone(rows[0]['score'])
        self.assertTrue(rows[0]['time'].endswith('+00:00'))
        self.assertIn('Traceback', rows[1]['exception'])
        self.assertIn('ValueError: Проверка traceback', rows[1]['exception'])
        self.assertIn('WARNING', self.console.getvalue())
        self.assertNotIn('hidden diagnostic', self.console.getvalue())

    def test_rotation_is_bounded_and_reconfiguration_does_not_duplicate_records(self):
        path = self.folder/'app.jsonl'
        with patch('captcha_lab.logging_config.MAX_LOG_BYTES', 400), \
             patch('captcha_lab.logging_config.LOG_BACKUPS', 2):
            configure_logging('DEBUG', path, stream=self.console)
            configure_logging('DEBUG', path, stream=self.console)
            log = logging.getLogger('captcha_lab.test')
            for i in range(15):
                log.info('Запись %s %s', i, 'x'*80)
        files = list(self.folder.glob('app.jsonl*'))
        self.assertEqual(len(files), 3)
        messages = [json.loads(line)['message'] for f in files for line in f.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(messages), len(set(messages)))
        self.assertTrue(any(message.startswith('Запись 14 ') for message in messages))
        self.assertEqual(self.console.getvalue().count('Запись 14 '), 1)

    def test_session_log_is_closed_even_when_session_raises(self):
        configure_logging('DEBUG', self.folder/'app.jsonl', stream=self.console)
        log = logging.getLogger('captcha_lab.android')
        path = self.folder/'session/events.jsonl'
        with self.assertRaises(RuntimeError):
            with session_logging(path):
                log.info('Ответ отправлен', extra={'fields': {'status': 'submitted'}})
                raise RuntimeError('session ended')
        log.info('После сеанса')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'submitted')
        path.rename(path.with_name('closed.jsonl'))  # Windows rejects renaming an open log.

    def test_cli_failure_returns_nonzero_and_records_full_diagnostics_without_phone(self):
        path = self.folder/'failure.jsonl'
        result = subprocess.run([
            sys.executable, '-B', '-X', 'utf8', str(ROOT/'scripts/run.py'), 'phone',
            '--profile', str(self.folder/'missing.json'), '--output', str(self.folder/'session'),
            '--log-file', str(path),
        ], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=20)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, '')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        failure = rows[-1]
        self.assertEqual(failure['level'], 'ERROR')
        self.assertIn('FileNotFoundError', failure['exception'])
        self.assertIn('Profile.load', failure['exception'])


if __name__ == '__main__':
    unittest.main()

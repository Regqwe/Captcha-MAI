import unittest
from unittest.mock import patch

import pandas as pd
import requests

from captcha_lab.benchmark import evaluate, edit_distance
from captcha_lab.gemini import GeminiClient, GeminiError


class MetricsTests(unittest.TestCase):
    def test_errors_remain_in_denominator_and_zeros_are_preserved(self):
        predictions=pd.DataFrame([
            dict(model='m',variant='rgb',image_id='a',prediction='0012',error='',median_ms=2),
            dict(model='m',variant='rgb',image_id='b',prediction='',error='Timeout',median_ms=3)])
        labels=pd.DataFrame([dict(image_id='a',label='0012',label_status='verified'),
                             dict(image_id='b',label='1234',label_status='verified')])
        row=evaluate(predictions,labels).iloc[0]
        self.assertEqual(row.exact_match,.5)
        self.assertEqual(row.cer,.5)
        self.assertEqual(row.errors,1)

    def test_unverified_labels_do_not_produce_accuracy(self):
        predictions=pd.DataFrame([dict(model='m',variant='rgb',image_id='a',prediction='123',error='',median_ms=1)])
        labels=pd.DataFrame([dict(image_id='a',label='123',label_status='unlabeled')])
        row=evaluate(predictions,labels).iloc[0]
        self.assertEqual(row.verified,0)
        self.assertTrue(pd.isna(row.exact_match))

    def test_distance(self):
        self.assertEqual(edit_distance('0012','012'),1)


class SecretsTests(unittest.TestCase):
    def test_repr_hides_both_secrets(self):
        c=GeminiClient('TEST_SECRET','socks5h://user:TEST_PASSWORD@127.0.0.1:1')
        self.assertNotIn('TEST_SECRET',repr(c))
        self.assertNotIn('TEST_PASSWORD',repr(c))

    def test_network_error_is_redacted_and_has_no_direct_retry(self):
        c=GeminiClient('TEST_SECRET','socks5h://user:TEST_PASSWORD@127.0.0.1:1')
        with patch('requests.Session') as constructor:
            session=constructor.return_value.__enter__.return_value
            session.request.side_effect=requests.ConnectionError('TEST_PASSWORD TEST_SECRET')
            with self.assertRaises(GeminiError) as caught:c.generate('test')
            self.assertNotIn('TEST_SECRET',str(caught.exception))
            self.assertNotIn('TEST_PASSWORD',str(caught.exception))
            self.assertEqual(session.request.call_count,1)
            self.assertFalse(session.trust_env)


if __name__=='__main__':unittest.main()

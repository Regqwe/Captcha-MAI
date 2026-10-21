import tempfile
from pathlib import Path
import unittest
from decimal import Decimal
from captcha_lab.bounded_pilot import Ledger,BudgetStop,PRICES,CONFIG,INPUT_TOKEN_BOUND,MAX_USD,RESERVE_USD


class PaidBudgetContracts(unittest.TestCase):
    def test_tenth_attempt_exhausts_authorization_across_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ledger.json'
            ledger=Ledger(path,'fixed-plan')
            for i in range(10):ledger.reserve('m',str(i))
            restored=Ledger(path,'fixed-plan')
            with self.assertRaises(BudgetStop):restored.reserve('m','eleventh')
            self.assertEqual(len(restored.data['attempts']),10)
            self.assertLessEqual(sum(Decimal(a['reserved_usd']) for a in restored.data['attempts']),MAX_USD)

    def test_failure_keeps_reservation_and_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=Ledger(Path(folder)/'ledger.json','p')
            record=ledger.reserve('m','a');ledger.finish(record,dict(status='error'))
            self.assertEqual(record['reserved_usd'],str(RESERVE_USD))
            with self.assertRaises(BudgetStop):ledger.reserve('m','a')
            ledger.close('done')
            with self.assertRaises(BudgetStop):ledger.reserve('m','b')

    def test_price_reservations_cover_configured_tokens(self):
        for input_price,output_price in PRICES.values():
            bound=(Decimal(INPUT_TOKEN_BOUND)*Decimal(input_price)+Decimal(CONFIG['maxOutputTokens'])*Decimal(output_price))/Decimal(1_000_000)
            self.assertLess(bound,RESERVE_USD)
        self.assertLessEqual(RESERVE_USD*10,MAX_USD)

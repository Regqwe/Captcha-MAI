import unittest
import pandas as pd

from captcha_lab.evaluation import metrics, wilson


class StatisticalContracts(unittest.TestCase):
    def test_wilson_never_certifies_perfection_from_small_sample(self):
        lower,upper=wilson(7,7)
        self.assertLess(lower,.7)
        self.assertAlmostEqual(upper,1.)
        self.assertEqual(wilson(0,0),(None,None))

    def test_labeled_examples_zeros_errors_and_ambiguous_exclusion(self):
        labels=pd.DataFrame([
            dict(image_id='a',label='001234',label_status='dataset_provided'),
            dict(image_id='b',label='123456',label_status='dataset_provided'),
            dict(image_id='c',label='',label_status='ambiguous')])
        predictions=pd.DataFrame([
            dict(image_id='a',prediction='001234',error='',elapsed_ms=10),
            dict(image_id='b',prediction='',error='Network',elapsed_ms=1000),
            dict(image_id='c',prediction='123456',error='',elapsed_ms=10)])
        result=metrics(predictions,labels)
        self.assertEqual(result['n'],2)
        self.assertEqual(result['correct'],1)
        self.assertEqual(result['errors'],1)
        self.assertEqual(result['cer'],.5)
        self.assertEqual(result['exact_match'],.5)
        self.assertGreater(result['median_ms'],result['success_median_ms'])

    def test_missing_predictions_are_counted_as_failures(self):
        labels=pd.DataFrame([dict(image_id='a',label='000000',label_status='dataset_provided')])
        predictions=pd.DataFrame(columns=['image_id','prediction','error','elapsed_ms'])
        result=metrics(predictions,labels)
        self.assertEqual(result['missing'],1)
        self.assertEqual(result['exact_match'],0.)


if __name__=='__main__':unittest.main()
